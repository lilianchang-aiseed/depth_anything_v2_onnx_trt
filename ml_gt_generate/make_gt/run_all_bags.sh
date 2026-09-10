#!/usr/bin/env bash
# Run make_gt_depthanything_clamp.py sequentially for every MCAP under one folder.
#
# Example:
#   bash make_gt/run_all_bags.sh \
#     --bag-dir "$COMMON_SHARE/bags/nx-2.0/0901" \
#     --gt-config make_gt/gt_config_0828.yaml \
#     --out-dir out_data/0901_batch
#
# Extra clamp-generator arguments can be appended after --, for example:
#   bash make_gt/run_all_bags.sh ... -- --no-sky --fit-samples 12000

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname -- "$script_dir")"
generator_script="$script_dir/make_gt_depthanything_clamp.py"
python_bin="python3"
bag_dir="${COMMON_SHARE:+$COMMON_SHARE/bags/nx-2.0/0901}"
bag_file=""
out_dir="$project_root/out_data/0901_batch"
gt_config=""
max_pairs=""
frame_interval=""
dry_run=0
extra_args=()

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --bag-dir DIR          Recursively search DIR for .mcap files.
                         Default: $COMMON_SHARE/bags/nx-2.0/0901
  --bag-file FILE        Process exactly one MCAP instead of searching --bag-dir.
  --gt-config FILE       Required dated GT config.
  --out-dir DIR          Common output root. Default: sml/out_data/0901_batch
  --max-pairs N          Optional successful-output limit for each bag.
  --frame-interval N     Optional source-frame interval for each bag.
  --python PATH          Python executable. Default: python3
  --dry-run              Print commands without running inference.
  -h, --help             Show this help.
  -- ARGS...             Forward remaining arguments to the clamp GT generator.

Clamp defaults:
  --fit-mode isotonic-pchip
  --metric-depth-max 15
  --far-depth 19
  --model-max-depth 20

Flat output shared by all MCAP files:
  OUT_DIR/dataset/       All NPZ files
  OUT_DIR/vis/           All montage PNG files
  OUT_DIR/statistics/    One CSV per bag
  OUT_DIR/logs/          One log per bag
  OUT_DIR/completed/     Successful-bag markers
  OUT_DIR/bag_manifest.csv
EOF
}

while (($#)); do
  case "$1" in
    --bag-dir)
      bag_dir="${2:?--bag-dir requires a directory}"
      shift 2
      ;;
    --bag-file)
      bag_file="${2:?--bag-file requires an MCAP file}"
      shift 2
      ;;
    --gt-config)
      gt_config="${2:?--gt-config requires a file}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?--out-dir requires a directory}"
      shift 2
      ;;
    --max-pairs)
      max_pairs="${2:?--max-pairs requires an integer}"
      shift 2
      ;;
    --frame-interval|--max-pairs-interval)
      frame_interval="${2:?--frame-interval requires an integer}"
      shift 2
      ;;
    --python)
      python_bin="${2:?--python requires an executable}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$bag_file" && -z "$bag_dir" ]]; then
  echo "--bag-dir is required when COMMON_SHARE is not set." >&2
  exit 2
fi
if [[ -n "$bag_file" && ! -f "$bag_file" ]]; then
  echo "MCAP file does not exist: $bag_file" >&2
  exit 2
fi
if [[ -z "$bag_file" && ! -d "$bag_dir" ]]; then
  echo "Bag directory does not exist: $bag_dir" >&2
  exit 2
fi
if [[ -z "$gt_config" ]]; then
  echo "--gt-config is required; select calibration matching these bags." >&2
  exit 2
fi
if [[ ! -f "$gt_config" ]]; then
  echo "GT config does not exist: $gt_config" >&2
  exit 2
fi
if [[ ! -f "$generator_script" ]]; then
  echo "Clamp GT generator does not exist: $generator_script" >&2
  exit 2
fi
if [[ -n "$bag_file" ]]; then
  bag_file="$(realpath -- "$bag_file")"
  bag_dir="$(dirname -- "$bag_file")"
else
  bag_dir="$(realpath -- "$bag_dir")"
fi
gt_config="$(realpath -- "$gt_config")"
out_dir="$(realpath -m -- "$out_dir")"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 2
fi
if [[ -n "$max_pairs" && ! "$max_pairs" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-pairs must be a positive integer." >&2
  exit 2
fi
if [[ -n "$frame_interval" && ! "$frame_interval" =~ ^[1-9][0-9]*$ ]]; then
  echo "--frame-interval must be a positive integer." >&2
  exit 2
fi

if [[ -n "$bag_file" ]]; then
  mcap_files=("$bag_file")
else
  mapfile -d '' mcap_files < <(find "$bag_dir" -type f -name '*.mcap' -print0 | sort -z)
fi
if ((${#mcap_files[@]} == 0)); then
  echo "No .mcap files found under: $bag_dir" >&2
  exit 1
fi

echo "Bag root : $bag_dir"
echo "GT config: $gt_config"
echo "Output   : $out_dir"
echo "MCAP files: ${#mcap_files[@]}"

dataset_dir="$out_dir/dataset"
vis_dir="$out_dir/vis"
statistics_dir="$out_dir/statistics"
logs_dir="$out_dir/logs"
completed_dir="$out_dir/completed"
manifest_path="$out_dir/bag_manifest.csv"

if ((dry_run == 0)); then
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to protect a shared batch output directory." >&2
    exit 2
  fi
  mkdir -p "$dataset_dir" "$vis_dir" "$statistics_dir" "$logs_dir" "$completed_dir"
  exec 9>>"$out_dir/.run_all.lock"
  if ! flock -n 9; then
    echo "Another run_all_bags.sh is already writing to: $out_dir" >&2
    exit 1
  fi
fi

declare -A manifest_number=()
declare -A manifest_prefix=()
next_number=1

if [[ -f "$manifest_path" ]]; then
  while IFS=, read -r stored_number stored_prefix _ stored_path; do
    [[ "$stored_number" == "bag_number" || -z "$stored_number" ]] && continue
    manifest_number["$stored_path"]="$stored_number"
    manifest_prefix["$stored_path"]="$stored_prefix"
    decimal_number=$((10#$stored_number))
    ((decimal_number >= next_number)) && next_number=$((decimal_number + 1))
  done < "$manifest_path"
elif ((dry_run == 0)); then
  (set -o noclobber; printf 'bag_number,output_prefix,relative_path,bag_path\n' > "$manifest_path") || {
    echo "Cannot create manifest without overwriting: $manifest_path" >&2
    exit 1
  }
fi

make_session_label() {
  local bag_path="$1"
  local folder date_name="" cursor component child_name="" session raw
  folder="$(dirname -- "$bag_path")"
  cursor="$folder"
  session="$(basename -- "$folder")"
  while [[ "$cursor" != "/" && "$cursor" != "." ]]; do
    component="$(basename -- "$cursor")"
    if [[ "$component" =~ ^[0-9]{4}$ ]]; then
      date_name="$component"
      [[ -n "$child_name" ]] && session="$child_name"
      break
    fi
    child_name="$component"
    cursor="$(dirname -- "$cursor")"
  done
  raw="$session"
  if [[ -n "$date_name" && "$session" != "$date_name" ]]; then
    raw="${date_name}_${session}"
  fi
  REPLY="${raw//[^A-Za-z0-9_-]/_}"
}

encode_manifest_field() {
  local value="$1"
  value="${value//%/%25}"
  REPLY="${value//,/%2C}"
}

success=0
failed=0
skipped=0

for bag_path in "${mcap_files[@]}"; do
  bag_path="$(realpath -- "$bag_path")"
  relative_path="${bag_path#"$bag_dir"/}"
  encode_manifest_field "$bag_path"
  manifest_path_key="$REPLY"

  if [[ -n "${manifest_number[$manifest_path_key]+set}" ]]; then
    bag_number="${manifest_number[$manifest_path_key]}"
    output_prefix="${manifest_prefix[$manifest_path_key]}"
  else
    printf -v bag_number '%02d' "$next_number"
    ((next_number += 1))
    make_session_label "$bag_path"
    output_prefix="${REPLY}_bag${bag_number}"
    manifest_number["$manifest_path_key"]="$bag_number"
    manifest_prefix["$manifest_path_key"]="$output_prefix"
    if ((dry_run == 0)); then
      encode_manifest_field "$relative_path"
      manifest_relative_path="$REPLY"
      printf '%s,%s,%s,%s\n' \
        "$bag_number" "$output_prefix" "$manifest_relative_path" \
        "$manifest_path_key" >> "$manifest_path"
    fi
  fi

  stats_file="$statistics_dir/$output_prefix.csv"
  log_file="$logs_dir/$output_prefix.log"
  done_file="$completed_dir/$output_prefix.complete"
  running_file="$completed_dir/$output_prefix.running"

  if [[ -f "$done_file" && $dry_run -eq 0 ]]; then
    echo "[SKIP] $output_prefix already completed: $relative_path"
    ((skipped += 1))
    continue
  fi

  if ((dry_run == 0)); then
    shopt -s nullglob
    collisions=(
      "$stats_file" "$log_file" "$running_file"
      "$vis_dir/${output_prefix}_"*.png
      "$vis_dir/_review/${output_prefix}_"*.png
      "$dataset_dir/${output_prefix}_"*.npz
      "$dataset_dir/_review/${output_prefix}_"*.npz
    )
    shopt -u nullglob
    existing=()
    for candidate in "${collisions[@]}"; do
      [[ -e "$candidate" ]] && existing+=("$candidate")
    done
    if ((${#existing[@]})); then
      echo "[INCOMPLETE] Refusing to overwrite existing data for $output_prefix:" >&2
      printf '  %s\n' "${existing[@]:0:5}" >&2
      ((failed += 1))
      continue
    fi
  fi

  command=(
    "$python_bin" "$generator_script"
    --bag "$bag_path"
    --gt-config "$gt_config"
  )
  command+=("${extra_args[@]}")
  command+=(
    --vis-dir "$vis_dir"
    --export-dir "$dataset_dir"
    --output-prefix "$output_prefix"
    --stats-csv "$stats_file"
  )
  if [[ -n "$max_pairs" ]]; then
    command+=(--max-pairs "$max_pairs")
  fi
  if [[ -n "$frame_interval" ]]; then
    command+=(--frame-interval "$frame_interval")
  fi

  printf '\n[RUN %s] %s\n' "$output_prefix" "$relative_path"
  printf '  '
  printf '%q ' "${command[@]}"
  printf '\n'

  if ((dry_run)); then
    continue
  fi

  mkdir -p "$dataset_dir" "$vis_dir" "$statistics_dir" "$logs_dir" "$completed_dir"
  if ! (set -o noclobber; printf 'running %(%Y-%m-%dT%H:%M:%S%z)T\n' -1 > "$running_file") 2>/dev/null; then
    echo "[INCOMPLETE] Running marker already exists: $running_file" >&2
    ((failed += 1))
    continue
  fi

  if "${command[@]}" 2>&1 | tee "$log_file"; then
    if (set -o noclobber; printf 'completed %(%Y-%m-%dT%H:%M:%S%z)T\n' -1 > "$done_file") 2>/dev/null; then
      unlink "$running_file"
      echo "[OK] $relative_path"
      ((success += 1))
    else
      echo "[FAILED] Completion marker exists: $done_file" >&2
      ((failed += 1))
    fi
  else
    echo "[FAILED] $relative_path (see $log_file; running marker retained)" >&2
    ((failed += 1))
  fi
done

printf '\nBatch summary: success=%d failed=%d skipped=%d total=%d\n' \
  "$success" "$failed" "$skipped" "${#mcap_files[@]}"

((failed == 0))
