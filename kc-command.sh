# python make_gt_depthanything.py  \
#  --bag /home/share/bags/nx-2.0/0825/train_1_0-5 \
#  --vis-dir /home/share/bags/nx-2.0/0825/train_1_0-5-v \
#  --export-dir /home/share/bags/nx-2.0/0825/train_1_0-5-d \
#  --max-pairs 5000 \
#  --da-model depth-anything/Depth-Anything-V2-Small-hf \
#  --sky-param /home/kc/Projects/depth/Depth-Anything-V2/EGE_165.ncnn.param \
#  --sky-bin /home/kc/Projects/depth/Depth-Anything-V2/EGE_165.ncnn.bin \
#  --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#  --sky-refine \
#  --d435-depth /d435/d435_node/depth/image_rect_raw \    
#  --d435-source infra1 \
#  --d435-info /d435/d435_node/depth/camera_info \
#  --d435-depth-scale 0.001 --d435-merge fill \
#  --sync-tol 0.05 \
#  --max-fit-depth 30.0

# # 0827 two rs (d435 tilt)
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0827/train_1_0_20260827_170239 \
#   --vis /home/share/bags/nx-2.0/0827/train_1_0_20260827_170239-kc_v \
#   --export-dir /home/share/bags/nx-2.0/0827/train_1_0_20260827_170239-kc_d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0827/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0826/calib/rs_2cam_in-extrinsic/flight_data_2026_08_27-10_27_07_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --sync-tol 0.05 --max-fit-depth 30.0

#   # 0828 
#   python make_gt_depthanything.py \
#     --bag /home/share/bags/nx-2.0/0828/train_1_0_20260828_154823 \
#     --vis-dir /home/share/bags/nx-2.0/0828/train_1_0_20260828_154823-vis \
#     --export-dir /home/share/bags/nx-2.0/0828/train_1_0_20260828_154823-data \
#     --max-pairs 5000 \
#     --da-model depth-anything/Depth-Anything-V2-Small-hf \
#     --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#     --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#     --left-calib   /home/share/bags/nx-2.0/0828/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#     --d435-depth   /d435/d435_node/depth/image_rect_raw \
#     --d435-source  kalibr_infra1 \
#     --d435-calib   /home/share/bags/nx-2.0/0828/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#     --d435-depth-scale 0.001 --d435-merge fill \
#     --sync-tol 0.05 --max-fit-depth 30.0


# 0825
# v-13
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0825/train_1_0-5 \
#   --vis /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-5/v-13 \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-5/d-13 \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid --sky-refine \
#   --left-calib /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source color \
#   --d435-calib /home/share/bags/nx-2.0/0825/calib/rs_2cam_in-extrinsic/flight_data_2026_08_25-14_17_16_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --sync-tol 0.05 --max-fit-depth 30.0 

# v-14
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0825/train_1_0-1 \
#   --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-1/v \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-1/d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#   --lr-calib   /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-01.txt \
#   --targets both \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source color \
#   --d435-calib /home/share/bags/nx-2.0/0825/calib/rs_2cam_in-extrinsic/flight_data_2026_08_25-14_17_16_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --fit-mode piecewise --fit-bins 4 \
#   --sync-tol 0.05 --max-fit-depth 25.0 \
#   --qc-min-spread 1.8 --qc-min-s 0.01 --qc-min-inl 0.5

# v-18
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0825/train_1_0-5 \
#   --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-5-right/v \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-5-right/d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#   --lr-calib   /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-01.txt \
#   --targets right \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_color_via_d2c \
#   --d435-calib /home/share/bags/nx-2.0/0825/calib/rs_2cam_in-extrinsic/flight_data_2026_08_25-14_17_16_0-results-cam.txt \
#   --d435-info /d435/d435_node/depth/camera_info \
#   --d435-depth-scale 0.001 --d435-merge fill --d435-max-depth 6.0 \
#   --fit-mode affine --fit-bins 4 --extrap-near \
#   --gt-max-depth 20.0 --max-fit-depth 20.0 --dmax 20 \
#   --sync-tol 0.05 \
#   --qc-min-spread 1.25 --qc-min-s 0.005 --qc-min-inl 0.35 \
#   --stats-csv /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-5-right/qc_stats.csv


# python compare_d435_chain.py \
#  --bag /home/share/bags/nx-2.0/0825/train_1_0-1 \
#  --left-calib /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#  --lr-calib   /home/share/bags/nx-2.0/0825/calib/d455-left/calib_ir_20260814_192834-results-cam-01.txt \
#  --d435-calib /home/share/bags/nx-2.0/0825/calib/rs_2cam_in-extrinsic/flight_data_2026_08_25-14_17_16_0-results-cam.txt \
#  --d435-source kalibr_color_via_d2c \
#  --out /home/kc/Projects/dataset/depth/nx-2.0/0825/train_1_0-1/d435_chain \
#  --n 30


# 0827
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533 \
#   --vis /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533-kc_v \
#   --export-dir /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533-kc_d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0827/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0826/calib/rs_2cam_in-extrinsic/flight_data_2026_08_27-10_27_07_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --sync-tol 0.05 --max-fit-depth 30.0

# v-18
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0827/train_1_0_20260827_170239 \
#   --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0827/train_1_0_20260827_170239-right/v \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0827/train_1_0_20260827_170239-right/d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0827/calib/d455-left/calib_ir_20260814_192834-results-cam-0.txt \
#   --lr-calib   /home/share/bags/nx-2.0/0827/calib/d455-left/calib_ir_20260814_192834-results-cam-01.txt \
#   --targets right \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0827/calib/rs_2cam_in-extrinsic/flight_data_2026_08_27-10_27_07_0-results-cam.txt \
#   --d435-info /d435/d435_node/depth/camera_info \
#   --d435-depth-scale 0.001 --d435-merge fill --d435-max-depth 6.0 \
#   --fit-mode affine --fit-bins 4 --extrap-near \
#   --gt-max-depth 20.0 --max-fit-depth 20.0 --dmax 20 \
#   --sync-tol 0.05 \
#   --qc-min-spread 1.25 --qc-min-s 0.005 --qc-min-inl 0.35 \
#   --stats-csv /home/kc/Projects/dataset/depth/nx-2.0/0827/train_1_0_20260827_170239-right/qc_stats.csv

# 0901
# python make_gt_depthanything.py \
#     --bag /home/share/bags/nx-2.0/0901/train_2_1_20260901_161821_recovered/recovered.mcap \
#     --vis-dir /home/share/bags/nx-2.0/0901/train_2_1_20260901_161821-kc_vis \
#     --export-dir /home/share/bags/nx-2.0/0901/train_2_1_20260901_161821-kc_data \
#     --max-pairs 5000 \
#     --da-model depth-anything/Depth-Anything-V2-Small-hf \
#     --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#     --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#     --left-calib   /home/share/bags/nx-2.0/0901/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#     --d435-depth   /d435/d435_node/depth/image_rect_raw \
#     --d435-source  kalibr_infra1 \
#     --d435-calib   /home/share/bags/nx-2.0/0901/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#     --d435-depth-scale 0.001 --d435-merge fill \
#     --sync-tol 0.05 --max-fit-depth 30.0

# v-18
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0901/train_2_1_20260901_155219/\
#   --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0901/train_2_1_20260901_155219-right/v \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0901/train_2_1_20260901_155219-right/d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0901/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#   --lr-calib /home/share/bags/nx-2.0/0901/calib/calib_ir_20260907_121307-results-cam-2_1.txt \
#   --targets right \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0901/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#   --d435-info /d435/d435_node/depth/camera_info \
#   --d435-depth-scale 0.001 --d435-merge fill --d435-max-depth 6.0 \
#   --fit-mode affine --fit-bins 4 --extrap-near \
#   --gt-max-depth 20.0 --max-fit-depth 20.0 --dmax 20 \
#   --sync-tol 0.05 \
#   --qc-min-spread 1.25 --qc-min-s 0.005 --qc-min-inl 0.35 \
#   --stats-csv /home/kc/Projects/dataset/depth/nx-2.0/0901/train_2_1_20260901_155219-right/qc_stats.csv

# 0902
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0902/train_2_1_20260902_120036 \
#   --vis /home/share/bags/nx-2.0/0902/train_2_1_20260902_120036-kc_v \
#   --export-dir /home/share/bags/nx-2.0/0902/train_2_1_20260902_120036-kc_d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --sync-tol 0.05 --max-fit-depth 30.0

# v-18
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0902/train_2_1_20260902_121639 \
#   --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0902/train_2_1_20260902_121639-left/v \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0902/train_2_1_20260902_121639-left/d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --left-calib /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#   --lr-calib   /home/share/bags/nx-2.0/0902/calib/calib_ir_20260907_121307-results-cam-2_1.txt \
#   --targets left \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#   --d435-info /d435/d435_node/depth/camera_info \
#   --d435-depth-scale 0.001 --d435-merge fill --d435-max-depth 6.0 \
#   --fit-mode affine --fit-bins 4 --extrap-near \
#   --gt-max-depth 20.0 --max-fit-depth 20.0 --dmax 20 \
#   --sync-tol 0.05 \
#   --qc-min-spread 1.25 --qc-min-s 0.005 --qc-min-inl 0.35 \
#   --stats-csv /home/kc/Projects/dataset/depth/nx-2.0/0902/train_2_1_20260902_121639-left/qc_stats.csv

# python make_gt_depthanything.py \
#     --bag /home/share/bags/nx-2.0/0902/train_2_1_20260902_124546_recovered/recovered.mcap \
#     --vis-dir /home/share/bags/nx-2.0/0902/train_2_1_20260902_124546-kc_vis \
#     --export-dir /home/share/bags/nx-2.0/0902/train_2_1_20260902_124546-kc_data \
#     --max-pairs 5000 \
#     --da-model depth-anything/Depth-Anything-V2-Small-hf \
#     --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#     --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#     --left-calib   /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#     --d435-depth   /d435/d435_node/depth/image_rect_raw \
#     --d435-source  kalibr_infra1 \
#     --d435-calib   /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#     --d435-depth-scale 0.001 --d435-merge fill \
#     --sync-tol 0.05 --max-fit-depth 30.0

# # 0904 indoor
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0904/train_2_1_20260904_160020 \
#   --vis /home/share/bags/nx-2.0/0904/train_2_1_20260904_160020-kc_v \
#   --export-dir /home/share/bags/nx-2.0/0904/train_2_1_20260904_160020-kc_d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --left-calib /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0902/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --sync-tol 0.05 --max-fit-depth 30.0

# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0904/train_2_1_20260904_155650 \
#   --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0904/train_2_1_20260904_155650-right/v \
#   --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0904/train_2_1_20260904_155650-right/d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --left-calib /home/share/bags/nx-2.0/0904/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
#   --lr-calib   /home/share/bags/nx-2.0/0904/calib/calib_ir_20260907_121307-results-cam-2_1.txt \
#   --targets right \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0904/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
#   --d435-info /d435/d435_node/depth/camera_info \
#   --d435-depth-scale 0.001 --d435-merge fill --d435-max-depth 6.0 --d455-max-depth 30 \
#   --fit-mode affine --fit-bins 4 --extrap-near \
#   --gt-max-depth 20.0 --max-fit-depth 20.0 --dmax 20 \
#   --sync-tol 0.05 \
#   --qc-min-spread 1.3 --qc-min-s 0.0075 --qc-min-inl 0.4 \
#   --stats-csv /home/kc/Projects/dataset/depth/nx-2.0/0904/train_2_1_20260904_155650-lerightft/qc_stats.csv

  # python compare_d435_chain.py \
  # --bag /home/share/bags/nx-2.0/0904/train_2_1_20260904_155650 \
  # --left-calib /home/share/bags/nx-2.0/0904/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
  # --lr-calib /home/share/bags/nx-2.0/0904/calib/calib_ir_20260907_121307-results-cam-2_1.txt \
  # --d435-calib /home/share/bags/nx-2.0/0904/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
  # --d435-source kalibr_infra1 \
  # --out /home/kc/Projects/dataset/depth/nx-2.0/0904/train_2_1_20260904_155650/d435_chain \
  # --n 300

# 0904 walk
python make_gt_depthanything.py \
  --bag /home/share/bags/nx-2.0/0907/train_2_1_20260907_181406/recovered.mcap \
  --vis-dir /home/kc/Projects/dataset/depth/nx-2.0/0907/train_2_1_20260907_181406-left/v \
  --export-dir /home/kc/Projects/dataset/depth/nx-2.0/0907/train_2_1_20260907_181406-left/d \
  --max-pairs 5000 \
  --da-model depth-anything/Depth-Anything-V2-Small-hf \
  --left-calib /home/share/bags/nx-2.0/0907/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
  --lr-calib   /home/share/bags/nx-2.0/0907/calib/calib_ir_20260907_121307-results-cam-2_1.txt \
  --targets left \
  --d435-depth /d435/d435_node/depth/image_rect_raw \
  --d435-source kalibr_infra1 \
  --d435-calib /home/share/bags/nx-2.0/0907/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
  --d435-info /d435/d435_node/depth/camera_info \
  --d435-depth-scale 0.001 --d435-merge fill --d435-max-depth 6.0 --d455-max-depth 30 \
  --fit-mode affine --fit-bins 4 --extrap-near \
  --gt-max-depth 20.0 --max-fit-depth 20.0 --dmax 20 \
  --sync-tol 0.05 \
  --qc-min-spread 1.3 --qc-min-s 0.0075 --qc-min-inl 0.4 \
  --stats-csv /home/kc/Projects/dataset/depth/nx-2.0/0907/train_2_1_20260907_181406-left/qc_stats.csv