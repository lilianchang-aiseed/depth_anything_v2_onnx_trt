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

# 0827 two rs (d435 tilt)
# python make_gt_depthanything.py \
#   --bag /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533 \
#   --vis /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533-kc_v \
#   --export-dir /home/share/bags/nx-2.0/0827/train_1_0_20260827_164533-kc_d \
#   --max-pairs 5000 \
#   --da-model depth-anything/Depth-Anything-V2-Small-hf \
#   --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
#   --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
#   --d435-depth /d435/d435_node/depth/image_rect_raw \
#   --d435-source kalibr_infra1 \
#   --d435-calib /home/share/bags/nx-2.0/0826/calib/rs_2cam_in-extrinsic/flight_data_2026_08_27-10_27_07_0-results-cam.txt \
#   --d435-depth-scale 0.001 --d435-merge fill \
#   --sync-tol 0.05 --max-fit-depth 30.0

  # 0828 
  python make_gt_depthanything.py \
    --bag /home/share/bags/nx-2.0/0828/train_1_0_20260828_154823 \
    --vis-dir /home/share/bags/nx-2.0/0828/train_1_0_20260828_154823-vis \
    --export-dir /home/share/bags/nx-2.0/0828/train_1_0_20260828_154823-data \
    --max-pairs 5000 \
    --da-model depth-anything/Depth-Anything-V2-Small-hf \
    --sky-param ./EGE_165.ncnn.param --sky-bin ./EGE_165.ncnn.bin \
    --sky-input-name in0 --sky-output-name out0 --sky-size 384 --sky-no-sigmoid \
    --left-calib   /home/share/bags/nx-2.0/0828/calib/stereo_d455_d435_extrinsic/d455-left/flight_data_2026_08_31-12_42_27_0-results-cam.txt \
    --d435-depth   /d435/d435_node/depth/image_rect_raw \
    --d435-source  kalibr_infra1 \
    --d435-calib   /home/share/bags/nx-2.0/0828/calib/stereo_d455_d435_extrinsic/d455-d435/flight_data_2026_08_31-12_28_13_0-results-cam.txt \
    --d435-depth-scale 0.001 --d435-merge fill \
    --sync-tol 0.05 --max-fit-depth 30.0