#### set the maximum number of correction frames
max_frame=3

#### set the IoU threshold to trigger correction
iou=0.9

#### set category
#category="arm"
#category="gripper"
category="robot"

#### set number of click
num_click=3

#### set model and save path
## sam2.1
#ckpt="sam2.1_hiera_tiny"
#save_dir_name="sam2.1"
## robotseg
ckpt="robotseg"
save_dir_name="robotseg"

#### set workers
workers=64

#### VRS dataset
prediction_name="Interactive_VRSTest_${ckpt}_${num_click}click_${max_frame}maxframe_${iou}iou"
python ../tools/evaluator.py \
--gt_root /workspace/RobotSeg/dataset/VRS/test/mask_gt \
--pred_root ./output_interactive/${save_dir_name}/${prediction_name} \
--num_processes ${workers} \
--do_not_skip_first_and_last_frame \
--category ${category}