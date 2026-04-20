#### set category
#category="arm"
#category="gripper"
category="robot"

#### set input mode
#input="robot"  # for method roboengine and roviaug
input="auto"
#input="1c"
#input="3c"
#input="bb"

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
prediction_name="Auto_Semi_VRSTest_${ckpt}_${input}"
python ../tools/evaluator.py \
--gt_root /workspace/RobotSeg/dataset/VRS/test/mask_gt \
--pred_root ./output_auto_semi/${save_dir_name}/${prediction_name} \
--num_processes ${workers} \
--do_not_skip_first_and_last_frame \
--category ${category}

#### RoboEngine dataset (only supports category="robot" because the dataset contains only whole-robot masks)
prediction_name="RoboEngineTest_${ckpt}_${input}"
python ../tools/evaluator.py \
--gt_root /workspace/RobotSeg/dataset/RoboEngine/test/mask \
--pred_root ./output_auto_semi/${save_dir_name}/${prediction_name} \
--num_processes ${workers} \
--do_not_skip_first_and_last_frame