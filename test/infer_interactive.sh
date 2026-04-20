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
#yaml="sam2.1_hiera_t"
#save_dir_name="sam2.1"
## robotseg
ckpt="robotseg"
yaml="robotseg-infer"
save_dir_name="robotseg"

#### VRS dataset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python inference_interactive_vrs.py \
--max_frame ${max_frame} \
--iou ${iou} \
--category ${category} \
--num_click ${num_click} \
--ckpt ${ckpt} \
--yaml ${yaml} \
--save_dir_name ${save_dir_name}