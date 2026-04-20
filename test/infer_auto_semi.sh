#### set category
#category="arm"
#category="gripper"
category="robot"

#### set input mode
input="auto"
#input="1c"
#input="3c"
#input="bb"

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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python inference_auto_semi_vrs.py \
--category ${category} \
--input ${input} \
--ckpt ${ckpt} \
--yaml ${yaml} \
--save_dir_name ${save_dir_name}

#### RoboEngine dataset (only supports category="robot" because the dataset contains only whole-robot masks)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python inference_roboengine.py \
--category ${category} \
--input ${input} \
--ckpt ${ckpt} \
--yaml ${yaml} \
--save_dir_name ${save_dir_name}