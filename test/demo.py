# -*- coding: utf-8 -*-
# @FileName: demo.py
# @Time    : 21/4/26 12:03
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

import os
import glob
import cv2
import numpy as np
from natsort import natsorted
from utils import *

INPUT_DIR = "./demo_video"
OUTPUT_DIR = "./demo_video_robotseg"

CATEGORY = "robot"   # choices: "arm", "gripper", "robot"
CKPT_NAME = "robotseg"
YAML_NAME = "robotseg-infer"


def get_image_list(input_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_list = []
    for ext in exts:
        image_list.extend(glob.glob(os.path.join(input_dir, ext)))
    return natsorted(image_list)


def main():
    image_list = get_image_list(INPUT_DIR)
    if len(image_list) == 0:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import torch
    from robotseg.build_robotseg import build_robotseg_video_predictor

    torch.cuda.set_device(0)

    model_cfg = f"../robotseg/configs/{YAML_NAME}"
    checkpoint = f"../checkpoints/{CKPT_NAME}.pt"

    print("Building RobotSeg predictor...")
    predictor = build_robotseg_video_predictor(model_cfg, checkpoint)

    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Category: {CATEGORY}")
    print(f"Found {len(image_list)} images.")
    print("Running inference...")

    results = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=INPUT_DIR,
            async_loading_frames=False,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )

        predictor.add_new_robot(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            robot=CATEGORY,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state=state,
            robot=CATEGORY,
        ):
            results[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).detach().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    print("Saving overlay results...")

    for frame_idx, image_path in enumerate(image_list):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: failed to read {image_path}, skip.")
            continue

        if frame_idx in results and 0 in results[frame_idx]:
            mask = np.squeeze(results[frame_idx][0])
            mask = (mask > 0).astype(np.uint8) * 255
            mask = guided_refine_mask(mask, image)
            output = overlay_mask_blue(image, mask)
        else:
            output = image

        save_name = os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, output)

    print("Done.")


if __name__ == "__main__":
    main()

