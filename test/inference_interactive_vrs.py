# -*- coding: utf-8 -*-
# @FileName: inference_interactive_vrs.py
# @Time    : 22/9/25 22:37
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

import cv2
import numpy as np
from glob import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from tqdm import tqdm
from scipy.ndimage import label
from utils import *
from natsort import natsorted

dataset_image_path = '/workspace/RobotSeg/dataset/VRS/test/image'
dataset_anno_path = '/workspace/RobotSeg/dataset/VRS/test/mask_gt'
dataset_gt_path = '/workspace/RobotSeg/dataset/VRS/test/mask_gt_info'

dataset_dir = sorted([
    d for d in os.listdir(dataset_image_path)
    if os.path.isdir(os.path.join(dataset_image_path, d)) and not d.startswith('.')
])

def _save_mask(path_, mask_np_):
    cv2.imwrite(path_, mask_np_)

def guided_refine_mask(mask, image,
                       small_radius=3,
                       large_radius=7,
                       eps=1e-3,
                       band_width=3,
                       thick_thresh=6.0,
                       area_change_thresh=0.03
                       ):
    """Boundary-only adaptive guided refinement.

    Strategy:
        1) Only refine pixels in a narrow boundary band.
        2) Run guided filter with two radii (small / large).
        3) Use local thickness to blend them:
           - thin regions -> prefer small radius
           - thick regions -> prefer large radius
        4) Fallback to original mask if area changes too much.
    """
    if image is None:
        return mask.astype(np.uint8)

    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() <= 1:
        mask_uint8 = mask_uint8 * 255

    # Empty mask: nothing to refine
    if np.count_nonzero(mask_uint8) == 0:
        return mask_uint8

    # Guide image
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    src = mask_uint8.astype(np.float32) / 255.0

    # Two guided-filter results
    refined_small = cv2.ximgproc.guidedFilter(
        guide=guide, src=src, radius=small_radius, eps=eps
    )
    refined_large = cv2.ximgproc.guidedFilter(
        guide=guide, src=src, radius=large_radius, eps=eps
    )

    # Build boundary band from original mask
    k = 2 * band_width + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    boundary_band = cv2.subtract(dilated, eroded) > 0

    # Estimate local thickness:
    # distance transform itself is small on boundary, so use local maximum
    # around each pixel to reflect whether this boundary belongs to a thick part.
    binary_fg = (mask_uint8 > 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(binary_fg, cv2.DIST_L2, 5)

    # Local max thickness around each pixel
    local_ksize = max(3, 2 * large_radius + 1)
    local_thickness = cv2.dilate(
        dist_map, np.ones((local_ksize, local_ksize), np.uint8), iterations=1
    )

    # Blend weight: 0 -> small radius, 1 -> large radius
    # local_thickness >= thick_thresh means "thick region"
    alpha = np.clip((local_thickness - 2.0) / max(thick_thresh - 2.0, 1e-6), 0.0, 1.0)

    refined_soft = (1.0 - alpha) * refined_small + alpha * refined_large
    refined_bin = (refined_soft > 0.5).astype(np.uint8) * 255

    # Only replace boundary band; keep confident interior/exterior unchanged
    final_mask = mask_uint8.copy()
    final_mask[boundary_band] = refined_bin[boundary_band]

    # Safety check: if area changes too much, keep original
    orig_area = np.count_nonzero(mask_uint8)
    new_area = np.count_nonzero(final_mask)
    if orig_area > 0:
        rel_change = abs(new_area - orig_area) / float(orig_area)
        if rel_change > area_change_thresh:
            return mask_uint8

    return final_mask

def process_sequences(args_settings, gpu_id, seq_list):

    save_pool = ThreadPoolExecutor(max_workers=16)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    torch.cuda.set_device(0)

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    from robotseg.build_robotseg import build_robotseg_video_predictor

    max_frame = args_settings.max_frame
    IOU_THRESHOLD = args_settings.iou
    print(f"IOU_THRESHOLD: {IOU_THRESHOLD}")
    category = args_settings.category
    num_click = args_settings.num_click
    ckpt = args_settings.ckpt
    yaml = args_settings.yaml
    save_dir_name = args_settings.save_dir_name
    guided_filter = args_settings.guided_filter

    model_cfg = f"../robotseg/configs/{yaml}"
    checkpoint = f"../checkpoints/{ckpt}.pt"
    save_path = f"./output_interactive/{save_dir_name}/Interactive_VRSTest_{ckpt}_{num_click}click_{max_frame}maxframe_{IOU_THRESHOLD}iou"

    predictor = build_robotseg_video_predictor(model_cfg, checkpoint)

    category2id = {"arm": "000", "gripper": "001", "robot": "002"}
    instance_id_str = category2id.get(category.lower(), None)
    if instance_id_str is None:
        raise ValueError(f"Unknown category: {category}")

    results_record = {}

    for n_video, seq_name in enumerate(tqdm(seq_list, desc=f"GPU {gpu_id}: ")):

        seq_path = os.path.join(dataset_image_path, seq_name)
        frame_name = natsorted(os.listdir(seq_path))
        frame_name = [i.split('.')[0] for i in frame_name]

        gt_mask_path = os.path.join(dataset_gt_path, seq_name)
        instance_list = [f"{instance_id_str}.npy"]  # only process the selected instance

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for save_id in instance_list:
                save_id = save_id.split('.')[0]

                instance_save_path = os.path.join(save_path, seq_name, save_id)
                if os.path.exists(instance_save_path) and len(os.listdir(instance_save_path)) != 0:
                    continue
                else:
                    os.makedirs(instance_save_path,exist_ok=True)

                state = predictor.init_state(
                    video_path=seq_path,
                    async_loading_frames=False,
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                )
                anno_point_record = []
                gt_mask = np.load(os.path.join(gt_mask_path, save_id + '.npy'), allow_pickle=True).item()
                gt_mask = check_mask(gt_mask)

                start_idx = 0
                for k,v in gt_mask.items():
                    if v[0].any():
                        start_idx = k
                        break

                interact_points, gt_state = select_interact_point_center2(gt_mask[start_idx][0])

                anno_point_record.append(interact_points['0'])

                prompt_iou = []
                point = interact_points['0']
                video_segments = {}

                input_points = torch.tensor(point.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                input_labels = torch.tensor([[1]],dtype=torch.int32)

                frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                               frame_idx=start_idx,
                                                                               obj_id=int(0),
                                                                               points=input_points,
                                                                               labels=input_labels,
                                                                               robots=category,
                                                                               )

                video_segments[0] = {
                    out_obj_id: (masks[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(object_ids)
                }

                frame_iou = cal_maskIoU(video_segments[0][0], gt_mask[start_idx][0])

                prompt_iou.append(frame_iou.item())

                for i in range(num_click-1):

                    point, label = get_next_point(torch.tensor(gt_mask[start_idx][0]).unsqueeze(0).unsqueeze(0),
                                                  torch.tensor(video_segments[0][0]).unsqueeze(0),'center')

                    anno_point_record.append(np.array(point[0][0]))

                    input_points = torch.cat((input_points,point),dim=1)
                    input_labels = torch.cat((input_labels,label),dim=1)

                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                                   frame_idx=start_idx,
                                                                                   obj_id=0,
                                                                                   points=input_points,
                                                                                   labels=input_labels,
                                                                                   robots=category,
                                                                                   )
                    video_segments = {}
                    video_segments[0] = {out_obj_id: (masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(object_ids)}

                    frame_iou = cal_maskIoU(video_segments[0][0], gt_mask[start_idx][0])
                    prompt_iou.append(frame_iou.item())

                max_idx = np.argmax(prompt_iou).item()
                input_points = input_points[:, :max_idx+1, :]
                input_labels = input_labels[:, :max_idx+1]
                frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                               frame_idx=start_idx,
                                                                               obj_id=0,
                                                                               points=input_points,
                                                                               labels=input_labels,
                                                                               robots=category,
                                                                               )

                gpu_masks = []
                frame_indices = []

                video_segments = {}
                IoU_segments = {}
                F1_segments = {}
                JF_segments = {}
                corrected_frame_set = set()
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state=state, robot=category):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                    IoU_segments[out_frame_idx] = {}
                    F1_segments[out_frame_idx] = {}
                    JF_segments[out_frame_idx] = {}

                    for instance_id in video_segments[out_frame_idx]:
                        if out_frame_idx not in gt_mask.keys():
                            continue
                        if not gt_mask[out_frame_idx][int(instance_id)].any():
                            continue
                        IoU_segments[out_frame_idx][instance_id] = cal_maskIoU(video_segments[out_frame_idx][instance_id], gt_mask[out_frame_idx][int(instance_id)]).item()
                        F1_segments[out_frame_idx][instance_id] = cal_F1score(video_segments[out_frame_idx][instance_id], gt_mask[out_frame_idx][int(instance_id)])
                        JF_segments[out_frame_idx][instance_id] = (F1_segments[out_frame_idx][instance_id] + IoU_segments[out_frame_idx][instance_id])/2

                        if IoU_segments[out_frame_idx][instance_id] < IOU_THRESHOLD and len(corrected_frame_set) < max_frame:
                            orig_mask = video_segments[out_frame_idx][instance_id]
                            orig_iou = cal_maskIoU(orig_mask, gt_mask[out_frame_idx][0])

                            prompt_iou = []
                            click_points = []
                            click_labels = []

                            tmp_pred = orig_mask

                            for ii in range(num_click):
                                point, label = get_next_point(
                                    torch.tensor(gt_mask[out_frame_idx][0]).unsqueeze(0).unsqueeze(0),
                                    torch.tensor(tmp_pred).unsqueeze(0),
                                    'center'
                                )
                                click_points.append(point)
                                click_labels.append(label)
                                cur_input_points = torch.cat(click_points, dim=1)
                                cur_input_labels = torch.cat(click_labels, dim=1)
                                frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                                    inference_state=state,
                                    frame_idx=out_frame_idx,
                                    obj_id=0,
                                    points=cur_input_points,
                                    labels=cur_input_labels,
                                    robots=category,
                                )
                                mask_np = (masks[0] > 0.0).cpu().numpy()
                                cur_iou = cal_maskIoU(mask_np, gt_mask[out_frame_idx][0])
                                prompt_iou.append(cur_iou)
                                tmp_pred = mask_np

                            max_idx = np.argmax(prompt_iou)
                            best_points = torch.cat(click_points[:max_idx + 1], dim=1)
                            best_labels = torch.cat(click_labels[:max_idx + 1], dim=1)

                            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=out_frame_idx,
                                obj_id=0,
                                points=best_points,
                                labels=best_labels,
                                robots=category,
                            )
                            best_mask = (masks[0] > 0.0).cpu().numpy()
                            new_iou = cal_maskIoU(best_mask, gt_mask[out_frame_idx][0])

                            if new_iou >= orig_iou:
                                print(f"{len(corrected_frame_set)}th correction on {out_frame_idx} frame bring better segmentation ^_^ {orig_iou:.2f} --> {new_iou:.2f}")
                                video_segments[out_frame_idx][instance_id] = best_mask
                                predictor.propagate_in_video_preflight(state)
                                corrected_frame_set.add(out_frame_idx)
                            else:
                                print(f"{len(corrected_frame_set)}th correction on {out_frame_idx} frame did not bring better segmentation {orig_iou:.2f} --- {new_iou:.2f}")
                                video_segments[out_frame_idx][instance_id] = orig_mask

                        IoU_segments[out_frame_idx][instance_id] = cal_maskIoU(video_segments[out_frame_idx][instance_id],gt_mask[out_frame_idx][int(instance_id)]).item()
                        F1_segments[out_frame_idx][instance_id] = cal_F1score(video_segments[out_frame_idx][instance_id],gt_mask[out_frame_idx][int(instance_id)])
                        JF_segments[out_frame_idx][instance_id] = (F1_segments[out_frame_idx][instance_id] + IoU_segments[out_frame_idx][instance_id])/2

                    gpu_masks.append(video_segments[out_frame_idx][0][0])  # CUDA Tensor
                    frame_indices.append(out_frame_idx)

                stacked_np = (np.stack(gpu_masks, axis=0) * 255.).astype(np.uint8)

                for i, idx in enumerate(frame_indices):
                    save_path_png = os.path.join(instance_save_path, frame_name[idx] + '.png')
                    mask_to_save = stacked_np[i]
                    if guided_filter and i != 0:
                        image_path = os.path.join(seq_path, frame_name[idx] + '.jpg')
                        image = cv2.imread(image_path)
                        if image is not None:
                            mask_to_save = guided_refine_mask(mask_to_save, image)
                        else:
                            print(f"{image_path}: image not found or failed to read, skip guided refine.")

                    save_pool.submit(_save_mask, save_path_png, mask_to_save)

                IoU_list = []
                F1_list = []
                JF_list = []

                for frame_id in IoU_segments.keys():
                    if IoU_segments[frame_id] == {}:
                        continue
                    IoU_list.append(IoU_segments[frame_id][int(0)])
                    F1_list.append(F1_segments[frame_id][int(0)])
                    JF_list.append(JF_segments[frame_id][int(0)])

                save_name = seq_name+'_id_'+str(save_id).zfill(3)

                results_record[save_name] = {
                                            'maskIoU': np.mean(IoU_list),
                                            'F1-score': np.mean(F1_list),
                                            'J&F': np.mean(JF_list)
                                            }

    save_pool.shutdown(wait=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--max_frame",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--iou",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--category",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--num_click",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--yaml",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--save_dir_name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--guided_filter",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
    )
    args_settings = parser.parse_args()

    mp.set_start_method('spawn')

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    available_gpu_ids = [int(x) for x in cuda_visible_devices.split(",") if x.strip() != ""]
    print(available_gpu_ids)

    gpu_sequences = [[] for _ in available_gpu_ids]
    for idx, seq_name in enumerate(dataset_dir):
        gpu_sequences[idx % len(available_gpu_ids)].append(seq_name)

    print(f"Start Interactive VRSTest {args_settings.ckpt}_{args_settings.num_click}click_{args_settings.max_frame}maxframe_{args_settings.iou}iou_{args_settings.category}...")

    processes = []
    for x, gpu_id in enumerate(available_gpu_ids):
        seq_list = gpu_sequences[x]
        if not seq_list:
            continue

        p = mp.Process(
            target=process_sequences,
            args=(args_settings, gpu_id, seq_list)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Finished Interactive VRSTest {args_settings.ckpt}_{args_settings.num_click}click_{args_settings.max_frame}maxframe_{args_settings.iou}iou_{args_settings.category}.\nSaved results in {args_settings.save_dir_name}.")
