# -*- coding: utf-8 -*-
# @FileName: generate_pseudo_masks.py
# @Time    : 2025-10-29
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com
#
# Multi-GPU parallelization: propagate the first-frame mask of each video forward
# to generate pseudo masks. The first-frame mask is kept unchanged; when rerun,
# all non-first-frame masks are overwritten.

import os
import math
import time
import datetime
import functools
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import lovely_tensors
import mediapy as mpv
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF

lovely_tensors.monkey_patch()
torch.set_grad_enabled(False)

# ==========================================
# === Constants and function definitions ===
# ==========================================

MAX_CONTEXT_LENGTH = 7
NEIGHBORHOOD_SIZE = 12
NEIGHBORHOOD_SHAPE = "circle"
TOPK = 5
TEMPERATURE = 0.2
SHORT_SIDE = 960


class ResizeToMultiple(TVT.Resize):
    def __init__(self, short_side, multiple):
        self.short_side = short_side
        self.multiple = multiple
    def _round_up(self, side):
        return math.ceil(side / self.multiple) * self.multiple
    def __call__(self, img):
        old_w, old_h = TVTF.get_image_size(img)
        if old_w > old_h:
            new_h = self._round_up(self.short_side)
            new_w = self._round_up(old_w * new_h / old_h)
        else:
            new_w = self._round_up(self.short_side)
            new_h = self._round_up(old_h * new_w / old_w)
        return TVTF.resize(img, [new_h, new_w], interpolation=TVT.InterpolationMode.BICUBIC)


@torch.compile(disable=True)
def forward(model: torch.nn.Module, img: Tensor) -> Tensor:
    feats = model.get_intermediate_layers(img.unsqueeze(0), n=1, reshape=True)[0]
    feats = feats.movedim(-3, -1)
    feats = F.normalize(feats, dim=-1, p=2)
    return feats.squeeze(0)


@torch.compile(disable=True)
def propagate(current_features, context_features, context_probs, neighborhood_mask, topk, temperature):
    t, h, w, M = context_probs.shape
    dot = torch.einsum("ijd,tuvd->ijtuv", current_features, context_features)
    dot = torch.where(neighborhood_mask[:, :, None, :, :], dot, -torch.inf)
    dot = dot.flatten(2, -1).flatten(0, 1)
    k_th = torch.topk(dot, dim=1, k=topk).values
    dot = torch.where(dot >= k_th[:, -1:], dot, -torch.inf)
    weights = F.softmax(dot / temperature, dim=1)
    current_probs = torch.mm(weights, context_probs.flatten(0, 2))
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)
    return current_probs.unflatten(0, (h, w))


def make_neighborhood_mask(h, w, size, shape, device):
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    ord = 2 if shape == "circle" else torch.inf
    norm = torch.linalg.vector_norm(
        ij[:, :, None, None, :] - ij[None, None, :, :, :],
        ord=ord,
        dim=-1,
    )
    return norm <= size


def postprocess_probs(probs):
    vmin = probs.flatten(2, 3).min(dim=2).values
    vmax = probs.flatten(2, 3).max(dim=2).values
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    probs = torch.nan_to_num(probs, nan=0)
    return probs


def list_dirs(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def list_images(folder):
    imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    imgs.sort()
    return [os.path.join(folder, f) for f in imgs]


def read_mask(path):
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.uint8)


def save_mask(path, mask):
    mask = (mask > 0).astype(np.uint8)
    Image.fromarray(mask * 255).save(path)


# ========================================
# === Main propagation logic per video ===
# ========================================

@torch.no_grad()
def process_video(video_id, image_root, mask_root, device, model, transform):
    image_dir = os.path.join(image_root, video_id)
    mask_video_dir = os.path.join(mask_root, video_id)
    if not os.path.exists(image_dir):
        return

    obj_dirs = list_dirs(mask_video_dir)
    if len(obj_dirs) == 0:
        return

    frames = list_images(image_dir)
    num_frames = len(frames)
    if num_frames == 0:
        return

    for obj_id in obj_dirs:
        obj_dir = os.path.join(mask_video_dir, obj_id)
        masks = [f for f in os.listdir(obj_dir) if f.endswith(".png")]
        masks.sort()
        first_mask_path = os.path.join(obj_dir, masks[0])
        start_idx = int(os.path.splitext(masks[0])[0])
        first_mask_np = read_mask(first_mask_path)
        mask_h, mask_w = first_mask_np.shape
        num_masks = int(first_mask_np.max() + 1)

        frames_to_process = [f for f in frames if int(os.path.splitext(os.path.basename(f))[0]) >= start_idx]
        if len(frames_to_process) <= 1:
            continue

        first_pil = Image.open(frames_to_process[0]).convert("RGB")
        first_tensor = transform(first_pil).to(device)
        first_feats = forward(model, first_tensor)
        feats_h, feats_w = first_feats.shape[:2]
        neighborhood_mask = make_neighborhood_mask(feats_h, feats_w, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SHAPE, device)

        first_probs = F.one_hot(torch.from_numpy(first_mask_np).long(), num_masks).float().to(device)
        first_probs = F.interpolate(first_probs.movedim(-1, -3)[None], size=(feats_h, feats_w), mode="nearest").squeeze(0).movedim(-3, -1)

        features_q, probs_q = [], []
        print(f"[{video_id}] obj={obj_id} start={start_idx} frames={len(frames_to_process)}")

        for i in tqdm(range(1, len(frames_to_process)), desc=f"[{video_id}] obj={obj_id}"):
            cur_path = frames_to_process[i]
            cur_idx = int(os.path.splitext(os.path.basename(cur_path))[0])

            cur_tensor = transform(Image.open(cur_path).convert("RGB")).to(device)
            cur_feats = forward(model, cur_tensor)
            context_feats = torch.stack([first_feats, *features_q], dim=0)
            context_probs = torch.stack([first_probs, *probs_q], dim=0)

            cur_probs = propagate(cur_feats, context_feats, context_probs, neighborhood_mask, TOPK, TEMPERATURE)

            features_q.append(cur_feats)
            probs_q.append(cur_probs)
            if len(features_q) > MAX_CONTEXT_LENGTH:
                features_q.pop(0)
                probs_q.pop(0)

            cur_probs = F.interpolate(cur_probs.movedim(-1, -3)[None], size=(mask_h, mask_w), mode="nearest")
            cur_probs = postprocess_probs(cur_probs).squeeze(0)
            pred = torch.argmax(cur_probs, dim=0).cpu().numpy().astype(np.uint8)

            out_path = os.path.join(obj_dir, f"{cur_idx:05d}.png")
            if cur_idx == start_idx:
                continue
            save_mask(out_path, pred)

        print(f"[{video_id}] obj={obj_id} done.")


# =====================================
# === Main entry and GPU assignment ===
# =====================================

def process_wrapper(gpu_id, video_list, image_root, mask_root):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    DINOV3_LOCATION = "/workspace/RobotSeg/train/dinov3"
    MODEL_NAME = "dinov3_vith16plus"
    model = torch.hub.load(repo_or_dir=DINOV3_LOCATION, model=MODEL_NAME, source="local")
    model.to(device)
    model.eval()
    patch_size = model.patch_size
    print(f"[GPU {gpu_id}] Loaded DINOv3 model (patch size={patch_size})")

    transform = TVT.Compose([
        ResizeToMultiple(short_side=SHORT_SIDE, multiple=patch_size),
        TVT.ToTensor(),
        TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[GPU {gpu_id}] assigned {len(video_list)} videos.")

    for vid in video_list:
        image_dir = os.path.join(image_root, vid)
        mask_dir = os.path.join(mask_root, vid)
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            continue

        image_count = len([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])

        need_rerun = False
        for obj_name in os.listdir(mask_dir):
            obj_dir = os.path.join(mask_dir, obj_name)
            if not os.path.isdir(obj_dir):
                continue
            mask_count = len([f for f in os.listdir(obj_dir) if f.lower().endswith(".png")])
            if mask_count < image_count:
                print(f"[incomplete] {vid}/{obj_name}: image={image_count}, mask={mask_count}")
                need_rerun = True
                break

        if need_rerun:
            print(f"[rerun] {vid}")
            process_video(vid, image_root, mask_root, device, model, transform)

        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] finished.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="/workspace/RobotSeg/dataset/VRS/train/image")
    parser.add_argument("--mask_root", type=str, default="/workspace/RobotSeg/dataset/VRS/train/mask_gt")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    # =============================
    # GPU Assignment
    # =============================
    mp.set_start_method('spawn', force=True)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", args.gpus)
    available_gpu_ids = [int(x) for x in cuda_visible_devices.split(",") if x.strip() != ""]
    print(f"Available GPU: {available_gpu_ids}")

    videos = list_dirs(args.image_root)
    videos.sort()

    # ===== Set the range of videos to process =====
    start_idx, end_idx = 0, 3000
    videos = videos[start_idx:end_idx]
    print(f"Video range processed in this run: [{start_idx}, {end_idx}) → {len(videos)} videos.")

    gpu_sequences = [[] for _ in available_gpu_ids]
    for idx, vid in enumerate(videos):
        gpu_sequences[idx % len(available_gpu_ids)].append(vid)

    processes = []
    for x, gpu_id in enumerate(available_gpu_ids):
        seq_list = gpu_sequences[x]
        if not seq_list:
            continue
        p = mp.Process(target=process_wrapper, args=(gpu_id, seq_list, args.image_root, args.mask_root))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All done.")


if __name__ == "__main__":
    main()
