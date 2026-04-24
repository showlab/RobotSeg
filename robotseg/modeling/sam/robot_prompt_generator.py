# -*- coding: utf-8 -*-
# @FileName: robot_prompt_generator.py
# @Time    : 2025-01-11 18:34
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com
#
# Extract the foreground regions of each frame from memory features and masks.
# Apply a hierarchical two-level KMeans clustering:
#   - Level 1 (macro): divide the foreground into object_token_num regions.
#   - Level 2 (micro): further cluster each region into 4 sub-clusters and concatenate them into a 256-D feature.
# Finally, generate object_token_num prompt tokens for each frame,
# which represent regional semantic prototypes of the target object and guide the subsequent segmentation.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobotPromptGenerator(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        object_token_num: int,
        dropout: float = 0.0,
        use_cosine: bool = True,
        mask_thresh: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.object_token_num = object_token_num
        self.dropout = nn.Dropout(dropout)
        self.use_cosine = use_cosine
        self.mask_thresh = mask_thresh

        self.no_memory_prompt_embeddings_fg = nn.Embedding(1, embed_dim)

    # ---------------------------------------------------------------------- #
    # Helper functions
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _coords_grid(H: int, W: int, device) -> torch.Tensor:
        """Build a 2D coordinate grid of shape (H*W, 2), where each row is [y, x]."""
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack([yy, xx], dim=-1).reshape(-1, 2)

    @staticmethod
    def _farthest_point_sampling(coords: torch.Tensor, k: int) -> torch.Tensor:
        """Perform greedy FPS on 2D coordinates."""
        M = coords.shape[0]
        if M == 0:
            return torch.empty(0, dtype=torch.long, device=coords.device)
        k = min(k, M)
        first = torch.randint(low=0, high=M, size=(1,), device=coords.device)
        selected = [first.item()]
        dists = torch.full((M,), float("inf"), device=coords.device)
        last = coords[first]
        for _ in range(1, k):
            diff = coords - last
            dist2 = (diff * diff).sum(dim=-1)
            dists = torch.minimum(dists, dist2)
            farthest = torch.argmax(dists).item()
            selected.append(farthest)
            last = coords[farthest:farthest+1]
        return torch.tensor(selected, dtype=torch.long, device=coords.device)

    def _one_iter_kmeans(self, feats, fg_mask, init_idx, k):
        """
        Run one-pass KMeans clustering for both macro/micro levels:
        - Initialize cluster centers with FPS.
        - Assign each point to the nearest center.
        - Use the mean feature of each cluster as its prototype.
        """
        device = feats.device
        C = feats.shape[-1]
        fg_idx = torch.nonzero(fg_mask > 0, as_tuple=False).squeeze(-1)
        if fg_idx.numel() == 0:
            return None, None

        if init_idx.numel() < k:
            pad_num = k - init_idx.numel()
            pad_idx = init_idx[0].repeat(pad_num) if init_idx.numel() > 0 else torch.zeros(pad_num, dtype=torch.long, device=device)
            init_idx = torch.cat([init_idx, pad_idx], dim=0)

        centers = feats[init_idx.clamp(0, feats.shape[0] - 1)]  # (k, C)
        feats_fg = feats[fg_idx]  # (Mf, C)

        if self.use_cosine:
            feats_n = F.normalize(feats_fg, dim=-1)
            centers_n = F.normalize(centers, dim=-1)
            dists = 1.0 - feats_n @ centers_n.t()
        else:
            dists = torch.cdist(feats_fg, centers, p=2)

        assign = torch.argmin(dists, dim=-1)  # (Mf,)
        prototypes = torch.zeros(k, C, device=device)
        cluster_masks = []

        for ci in range(k):
            sel = assign == ci
            if sel.any():
                prototypes[ci] = feats_fg[sel].mean(dim=0)
                cur_idx = fg_idx[sel]
                local_mask = torch.zeros(feats.shape[0], dtype=torch.float32, device=device)
                local_mask[cur_idx] = 1.0
                cluster_masks.append(local_mask)
            else:
                prototypes[ci] = centers[min(ci, centers.shape[0] - 1)]
                cluster_masks.append(torch.zeros_like(fg_mask))

        return prototypes, cluster_masks

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        memory: torch.Tensor,           # (N, B, C)
        memory_pos_embed: torch.Tensor, # (N, B, C)
        mask_fg: torch.Tensor,          # (N, B, 1) or (N, B)
        num_obj_ptr_tokens: int = 0,
        bs: int = 1,
    ) -> torch.Tensor:
        """
        Generate robot prompt embeddings via hierarchical clustering:
        - Level 1 (macro): cluster into object_token_num regions.
        - Level 2 (micro): further cluster each region into 4 sub-clusters and concatenate them.
        Return shape: (B, num_images * object_token_num, 256).
        """
        if memory is None:
            base = self.no_memory_prompt_embeddings_fg.weight.expand(bs, self.object_token_num, -1)
            return base

        assert memory.dim() == 3
        N, B, C = memory.shape
        if mask_fg.dim() == 3:
            mask_fg = mask_fg.squeeze(-1)

        num_regions = self.object_token_num
        micro_per_region = 4
        embed_dim_final = micro_per_region * C  # e.g., 4x64=256

        if N <= num_obj_ptr_tokens:
            base = self.no_memory_prompt_embeddings_fg.weight.expand(B, num_regions, C)
            return base

        img_token_N = N - num_obj_ptr_tokens
        assert img_token_N % 4096 == 0, "N - num_obj_ptr_tokens must be divisible by 4096 (64x64 tokens per image)"
        num_images = img_token_N // 4096
        H = W = 64

        mem_img = (
            memory[:img_token_N]
            .view(num_images, 4096, B, C)
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        mask_img = (
            mask_fg[:img_token_N]
            .view(num_images, 4096, B)
            .permute(2, 0, 1)
            .contiguous()
        )
        coords = self._coords_grid(H, W, memory.device)

        all_prompts = []

        for b in range(B):
            frame_prompts = []
            for t in range(num_images):
                feats_hw = mem_img[b, t]
                mask_hw = mask_img[b, t]
                fg_mask = (mask_hw >= self.mask_thresh).float()

                if fg_mask.sum() < 1:
                    proto = self.no_memory_prompt_embeddings_fg.weight.expand(num_regions, embed_dim_final)
                    frame_prompts.append(proto)
                    continue

                coords_fg = coords[fg_mask.bool()]
                fg_global_idx = torch.nonzero(fg_mask.bool(), as_tuple=False).squeeze(-1)
                sel_idx_macro = self._farthest_point_sampling(coords_fg, num_regions)
                init_idx_macro = fg_global_idx[sel_idx_macro.clamp(0, fg_global_idx.numel() - 1)]

                # === 1: macro clustering ===
                macro_prototypes, macro_masks = self._one_iter_kmeans(feats_hw, fg_mask, init_idx_macro, num_regions)

                # === 2: micro clustering within each macro region ===
                region_tokens = []
                for i in range(num_regions):
                    local_mask = macro_masks[i]
                    if local_mask.sum() < 1:
                        local_mask = fg_mask

                    coords_local = coords[local_mask.bool()]
                    if coords_local.numel() == 0:
                        micro_proto = macro_prototypes[i].unsqueeze(0).repeat(micro_per_region, 1)
                    else:
                        sel_idx_local = self._farthest_point_sampling(coords_local, micro_per_region)
                        fg_local_idx = torch.nonzero(local_mask.bool(), as_tuple=False).squeeze(-1)
                        init_idx_local = fg_local_idx[sel_idx_local.clamp(0, fg_local_idx.numel() - 1)]
                        micro_proto, _ = self._one_iter_kmeans(feats_hw, local_mask, init_idx_local, micro_per_region)
                        if micro_proto is None:
                            micro_proto = macro_prototypes[i].unsqueeze(0).repeat(micro_per_region, 1)

                    micro_proto = F.normalize(micro_proto, dim=-1)
                    fused_region = micro_proto.reshape(1, -1)  # (1, 256)
                    region_tokens.append(fused_region)

                proto = torch.cat(region_tokens, dim=0)
                frame_prompts.append(proto)

            frame_prompts = torch.cat(frame_prompts, dim=0)
            all_prompts.append(frame_prompts)

        robot_prompt_embeddings_fg = torch.stack(all_prompts, dim=0)
        robot_prompt_embeddings_fg = self.dropout(robot_prompt_embeddings_fg)
        return robot_prompt_embeddings_fg
