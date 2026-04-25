# -*- coding: utf-8 -*-
# @FileName: memory_attention_with_structure.py
# @Author  : Haiyang Mei
# @Desc    : An extension of the original MemoryAttention module with an additional structure branch.
#            The branch extracts Canny edges from the raw image, downsamples them to 1/16 resolution,
#            multiplies them with the current features, builds a multi-scale pyramid, and performs
#            cross-attention with memory features to generate structure maps. These structure maps
#            are used to spatially enhance the final features of the main branch. The structure maps
#            from each layer are also returned after 16x upsampling.

from typing import Optional, Tuple, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
try:
    import kornia
    import kornia.filters as KF
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False
from robotseg.modeling.sam.transformer import RoPEAttention
from robotseg.modeling.robotseg_utils import get_activation_fn, get_clones


# =============== Helper Modules ===============

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise followed by pointwise convolution to reduce computation."""
    def __init__(self, channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                            padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(num_groups=channels // 8 if channels >= 8 else 1, num_channels=channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MultiScalePyramid(nn.Module):
    """Multi-scale feature pyramid: applies and fuses 3x3, 5x5, and 7x7 depthwise separable convolutions."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = DepthwiseSeparableConv(channels, 3, 1)
        self.conv5 = DepthwiseSeparableConv(channels, 5, 2)
        self.conv7 = DepthwiseSeparableConv(channels, 7, 3)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x3, x5, x7], dim=1)
        out = self.fuse(out)
        out = self.act(out)
        return out


def _infer_hw_from_tokens(n_tokens: int) -> Tuple[int, int]:
    """Infer H and W from the number of tokens, assuming a square grid by default."""
    H = int(math.sqrt(n_tokens))
    W = H
    if H * W != n_tokens:
        # If it is not a perfect square, fall back to a shape that is as close to square as possible.
        W = n_tokens // H
        if H * W != n_tokens:
            # Final fallback: directly use (1, N).
            H, W = 1, n_tokens
    return H, W


def _safe_to_gray(raw_image: Tensor) -> Tensor:
    """
    Convert RGB images to grayscale.

    Input:  [B, 3, H, W] or [B, 1, H, W]
    Output: [B, 1, H, W]

    The raw image is assumed to be in the range [0, 1] or [0, 255];
    normalization is handled internally.
    """
    if raw_image.dim() != 4:
        raise ValueError("raw_image should be [B,C,H,W]")
    B, C, H, W = raw_image.shape
    x = raw_image
    if x.dtype.is_floating_point:
        if x.max() > 1.5:
            x = x / 255.0
    else:
        x = x.float() / 255.0

    if C == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif C == 1:
        gray = x
    else:
        if C > 3:
            x = x[:, :3]
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = x.mean(dim=1, keepdim=True)
    return gray.clamp(0, 1)


def _canny_or_sobel_edges(raw_image: Tensor, low: float = 0.1, high: float = 0.2) -> Tensor:
    """
    Return an edge map with shape [B, 1, H, W].

    Canny edges from Kornia are used when available; otherwise, Sobel magnitude
    with thresholding is used as an approximation.

    The computation is wrapped in no_grad(), so gradients are not propagated.
    """
    gray = _safe_to_gray(raw_image)
    with torch.no_grad():
        if _HAS_KORNIA:
            edges, _ = KF.canny(gray, low_threshold=low, high_threshold=high)
            if edges.dtype != torch.float32:
                edges = edges.float()
            edges = edges.clamp(0, 1)
        else:
            kernel = 5
            gray_blur = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
            gx = F.conv2d(gray_blur, sobel_x, padding=1)
            gy = F.conv2d(gray_blur, sobel_y, padding=1)
            mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
            mmin, mmax = mag.amin(dim=(2, 3), keepdim=True), mag.amax(dim=(2, 3), keepdim=True)
            mag = (mag - mmin) / (mmax - mmin + 1e-6)
            edges = (mag > 0.2).float()
    return edges


# =============== Main Module ===============

class MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
        # ---- Cross-attention and tunable hyperparameters for the structure branch ----
        cross_attention_structure: nn.Module,
        structure_strength: float = 1.0,  # Strength of structure enhancement for multiplicative gating.
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention
        self.cross_attn_structure = cross_attention_structure

        # feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # activation
        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # positional flags
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        # ===== Structure Branch Components =====
        # Multi-scale pyramid applied after converting tokens to feature maps
        self.ms_pyramid = MultiScalePyramid(d_model)
        # Structure-map prediction head: a 1x1 convolution that generates a single-channel structure map
        self.structure_head = nn.Conv2d(d_model, 1, kernel_size=3, padding=1, bias=True)
        # Structure gating strength
        self.structure_strength = nn.Parameter(torch.tensor(float(structure_strength), dtype=torch.float32))

    # -------- Main Self-Attention Branch --------
    def _forward_sa(self, tgt: Tensor, query_pos: Optional[Tensor]) -> Tensor:
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    # -------- Main Cross-Attention Branch --------
    def _forward_ca(self, tgt: Tensor, memory: Tensor, query_pos: Optional[Tensor],
                    pos: Optional[Tensor], num_k_exclude_rope=0) -> Tensor:
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    # -------- Structure Branch --------
    def _forward_structure_branch(
        self,
        raw_image: Tensor,        # [B,3,h_img,w_img] or [B,1,h_img,w_img]
        curr_tokens: Tensor,      # [B,N,C]
        memory_tokens: Tensor,    # [B,N_mem,C]
        query_pos: Optional[Tensor],
        memory_pos: Optional[Tensor],
        num_k_exclude_rope: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            structure_feat_tokens: [B, N, C] structure-enhanced features used for map prediction.
            structure_map_up: [B, 1, 16*H, 16*W] high-resolution structure map for supervision.
        """
        B, N, C = curr_tokens.shape
        H, W = _infer_hw_from_tokens(N)

        # --- 1) raw_image -> Canny edges ---
        edges = _canny_or_sobel_edges(raw_image)  # [B,1,h_img,w_img]

        # --- 2) Resize to the token grid size H x W ---
        edges_small = F.interpolate(edges, size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]
        edges_small = edges_small.clamp(0, 1)

        # --- 3) Convert curr_tokens to a feature map and enhance it pixel-wise with edges_small ---
        feat = curr_tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)  # [B,C,H,W]
        feat_edge_enh = feat * (1.0 + edges_small)

        # --- 4) Extract structure-aware features using the multi-scale pyramid ---
        feat_ms = self.ms_pyramid(feat_edge_enh)  # [B,C,H,W]

        # --- 5) Perform cross-attention with memory features as keys/values and structure features as queries ---
        q_tokens = feat_ms.flatten(2).permute(0, 2, 1).contiguous()  # [B,N,C]

        kwds = {}
        if num_k_exclude_rope > 0 and isinstance(self.cross_attn_structure, RoPEAttention):
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        q_for_attn = q_tokens + query_pos if (query_pos is not None and self.pos_enc_at_cross_attn_queries) else q_tokens
        k_for_attn = memory_tokens + memory_pos if (memory_pos is not None and self.pos_enc_at_cross_attn_keys) else memory_tokens

        struct_tokens = self.cross_attn_structure(q=q_for_attn, k=k_for_attn, v=memory_tokens, **kwds)  # [B,N,C]

        # --- 6) Convert the structure response to a feature map, predict a single-channel structure map, and upsample it by 16x ---
        struct_fmap = struct_tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)  # [B,C,H,W]
        structure_map = self.structure_head(struct_fmap)  # [B,1,H,W]
        structure_map_up = F.interpolate(
            structure_map, scale_factor=16, mode="bilinear", align_corners=False
        )  # [B,1,16H,16W]

        return struct_tokens, structure_map_up

    def forward(
        self,
        tgt: Tensor,  # [S, B, C] or [B, S, C], depending on batch_first
        memory: Tensor,  # [S_mem,B,C] or [B,S_mem,C]
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
        # --- Additional inputs for the structure branch ---
        raw_image: Optional[Tensor] = None,  # [B,3,h_img,w_img]
        memory_pos: Optional[Tensor] = None,
        batch_first: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            tgt_enhanced: main-branch output enhanced by the structure map, with the same shape as the input.
            structure_map_up: [B, 1, 16H, 16W] structure map from each layer, used for supervision.
        """
        # ---- batch_first [B,S,C] ----
        if not batch_first:
            tgt = tgt.transpose(0, 1).contiguous()
            memory = memory.transpose(0, 1).contiguous()
            if pos is not None:
                pos = pos.transpose(0, 1).contiguous()
            if query_pos is not None:
                query_pos = query_pos.transpose(0, 1).contiguous()
            if memory_pos is not None:
                memory_pos = memory_pos.transpose(0, 1).contiguous()

        # ===== Main branch: Self-Attn -> Cross-Attn -> MLP =====
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt_main = tgt + self.dropout3(tgt2)  # Final tokens from the main branch before structure enhancement

        # ===== Structure branch: enabled only when raw_image is provided =====
        if raw_image is not None:
            struct_tokens, structure_map_up = self._forward_structure_branch(
                raw_image=raw_image,
                curr_tokens=tgt,
                memory_tokens=memory,
                query_pos=query_pos,
                memory_pos=pos if memory_pos is None else memory_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
            # Use the sigmoid-activated structure map (H x W) as a gate and apply it pixel-wise to the main-branch output
            B, N, C = tgt_main.shape
            H, W = _infer_hw_from_tokens(N)
            struct_map = torch.sigmoid(
                structure_map_up  # [B,1,16H,16W]
            )
            # Downsample back to H x W for pixel-level gating
            struct_gate = F.interpolate(struct_map, size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]
            # token -> fmap
            fmap_main = tgt_main.permute(0, 2, 1).contiguous().view(B, C, H, W)
            # Multiplicative enhancement: (1 + alpha * gate)
            fmap_main = fmap_main * (1.0 + self.structure_strength * struct_gate)
            # fmap -> tokens
            tgt_enhanced = fmap_main.flatten(2).permute(0, 2, 1).contiguous()
        else:
            # If raw_image is not provided, skip structure enhancement and return a placeholder map filled with zeros
            B, N, C = tgt_main.shape
            H, W = _infer_hw_from_tokens(N)
            zero_map = torch.zeros((B, 1, 16 * H, 16 * W), device=tgt_main.device, dtype=tgt_main.dtype)
            structure_map_up = zero_map
            tgt_enhanced = tgt_main

        # ---- Restore the layout to match the input ----
        if not batch_first:
            tgt_enhanced = tgt_enhanced.transpose(0, 1).contiguous()

        return tgt_enhanced, structure_map_up


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        raw_image: torch.Tensor,            # raw image rgb inputs, [B,C,H,W], B is object number
        curr: torch.Tensor,                 # self-attention inputs  [S,B,C] or [B,S,C]
        memory: torch.Tensor,               # cross-attention inputs [S_mem,B,C] or [B,S_mem,C]
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,# pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,        # number of object pointer *tokens*
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            normed_output:   [S, B, C] or [B, S, C], matching the input layout.
            structure_maps:  [B, L, 1, 16H, 16W], one 16x upsampled structure supervision map per layer.
        """
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Convert to batch-first layout before each layer
        need_transpose = False
        if self.batch_first:
            # Convert to batch first [B,S,C]
            output     = output.transpose(0, 1).contiguous()
            curr_pos   = None if curr_pos is None else curr_pos.transpose(0, 1).contiguous()
            memory     = memory.transpose(0, 1).contiguous()
            memory_pos = None if memory_pos is None else memory_pos.transpose(0, 1).contiguous()
            need_transpose = True

        structure_maps: List[Tensor] = []
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            # Run a single layer and obtain the structure map upsampled by 16x
            output, structure_map_up = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                raw_image=raw_image,
                memory_pos=memory_pos,
                batch_first=True,  # we use batch_first
                **kwds,
            )
            structure_maps.append(structure_map_up)

        normed_output = self.norm(output)

        # Restore the layout to match the input
        if need_transpose:
            normed_output = normed_output.transpose(0, 1).contiguous()  # [S,B,C]
            # curr_pos is only used for input enhancement, so no restoration is needed here

        # Concatenate structure maps from all layers into [B, L, 16H, 16W]
        structure_maps_tensor = torch.cat(structure_maps, dim=1)

        return normed_output, structure_maps_tensor
