from collections import defaultdict
from typing import Dict, List
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
from train.trainer import CORE_LOSS_KEY
from train.utils.distributed import get_world_size, is_dist_avail_and_initialized


#######################################################################################
############################### Basic Loss Functions ##################################
#######################################################################################

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    DICE loss
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W]
        assert inputs.dim() == 4 and targets.dim() == 4
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    RetinaNet focal loss
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # [N, M, H, W]
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    # inputs:[N,M,H,W] or [N,1,H,W] => reduce on channel dim
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    IoU supervision for predicted IoU head
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


#######################################################################################
########## Label-Efficient Training: Cycle + Semantic + Patch + Structure Loss ########
#######################################################################################

def masked_avg_feat(emb, mask):
    """
    Compute the mask-weighted average embedding
    emb: [N,1,C,H,W], mask: [N,1,H_full,W_full]
    """
    N, _, C, H, W = emb.shape
    if mask.shape[-2:] != (H, W):
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
    num = (emb * mask[:, :, None, :, :]).flatten(3).sum(-1)
    den = mask.flatten(2).sum(-1).clamp_min(1e-6)
    return num / den.unsqueeze(-1)


def select_best_mask_index(loss_multimask, loss_multidice, w_mask, w_dice):
    """Select the optimal mask index based on the combined loss."""
    combo = loss_multimask * w_mask + loss_multidice * w_dice
    best_inds = torch.argmin(combo, dim=-1)
    return best_inds


def select_mask_index_no_gt_by_iou(pred_ious: torch.Tensor):
    """
    Select the channel using predicted IoU when GT is unavailable.
    pred_ious: [N, M]
    Returns a tensor of shape [N], where each element is the index of the mask
    with the highest predicted IoU for that sample.
    """
    best_inds = torch.argmax(pred_ious, dim=-1)
    return best_inds


def patch_consistency_loss(
    pred_mask_prob: torch.Tensor,  # [N,1,H,W], after sigmoid
    target_mask_bin: torch.Tensor,  # [N,1,H,W], 0/1
    down_ratio: int = 16,
):
    """
    Patch Consistency Loss
    ------------------------------------------------------------
    Designed for scenarios where pseudo-labels exhibit noticeable aliasing
    (e.g., results propagated from DINOv3 patches).

    The loss is computed at a downsampled resolution (1/down_ratio, default 1/16)
    using soft IoU, in order to emphasize global shape consistency rather than
    being affected by jagged boundaries in pseudo-labels.

    loss = 1 - IoU(pred ↓, target ↓)
    ------------------------------------------------------------
    Args:
        pred_mask_prob: Predicted probability map after sigmoid, shape [N, 1, H, W]
        target_mask_bin: Binary pseudo-label, shape [N, 1, H, W]
        down_ratio: Downsampling factor (default: 16, corresponding to patch scale)
    Returns:
        Scalar IoU loss (torch.Tensor)
    """
    p = pred_mask_prob.clamp(0, 1)
    t = target_mask_bin.clamp(0, 1)

    Hs, Ws = max(1, p.shape[-2] // down_ratio), max(1, p.shape[-1] // down_ratio)
    p_small = F.interpolate(p, size=(Hs, Ws), mode="bilinear", align_corners=False)
    t_small = F.interpolate(t, size=(Hs, Ws), mode="bilinear", align_corners=False)

    inter = (p_small * t_small).sum(dim=[1, 2, 3])
    union = (p_small + t_small - p_small * t_small).sum(dim=[1, 2, 3])
    iou_loss = 1 - (inter + 1e-6) / (union + 1e-6)

    return iou_loss.mean()


class MultiStepMultiMasksAndIous_LET(nn.Module):
    """
    Label-Efficient Training
    ------------------------------------------------------------
    1) Cycle Consistency Loss:
       Applied to the first and last frames with ground-truth masks.

    2) Semantic Consistency Loss:
       Applied to intermediate frames only, enforcing consistency between
       predicted mask features and the first-frame anchor semantics.

    3) Patch Consistency Loss:
       Applied to intermediate frames only, encouraging patch-level shape consistency.

    4) Structure Supervision:
       Applied to the last frame, supervising the structure map generated by the
       Structure-Enhanced Memory Associator with the last-frame ground-truth mask.
    ------------------------------------------------------------
    """
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        semantic_loss_type="cosine",
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in weight_dict
        assert "loss_dice" in weight_dict
        assert "loss_iou" in weight_dict
        if "loss_class" not in weight_dict:
            self.weight_dict["loss_class"] = 0.0
        assert "loss_semantic" in weight_dict
        assert "loss_patch" in weight_dict
        assert "loss_structure" in weight_dict

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.semantic_loss_type = semantic_loss_type

        self.anchor_feat = None

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        """
        outs_batch: List of length T, where each element is a prediction dictionary
                    corresponding to one frame.
        targets_batch: Tensor of shape [T, N, H, W], containing the ground-truth or
                       pseudo ground-truth masks for each frame.
        """
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(targets_batch.shape[1], device=targets_batch.device, dtype=torch.float)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)

        # === Step 1: Build the anchor semantic features from the first frame ===
        first_outs = outs_batch[0]
        first_targets = targets_batch[0]
        target_masks_first = first_targets.unsqueeze(1).float()  # [N,1,H,W]

        first_src_masks = first_outs["multistep_pred_multimasks_high_res"][-1]  # [N,M,H,W]
        first_embeddings = first_outs["multistep_embeddings"][-1]               # [N,C,H,W]
        first_ious = first_outs["multistep_pred_ious"][-1]                      # [N,M]

        loss_multimask = sigmoid_focal_loss(first_src_masks, target_masks_first.expand_as(first_src_masks),
                                            num_objects, alpha=self.focal_alpha, gamma=self.focal_gamma, loss_on_multimask=True)
        loss_multidice = dice_loss(first_src_masks, target_masks_first.expand_as(first_src_masks),
                                   num_objects, loss_on_multimask=True)
        best_inds_first = select_best_mask_index(loss_multimask, loss_multidice,
                                                 self.weight_dict["loss_mask"], self.weight_dict["loss_dice"])

        batch_inds = torch.arange(first_src_masks.size(0), device=first_src_masks.device)
        # using model prediction
        # first_best_mask = first_src_masks.sigmoid()[batch_inds, best_inds_first][:, None, ...]  # [N,1,H,W]
        # using gt
        first_best_mask = target_masks_first  # [N,1,H,W]
        first_embeddings_sel = first_embeddings[:, None, ...]      # [N,1,C,H,W]
        self.anchor_feat = F.normalize(masked_avg_feat(first_embeddings_sel, first_best_mask), dim=-1)  # [N,1,C]

        # === Step 2: Iterate over all frames ===
        T = len(outs_batch)
        for idx, (outs, targets) in enumerate(zip(outs_batch, targets_batch)):
            compute_cyc = (idx == 0 or idx == T - 1)
            compute_sem = (idx > 0 and idx < T - 1)
            compute_pat = (idx > 0 and idx < T - 1)
            if T > 1:
                compute_str = (idx == T - 1)
            elif T == 1:
                compute_str = False

            cur_losses = self._forward_frame(outs, targets, num_objects, compute_cyc, compute_sem, compute_pat, compute_str)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward_frame(self, outputs: Dict, targets: torch.Tensor, num_objects, compute_cyc, compute_sem, compute_pat, compute_str):
        target_masks = targets.unsqueeze(1).float()  # [N,1,H,W]
        assert target_masks.dim() == 4  # [N, 1, H, W]

        device = target_masks.device
        dtype = torch.float32
        losses = {
            "loss_mask": torch.zeros((), device=device, dtype=dtype),
            "loss_dice": torch.zeros((), device=device, dtype=dtype),
            "loss_iou": torch.zeros((), device=device, dtype=dtype),
            "loss_class": torch.zeros((), device=device, dtype=dtype),
            "loss_semantic": torch.zeros((), device=device, dtype=dtype),
            "loss_patch": torch.zeros((), device=device, dtype=dtype),
            "loss_structure": torch.zeros((), device=device, dtype=dtype),
        }

        src_masks_list = outputs["multistep_pred_multimasks_high_res"]   # list([N,M,H,W]), N is multiple refinement steps, M is multiple mask channels

        src_embeddings_list = outputs["multistep_embeddings"]            # list([N,C,H,W])
        if len(src_embeddings_list) == 1 and len(src_masks_list) > 1:
            src_embeddings_list = src_embeddings_list * len(src_masks_list)

        ious_list = outputs["multistep_pred_ious"]                       # list([N,M])
        object_score_logits_list = outputs["multistep_object_score_logits"]  # list([N,M])

        # -------- (a) Multi-step refinement on a single frame -------- #
        best_inds_for_sem_last = None  # Record the channel selected at the final step
        pred_mask_prob_last = None    # Record the mask probability map selected at the final refinement step

        for src_masks, src_embeddings, ious, object_score_logits in zip(
            src_masks_list, src_embeddings_list, ious_list, object_score_logits_list
        ):
            b_inds, best_inds_for_sem, pred_mask_prob = self._update_losses_step(
                losses, src_masks, target_masks, ious, num_objects,
                object_score_logits, src_embeddings, compute_cyc, compute_sem
            )
            best_inds_for_sem_last = best_inds_for_sem
            pred_mask_prob_last = pred_mask_prob  # [N,1,H,W]

        if compute_str and ("multistep_pred_multimasks_structure" in outputs):
            structure_logits = outputs["multistep_pred_multimasks_structure"][-1]  # [B, C, H, W]
            H_full, W_full = structure_logits.shape[-2:]

            # Sigmoid to get predicted edges in 0–1 range
            structure_pred = torch.sigmoid(structure_logits)

            # Compute a structure map from target at the original resolution
            structure_true = torch.abs(KF.laplacian(target_masks, kernel_size=3))  # [B, 1, H, W]
            structure_true = structure_true / (structure_true.amax(dim=[2, 3], keepdim=True) + 1e-6)

            # Match channels if needed
            if structure_true.shape[1] == 1 and structure_pred.shape[1] > 1:
                structure_true = structure_true.expand(-1, structure_pred.shape[1], -1, -1)

            # L1 loss at full resolution
            structure_loss = F.l1_loss(structure_pred, structure_true, reduction="mean")
            losses["loss_structure"] += structure_loss / num_objects

        # -------- (b) after multi-step refinement, compute patch consistency loss on the last refinement step of this frame -------- #
        if compute_pat and (pred_mask_prob_last is not None):
            patch_loss = patch_consistency_loss(pred_mask_prob_last, target_masks)
            losses["loss_patch"] += patch_loss / num_objects

        # -------- (c) weighted-sum -------- #
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)

        return losses

    def _update_losses_step(self, losses, src_masks, target_masks, ious, num_objects,
                            object_score_logits, src_embeddings, compute_cyc, compute_sem):

        N, M, H, W = src_masks.shape
        batch_inds = torch.arange(N, device=src_masks.device)
        best_inds_for_cyc = None
        best_inds_for_sem = None

        # ---------- 1) Cycle Consistency (first/last frame) ----------
        if compute_cyc:
            target_masks_exp = target_masks.expand_as(src_masks)
            loss_multimask = sigmoid_focal_loss(src_masks, target_masks_exp, num_objects,
                                                alpha=self.focal_alpha, gamma=self.focal_gamma, loss_on_multimask=True)
            loss_multidice = dice_loss(src_masks, target_masks_exp, num_objects, loss_on_multimask=True)

            if not self.pred_obj_scores:
                loss_class = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
                target_obj = torch.ones(loss_multimask.shape[0], 1, dtype=loss_multimask.dtype, device=loss_multimask.device)
            else:
                target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[..., None].float()
                loss_class = sigmoid_focal_loss(object_score_logits, target_obj, num_objects,
                                                alpha=self.focal_alpha_obj_score, gamma=self.focal_gamma_obj_score)

            loss_multiiou = iou_loss(src_masks, target_masks_exp, ious, num_objects,
                                     loss_on_multimask=True, use_l1_loss=self.iou_use_l1_loss)

            best_inds_for_cyc = select_best_mask_index(
                loss_multimask, loss_multidice,
                self.weight_dict["loss_mask"], self.weight_dict["loss_dice"]
            )

            loss_mask = loss_multimask[batch_inds, best_inds_for_cyc].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_inds_for_cyc].unsqueeze(1)
            loss_iou  = loss_multiiou [batch_inds, best_inds_for_cyc].unsqueeze(1)

            # backprop focal, dice and iou loss only if obj present
            loss_mask = loss_mask * target_obj
            loss_dice = loss_dice * target_obj
            loss_iou  = loss_iou  * target_obj

            # sum over batch dimension (note that the losses are already divided by num_objects)
            losses["loss_mask"] += loss_mask.sum()
            losses["loss_dice"] += loss_dice.sum()
            losses["loss_iou"]  += loss_iou.sum()
            losses["loss_class"] += loss_class

        # ---------- 2) Semantic Consistency (intermediate frames) ----------
        pred_mask_prob = None
        if compute_sem:
            assert self.anchor_feat is not None
            if best_inds_for_cyc is None:
                best_inds_for_sem = select_mask_index_no_gt_by_iou(ious)
            else:
                best_inds_for_sem = best_inds_for_cyc

            pred_mask_prob = src_masks.sigmoid()[batch_inds, best_inds_for_sem][:, None, ...]  # [N,1,H,W]
            cur_embeddings_sel = src_embeddings[:, None, ...]  # [N,1,C,H,W]
            cur_feat = F.normalize(masked_avg_feat(cur_embeddings_sel, pred_mask_prob), dim=-1) # [N,1,C]

            if self.semantic_loss_type == "l2":
                semantic_loss = F.mse_loss(cur_feat, self.anchor_feat)
            else:
                semantic_loss = 1.0 - (cur_feat * self.anchor_feat).sum(dim=-1).mean()

            losses["loss_semantic"] += semantic_loss / num_objects
        else:
            if best_inds_for_cyc is None:
                best_inds_for_sem = select_mask_index_no_gt_by_iou(ious)
            else:
                best_inds_for_sem = best_inds_for_cyc
            pred_mask_prob = src_masks.sigmoid()[batch_inds, best_inds_for_sem][:, None, ...]

        return batch_inds, best_inds_for_sem, pred_mask_prob

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for key, w in self.weight_dict.items():
            if key not in losses:
                raise ValueError(f"{type(self)} missing {key}")
            if w != 0:
                reduced_loss += losses[key] * w
        return reduced_loss
