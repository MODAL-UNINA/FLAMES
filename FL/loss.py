import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth + 1e-16)
    return 1.0 - dice


def focal_loss(pred, target, alpha=0.8, gamma=2):
    """
    alpha (float): peso assegnato alla classe positiva
    gamma (float): esponente per la focal loss, più gamma è alto,
                    più la focal loss si concentra sui campioni difficili
    """
    pred = pred.view(-1)
    target = target.view(-1)

    BCE_loss = F.binary_cross_entropy(pred, target, reduction="mean")
    BCE_exp = torch.exp(-BCE_loss)
    focal_loss = alpha * (1 - BCE_exp) ** gamma * BCE_loss
    return focal_loss


def dice_focal_loss(
    pred, target, alpha=0.8, gamma=2, smooth=1.0, a=0.7
) -> torch.Tensor:

    assert (
        pred.size() == target.size()
    ), f"Prediction size: {pred.size()}, Target size: {target.size()}"

    dice = dice_loss(pred, target, smooth)
    focal = focal_loss(pred, target, alpha, gamma)

    return a * dice + (1 - a) * focal


# %%
# Knowledge Distillation

# Distillation Loss (Kullback-Leibler Divergence)
def distillation_loss(student_logits, teacher_logits, T=2.0):

    student_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)

    distill_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (
        T**2
    )
    return distill_loss


# Dice-Focal Loss and Distillation Loss
def combined_loss(
    pred,
    target,
    student_logits,
    teacher_logits,
    alpha=0.8,
    gamma=2,
    smooth=1.0,
    distill_weight=0.3,
    T=3.0,
):

    assert (
        student_logits.size() == teacher_logits.size()
    ), f"Logits dimensions don't match! {student_logits.size()} != {teacher_logits.size()}"

    segmentation_loss = dice_focal_loss(pred, target, alpha, gamma, smooth)

    if distill_weight == 0:
        return segmentation_loss
    distill_loss = distillation_loss(student_logits, teacher_logits, T)

    total_loss = (
        1 - distill_weight
    ) * segmentation_loss + distill_weight * distill_loss
    return total_loss
