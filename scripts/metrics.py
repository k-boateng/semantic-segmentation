import torch


def logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert segmentation logits to predicted class indices.

    Args:
        logits: Tensor of shape (B, C, H, W)

    Returns:
        preds: Tensor of shape (B, H, W)
    """
    return torch.argmax(logits, dim=1)


def _flatten_valid(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove ignored pixels and flatten predictions/targets.
    """
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    return preds, targets


def compute_pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """
    Compute pixel accuracy over all non-ignored pixels.
    """
    preds, targets = _flatten_valid(preds, targets, ignore_index)

    if targets.numel() == 0:
        return float("nan")

    correct = (preds == targets).sum().float()
    total = torch.tensor(targets.numel(), device=targets.device, dtype=torch.float32)
    return (correct / total).item()


def compute_per_class_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Compute IoU for each class separately.

    Returns:
        Tensor of shape (num_classes,) with NaN for undefined classes.
    """
    preds, targets = _flatten_valid(preds, targets, ignore_index)

    per_class_iou = torch.full(
        (num_classes,),
        float("nan"),
        dtype=torch.float32,
        device=preds.device if preds.numel() > 0 else targets.device,
    )

    if targets.numel() == 0:
        return per_class_iou

    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            per_class_iou[c] = intersection / union

    return per_class_iou


def compute_mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """
    Compute mean IoU across valid classes.
    """
    per_class_iou = compute_per_class_iou(preds, targets, num_classes, ignore_index)
    valid = ~torch.isnan(per_class_iou)

    if valid.sum() == 0:
        return float("nan")

    return per_class_iou[valid].mean().item()


def compute_per_class_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Compute Dice score for each class separately.

    Returns:
        Tensor of shape (num_classes,) with NaN for undefined classes.
    """
    preds, targets = _flatten_valid(preds, targets, ignore_index)

    per_class_dice = torch.full(
        (num_classes,),
        float("nan"),
        dtype=torch.float32,
        device=preds.device if preds.numel() > 0 else targets.device,
    )

    if targets.numel() == 0:
        return per_class_dice

    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c

        intersection = (pred_c & target_c).sum().float()
        denominator = pred_c.sum().float() + target_c.sum().float()

        if denominator > 0:
            per_class_dice[c] = (2.0 * intersection) / denominator

    return per_class_dice


def compute_mean_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """
    Compute mean Dice across valid classes.
    """
    per_class_dice = compute_per_class_dice(preds, targets, num_classes, ignore_index)
    valid = ~torch.isnan(per_class_dice)

    if valid.sum() == 0:
        return float("nan")

    return per_class_dice[valid].mean().item()


def compute_per_class_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Compute per-class accuracy (class-wise recall).

    For class c:
        correctly predicted pixels of class c / total true pixels of class c

    Returns:
        Tensor of shape (num_classes,) with NaN for undefined classes.
    """
    preds, targets = _flatten_valid(preds, targets, ignore_index)

    per_class_acc = torch.full(
        (num_classes,),
        float("nan"),
        dtype=torch.float32,
        device=preds.device if preds.numel() > 0 else targets.device,
    )

    if targets.numel() == 0:
        return per_class_acc

    for c in range(num_classes):
        target_c = targets == c
        denom = target_c.sum().float()

        if denom > 0:
            correct_c = ((preds == c) & target_c).sum().float()
            per_class_acc[c] = correct_c / denom

    return per_class_acc


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Rows = true classes
    Cols = predicted classes

    Returns:
        Tensor of shape (num_classes, num_classes)
    """
    preds, targets = _flatten_valid(preds, targets, ignore_index)

    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=targets.device)

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)

    return cm


def compute_hd95(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Compute per-class HD95 averaged over the batch.

    Args:
        preds: Tensor of shape (B, H, W)
        targets: Tensor of shape (B, H, W)

    Returns:
        Tensor of shape (num_classes,) with NaN for undefined classes.
    """
    from medpy.metric.binary import hd95

    if preds.ndim != 3 or targets.ndim != 3:
        raise ValueError("preds and targets must have shape (B, H, W) for HD95 computation.")

    batch_size = preds.shape[0]
    per_class_values = [[] for _ in range(num_classes)]

    for b in range(batch_size):
        pred_b = preds[b]
        target_b = targets[b]

        valid = target_b != ignore_index

        for c in range(num_classes):
            pred_c = ((pred_b == c) & valid).cpu().numpy()
            target_c = ((target_b == c) & valid).cpu().numpy()

            if pred_c.sum() == 0 or target_c.sum() == 0:
                continue

            value = hd95(pred_c, target_c)
            per_class_values[c].append(float(value))

    result = torch.full((num_classes,), float("nan"), dtype=torch.float32)

    for c in range(num_classes):
        if len(per_class_values[c]) > 0:
            result[c] = torch.tensor(per_class_values[c], dtype=torch.float32).mean()

    return result