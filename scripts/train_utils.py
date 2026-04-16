from metrics import (
    logits_to_preds,
    compute_pixel_accuracy,
    compute_mean_iou,
    compute_mean_dice,
)

import math
import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    num_classes,
    ignore_index=255,
):
    model.train()

    running_loss = 0.0
    running_pixel_acc = 0.0
    running_mean_iou = 0.0
    running_mean_dice = 0.0
    total_samples = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        preds = logits_to_preds(logits)

        pixel_acc = compute_pixel_accuracy(preds, masks, ignore_index)
        mean_iou = compute_mean_iou(preds, masks, num_classes, ignore_index)
        mean_dice = compute_mean_dice(preds, masks, num_classes, ignore_index)

        if math.isnan(pixel_acc):
            pixel_acc = 0.0
        if math.isnan(mean_iou):
            mean_iou = 0.0
        if math.isnan(mean_dice):
            mean_dice = 0.0

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        running_pixel_acc += pixel_acc * batch_size
        running_mean_iou += mean_iou * batch_size
        running_mean_dice += mean_dice * batch_size
        total_samples += batch_size

    output = {
        "loss": running_loss / total_samples,
        "pixel_acc": running_pixel_acc / total_samples,
        "mean_iou": running_mean_iou / total_samples,
        "mean_dice": running_mean_dice / total_samples,
    }

    return output

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    num_classes,
    ignore_index=255,
):
    model.eval()

    running_loss = 0.0
    running_pixel_acc = 0.0
    running_mean_iou = 0.0
    running_mean_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            preds = logits_to_preds(logits)

            pixel_acc = compute_pixel_accuracy(preds, masks, ignore_index)
            mean_iou = compute_mean_iou(preds, masks, num_classes, ignore_index)
            mean_dice = compute_mean_dice(preds, masks, num_classes, ignore_index)

            if math.isnan(pixel_acc):
                pixel_acc = 0.0
            if math.isnan(mean_iou):
                mean_iou = 0.0
            if math.isnan(mean_dice):
                mean_dice = 0.0

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            running_pixel_acc += pixel_acc * batch_size
            running_mean_iou += mean_iou * batch_size
            running_mean_dice += mean_dice * batch_size
            total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "pixel_acc": running_pixel_acc / total_samples,
        "mean_iou": running_mean_iou / total_samples,
        "mean_dice": running_mean_dice / total_samples,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    save_path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, save_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)