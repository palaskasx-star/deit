# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from utils import create_divergence_names, plot_divergence

import torch.nn.functional as F

import numpy as np
import os


@torch.no_grad()
def create_divergence_plots(data_loader, model, data_set, names, output_dir, device):
    criterion = torch.nn.CrossEntropyLoss()

    step_divergence_names, label_divergence_names, base_divergence_names = create_divergence_names(names)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = data_set + ':'

    # switch to evaluation mode
    model.eval()

    step_divergence_epoch = torch.zeros((len(step_divergence_names), 14*14))
    label_divergence_epoch = torch.zeros((len(label_divergence_names), 14*14))
    base_divergence_epoch = torch.zeros((len(base_divergence_names), 14*14))

    step_divergence_epoch_cls = torch.zeros((len(step_divergence_names), 1))
    label_divergence_epoch_cls = torch.zeros((len(label_divergence_names), 1))
    base_divergence_epoch_cls = torch.zeros((len(base_divergence_names), 1))

    for images, target in metric_logger.log_every(data_loader, 10, data_set):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        

        # compute output
        with torch.cuda.amp.autocast():
            representations, dist_tokens, output = model(images, require_feat=True)

            one_hot_labels_features = F.one_hot(target, num_classes=output.shape[1]).float()
            one_hot_labels_features = one_hot_labels_features

            # representations
            kernels = []
            for i in range(len(representations)):
                kernels.append(compute_kernels(representations[i], 1))
            kernels_ = kernels.copy()
            kernels_.append(compute_kernels(one_hot_labels_features.unsqueeze(1).float(), 1))
            step_divergence, label_divergence, base_divergence = compute_jeffreys_divergences(kernels_)

            step_divergence_epoch += step_divergence.detach().to('cpu')
            label_divergence_epoch += label_divergence.detach().to('cpu')
            base_divergence_epoch += base_divergence.detach().to('cpu')

            # dist tokens
            kernels_cls = []
            for i in range(len(dist_tokens)):
                kernels_cls.append(compute_kernels(dist_tokens[i], 1))
            kernels_cls_ = kernels_cls.copy()
            kernels_cls_.append(compute_kernels(one_hot_labels_features.unsqueeze(1).float(), 1))
            step_divergence_cls, label_divergence_cls, base_divergence_cls = compute_jeffreys_divergences(kernels_cls_)

            step_divergence_epoch_cls += step_divergence_cls.detach().to('cpu')
            label_divergence_epoch_cls += label_divergence_cls.detach().to('cpu')
            base_divergence_epoch_cls += base_divergence_cls.detach().to('cpu')

            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))


    step_divergence_epoch, label_divergence_epoch, base_divergence_epoch = step_divergence_epoch / len(data_loader), label_divergence_epoch / len(data_loader), base_divergence_epoch / len(data_loader)
    step_divergence_epoch_cls, label_divergence_epoch_cls, base_divergence_epoch_cls = step_divergence_epoch_cls / len(data_loader), label_divergence_epoch_cls / len(data_loader), base_divergence_epoch_cls / len(data_loader)

    plot_divergence(step_divergence_epoch.unsqueeze(0).mean(dim=-1), output_dir, data_set , step_divergence_names, "Consecutive layers", 1, False)
    plot_divergence(label_divergence_epoch.unsqueeze(0).mean(dim=-1), output_dir, data_set , label_divergence_names, "Label divergence", 1, False)
    plot_divergence(base_divergence_epoch.unsqueeze(0).mean(dim=-1), output_dir, data_set , base_divergence_names, "Base divergence", 1, False)

    plot_divergence(step_divergence_epoch_cls.unsqueeze(0).mean(dim=-1), output_dir, data_set , step_divergence_names, "Consecutive layers dist token", 1, False)
    plot_divergence(label_divergence_epoch_cls.unsqueeze(0).mean(dim=-1), output_dir, data_set , label_divergence_names, "Label divergence dist token", 1, False)
    plot_divergence(base_divergence_epoch_cls.unsqueeze(0).mean(dim=-1), output_dir, data_set , base_divergence_names, "Base divergence dist token", 1, False)

    save_path = os.path.join(output_dir, data_set)
    os.makedirs(save_path, exist_ok=True)
    
    divergence_data = {
        "base_divergence_epoch_cls": base_divergence_epoch_cls,
        "label_divergence_epoch_cls": label_divergence_epoch_cls,
        "step_divergence_epoch_cls": step_divergence_epoch_cls,
        "base_divergence_epoch": base_divergence_epoch,
        "label_divergence_epoch": label_divergence_epoch,
        "step_divergence_epoch": step_divergence_epoch
    }
    
    torch.save(divergence_data, os.path.join(save_path, "divergence_matrices.pt"))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_jeffreys_divergences(kernels, eps=1e-7):
    kernels_norm = []
    for i in range(len(kernels)):
        kernels_norm.append(kernels[i] / (kernels[i].sum(dim=1, keepdim=True)))

    step_divergence = torch.zeros(len(kernels_norm)-1, kernels_norm[0].shape[-1], device=kernels_norm[0].device)
    for i in range(len(kernels_norm)-1):
        step_divergence[i] = torch.mean((kernels_norm[i]-kernels_norm[i+1]) * (torch.log(kernels_norm[i])-torch.log(kernels_norm[i+1])), dim=(0,1))

    label_divergence = torch.zeros(len(kernels_norm)-1, kernels_norm[0].shape[-1], device=kernels[0].device)
    for i in range(len(kernels_norm)-1):
        label_divergence[i] = torch.mean((kernels_norm[-1]-kernels_norm[i]) * (torch.log(kernels_norm[-1])-torch.log(kernels_norm[i])), dim=(0,1))
    
    base_divergence =  torch.zeros(len(kernels_norm)-1, kernels_norm[0].shape[-1], device=kernels_norm[0].device)
    for i in range(len(kernels_norm)-1):
        base_divergence[i] = torch.mean((kernels_norm[0]-kernels_norm[i+1]) * (torch.log(kernels_norm[0])-torch.log(kernels_norm[i+1])), dim=(0,1))

    return step_divergence, label_divergence, base_divergence

def compute_kernels(x, sigma, eps= 1e-16):
    x = x.permute(0, 2, 1) #BxTxF -> BxFxT  and this results in a kernel matrix of shape (B, B, T)

    B, K, M = x.shape

    x_norm = (x - x.mean(dim=0, keepdim=True)).to(torch.float64)
    weights = 1 / ((x.std(dim=0, keepdim=True)**2).sum(dim=1, keepdim=True) + eps)
    sqrt_weights = torch.sqrt(weights).detach()
    x_norm = x_norm * sqrt_weights * torch.sqrt(torch.tensor(x_norm.shape[1], dtype=torch.float64))


    x_reshaped_cos = x_norm.permute(2, 0, 1).reshape(M, B, K)  # Shape: (N*M, B, K)
    pairwise_dots_x = torch.einsum('abc,aec->abe', x_reshaped_cos, x_reshaped_cos).permute(1, 2, 0) # Shape: (B, K, N*M)

    x_reshaped = x_norm.view(B, K, -1)
    norms_x = torch.norm(x_reshaped, p=2, dim=1) # Shape: (B, K, N*M)

    eye = torch.eye(B).unsqueeze(-1).to(x.device)

    dist_x = (norms_x.unsqueeze(1)**2 + norms_x.unsqueeze(0)**2 - 2*pairwise_dots_x) # Shape: (B, B, N*M)

    dist_x = dist_x/x.size(1)

    diffs_x = torch.exp(-(dist_x/(2 * sigma ** 2)))

    cumulative_kernel_x = diffs_x

    return cumulative_kernel_x


