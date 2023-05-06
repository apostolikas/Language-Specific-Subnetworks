#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the entropy of the head attentions
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def compute_importance(model,
                       eval_dataloader,
                       device,
                       compute_importance=True,
                       head_mask=None,
                       mlp_mask=None):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """

    # Prepare our tensors
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    mlp_importance = torch.zeros(n_layers).to(device)
    head_importance = torch.zeros(n_layers, n_heads).to(device)

    if mlp_mask is None and compute_importance:
        mlp_mask = torch.ones(n_layers).to(device)
        mlp_mask.requires_grad_(requires_grad=True)
        head_mask = torch.ones(n_layers, n_heads).to(device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        mlp_mask = mlp_mask.clone().detach()
        mlp_mask.requires_grad_(requires_grad=True)
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)

    preds = None
    labels = None
    tot_tokens = 0.0

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=label_ids,
                        head_mask=head_mask,
                        mlp_mask=mlp_mask)
        loss, logits, _ = outputs
        loss.backward(
        )  # Backpropagate to populate the gradients in the head mask

        if compute_importance:
            mlp_importance += mlp_mask.grad.abs().detach()
            head_importance += head_mask.grad.abs().detach()
            mlp_mask.grad *= 0
            head_mask.grad *= 0

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels,
                               label_ids.detach().cpu().numpy(),
                               axis=0)

        tot_tokens += input_mask.float().detach().sum().data

    if compute_importance:
        # Normalize
        mlp_importance /= tot_tokens

        exponent = 2
        norm_by_layer = torch.pow(
            torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        mlp_importance = (mlp_importance - mlp_importance.min()) / (
            mlp_importance.max() - mlp_importance.min())
        head_importance = (head_importance - head_importance.min()) / (
            head_importance.max() - head_importance.min())

        # Print/save matrices
        # np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())
        # np.save(os.path.join(args.output_dir, "mlp_importance.npy"), mlp_importance.detach().cpu().numpy())

    return head_importance, mlp_importance, preds, labels


def mask_model(args, model, eval_dataloader, device):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    head_importance, mlp_importance, preds, labels = compute_importance(
        model, eval_dataloader, device=device)
    preds = np.argmax(
        preds,
        axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    original_score = (preds == labels).mean()

    abs_threshold = args.masking_threshold * original_score
    print(
        f"Pruning: original score: {original_score}, threshold: {abs_threshold:.3f}",
    )

    new_mlp_mask = torch.ones_like(mlp_importance)
    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    best_score = original_score
    current_score = best_score
    iteration = 0

    while current_score >= original_score * args.masking_threshold:

        best_score = current_score
        # Head New mask
        head_mask = new_head_mask.clone()  # save current head mask
        ###################### heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask
                   ) == num_to_mask or head_importance.view(-1)[
                       head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.bert.config.num_attention_heads
            head_idx = head.item() % model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())

        if not selected_heads_to_mask:
            break

        ################################### MLP new mask
        mlp_mask = new_mlp_mask.clone()  # save current mlp mask
        iteration += 1

        # mlps from least important to most - keep only not-masked heads
        mlp_importance[mlp_mask == 0.0] = float("Inf")
        current_mlps_to_mask = mlp_importance.sort()[1]
        mlp_to_mask = current_mlps_to_mask[0]

        if mlp_importance[mlp_to_mask] == float("Inf"):
            break

        new_mlp_mask[mlp_to_mask] = 0.0

        # Compute metric and head,mlp importance again
        head_importance, mlp_importance, preds, labels = compute_importance(
            model,
            eval_dataloader,
            device=device,
            head_mask=new_head_mask,
            mlp_mask=new_mlp_mask)

        preds = np.argmax(preds, axis=1)
        current_score = (preds == labels).mean()

        # MLP score
        print(f"MLP Masking: current score: {current_score:.3f}", end=" ")
        print(f"remaining mlps {new_mlp_mask.sum()}", end=" ")
        print(
            f"({new_mlp_mask.sum() / new_mlp_mask.numel() * 100:.1f} percents)"
        )

        # Head score
        print(f"Head Masking: current score: {current_score}", end=" ")
        print(f"remaining heads {new_head_mask.sum()}", end=" ")
        print(
            f"({new_head_mask.sum() / new_head_mask.numel() * 100:.1f} percents)"
        )

    print("Finding additional head masks")
    current_score = best_score
    new_head_mask = head_mask
    # Only Heads
    while current_score >= original_score * args.masking_threshold:
        # Head New mask
        head_mask = new_head_mask.clone()  # save current head mask
        iteration += 1
        best_score = current_score
        ###################### heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask
                   ) == num_to_mask // 2 or head_importance.view(-1)[
                       head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.bert.config.num_attention_heads
            head_idx = head.item() % model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())

        if not selected_heads_to_mask:
            break

        # Compute metric and head,mlp importance again
        head_importance, mlp_importance, preds, labels = compute_importance(
            model,
            eval_dataloader,
            device=device,
            head_mask=new_head_mask,
            mlp_mask=mlp_mask)

        preds = np.argmax(preds, axis=1)
        current_score = (preds == labels).mean()
        # Head score
        print(f"Head Masking: current score: {current_score}", end=" ")
        print(f"remaining heads {new_head_mask.sum()}", end=" ")
        print(
            f"({new_head_mask.sum() / new_head_mask.numel() * 100:.1f} percents)"
        )

    print("Finding additional MLP masks")
    current_score = best_score
    new_mlp_mask = mlp_mask
    while current_score >= original_score * args.masking_threshold:
        best_score = current_score

        ################################### MLP new mask
        mlp_mask = new_mlp_mask.clone()  # save current mlp mask

        iteration += 1
        # mlps from least important to most - keep only not-masked heads
        mlp_importance[mlp_mask == 0.0] = float("Inf")
        current_mlps_to_mask = mlp_importance.sort()[1]
        mlp_to_mask = current_mlps_to_mask[0]

        if mlp_importance[mlp_to_mask] == float("Inf"):
            break
        new_mlp_mask[mlp_to_mask] = 0.0

        # Compute metric and head,mlp importance again
        head_importance, mlp_importance, preds, labels = compute_importance(
            model,
            eval_dataloader,
            device=device,
            head_mask=head_mask,
            mlp_mask=new_mlp_mask)

        preds = np.argmax(preds, axis=1)
        current_score = (preds == labels).mean()

        # MLP score
        print(f"MLP Masking: current score: {current_score:.3f}", end=" ")
        print(f"remaining mlps {new_mlp_mask.sum()}", end=" ")
        print(
            f"({new_mlp_mask.sum() / new_mlp_mask.numel() * 100:.1f} percents)"
        )

    # np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())
    # np.save(os.path.join(args.output_dir, "mlp_mask.npy"), mlp_mask.detach().cpu().numpy())

    return head_mask, mlp_mask


def prune_model(args, model, eval_dataloader, device, head_mask, mlp_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_importance(model,
                                             eval_dataloader,
                                             device=device,
                                             compute_importance=False,
                                             head_mask=head_mask,
                                             mlp_mask=mlp_mask)
    preds = np.argmax(preds, axis=1)
    score_masking = (preds == labels).mean()
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())

    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [
            h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()
        ]
        heads_to_prune[layer] = heads_to_mask
    assert sum(
        len(h)
        for h in heads_to_prune.values()) == (1 -
                                              head_mask.long()).sum().item()

    # TODO
    print(f"Pruning heads..")
    model.prune_heads(heads_to_prune)

    mlps_to_prune = [h[0] for h in (1 - mlp_mask.long()).nonzero().tolist()]

    # TODO
    print(f"Pruning mlps..")
    model.prune_mlps(mlps_to_prune)

    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_importance(model,
                                             eval_dataloader,
                                             device=device,
                                             compute_importance=False,
                                             head_mask=None,
                                             mlp_mask=None)
    preds = np.argmax(preds, axis=1)
    score_pruning = (preds == labels).mean()
    new_time = datetime.now() - before_time

    print(f"Pruning: original num of params: {original_num_params},", end=" ")
    print(f"after pruning {pruned_num_params}", end=" ")
    print(f"{pruned_num_params / original_num_params * 100:.2f} percents)")

    print(
        f"Pruning: score with masking: {score_masking:2f} score with pruning: {score_pruning:.2f}"
    )
    print(
        f"Pruning: speed ratio (original timing / new timing): {original_time / new_time * 100:.2} percents"
    )

    return model
