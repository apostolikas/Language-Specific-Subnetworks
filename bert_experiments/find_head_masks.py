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

from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def compute_heads_importance(args,
                             model,
                             eval_dataloader,
                             compute_entropy=True,
                             compute_importance=True,
                             head_mask=None):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    device = model.roberta.device
    n_layers, n_heads = model.roberta.config.num_hidden_layers, model.roberta.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)

    preds = None
    labels = None
    tot_tokens = 0.0
    # Note: optimizer won't optimize anthing, lr doesn't matter
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):

        input_ids = batch['input_ids'].to(device)
        input_mask = batch['attention_mask'].to(device)
        label_ids = batch['labels'].to(device)

        # batch = tuple(t.to(device) for t in batch)
        # input_ids, input_mask, segment_ids, label_ids = batch

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids,
                        token_type_ids=None,
                        attention_mask=input_mask,
                        labels=label_ids,
                        head_mask=head_mask)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        optimizer.zero_grad()
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

        tot_tokens += input_mask.float().detach().sum().data

    if compute_entropy:
        # Normalize
        attn_entropy /= tot_tokens

    if compute_importance:
        # Normalize
        head_importance /= tot_tokens
        # Layerwise importance normalization
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not args.dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (head_importance.max() -
                                                                           head_importance.min())

        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=device)
        head_ranks = head_ranks.view_as(head_importance)

    return attn_entropy, head_importance, preds, labels


def mask_heads(args, model, eval_dataloader, task_name):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    preds = np.argmax(preds, axis=1)
    original_score = (preds == labels).mean()

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    i = 0
    while current_score >= original_score * args.masking_threshold:
        head_mask = new_head_mask.clone()  # save current head mask

        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[
                    head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.roberta.config.num_attention_heads
            head_idx = head.item() % model.roberta.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())

        if not selected_heads_to_mask:
            break

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        preds = np.argmax(preds, axis=1)
        current_score = (preds == labels).mean()

        print(f"Head Masking: current score: {current_score}", end=" ")
        print(f"remaining heads {new_head_mask.sum()}", end=" ")
        print(f"({new_head_mask.sum() / new_head_mask.numel() * 100:.1f} percents)")

    return head_mask


def prune_heads(args, model, eval_dataloader, head_mask, task_name):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    preds = np.argmax(preds, axis=1)
    score_masking = (preds == labels).mean()
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()

    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=None
    )
    preds = np.argmax(preds, axis=1)
    score_pruning = (preds == labels).mean()
    new_time = datetime.now() - before_time

    print(f"Pruning: original num of params: {original_num_params},", end=" ")
    print(f"after pruning {pruned_num_params}", end=" ")
    print(f"{pruned_num_params / original_num_params * 100:.2f} percents)")

    print(
        f"Pruning: score with masking: {score_masking:2f} score with pruning: {score_pruning:.2f}")
    print(
        f"Pruning: speed ratio (original timing / new timing): {original_time / new_time * 100:.2} percents"
    )
