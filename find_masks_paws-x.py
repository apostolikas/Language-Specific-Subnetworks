import argparse
import logging
import os
from datetime import datetime
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import evaluate
import random
from dataclasses import dataclass

# @dataclass
# class SmartCollator():
#   pad_token_id: int

#   def __call__(self, batch):
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)    

def collate_function(batch):
    pad_token_id = tokenizer.pad_token_id
    batch_inputs = list()
    batch_attention_masks = list()
    labels = list()
    max_size = max([len(ex['input_ids']) for ex in batch])
    for item in batch:
        #batch_inputs += [pad_seq(item['input_ids'], max_size, self.pad_token_id)]
        batch_inputs += [pad_seq(item['input_ids'], max_size, pad_token_id)]
        batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
        labels.append(item['labels'])
    return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
            } 

def pad_seq(seq,max_batch_len, pad_value):
  return seq + (max_batch_len - len(seq))*[pad_value]  

accuracy_metric = evaluate.load("accuracy")
logger = logging.getLogger(__name__)

def set_all_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def fetch_data(args):
    dataset_en = load_dataset("paws-x", args.language)
    train_dataset = dataset_en['train']
    valid_dataset = dataset_en['validation']
    test_dataset = dataset_en['test']
    return train_dataset,valid_dataset,test_dataset


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

#! add logger
def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(args, model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = 12, 12
    
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)
    
    preds = None
    labels = None
    tot_tokens = 0.0

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        # ! str -> to(device) ?
 
        inp_ids = batch['input_ids'] # list of tensors
        att_mask = batch['attention_mask'] # list of tensors
        label = batch['labels'] # list of tensors
        
        inp_ids = [i_id.to(args.device) for i_id in inp_ids] # list comprehension for faster computation -> to device
        att_mask = [a_mask.to(args.device) for a_mask in att_mask]
        label = [lbl.to(args.device) for lbl in label]

        inp_ids = torch.LongTensor(inp_ids) #! batch_size, seq_length = input_shape -> wrong shape 
        att_mask = torch.FloatTensor(att_mask) 
        label = torch.ShortTensor(label) 

        #batch = tuple(batch.to(args.device) for t in batch)
        #inp_ids, att_mask, label = batch

        #! tokenizer gives input_ids and attention mask and label, token type ids = ? 
        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids=inp_ids, attention_mask=att_mask, labels=label, head_mask=head_mask)

        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last #! hidden states not needed
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * att_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label.detach().cpu().numpy(), axis=0)

        tot_tokens += att_mask.float().detach().sum().data

    if compute_entropy:
        # Normalize
        attn_entropy /= tot_tokens
        np.save(os.path.join(args.output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
        logger.info("Attention entropies")
        print_2d_tensor(attn_entropy)
    if compute_importance:
        # Normalize
        head_importance /= tot_tokens
        # ! By default arg = True
        # Layerwise importance normalization
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        #! By default arg = True
        if not args.dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

        # Print/save matrices
        np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())
        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels




def mask_heads(args, model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    preds = np.argmax(preds, axis=1) #if args.output_mode == "classification" else np.squeeze(preds)
    # ! usually computes val acc
    #original_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]

    original_score = accuracy_metric.compute(references=labels, predictions=preds)
    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * args.masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    i = 0
    while current_score >= original_score * args.masking_threshold:            
        head_mask = new_head_mask.clone()  # save current head mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // 12 #model.bert.config.num_attention_heads
            head_idx = head.item() % 12 #model.bert.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())
                
        if not selected_heads_to_mask:
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))
        
        #new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        #current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]

        current_score = accuracy_metric.compute(references=labels, predictions=preds)
        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )
        
    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(args, model, eval_dataloader, head_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    #score_masking = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    score_masking = accuracy_metric.compute(references=labels, predictions=preds)
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    logger.info(f"{heads_to_prune}")
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=None
    )
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score_pruning = accuracy_metric.compute(references=labels, predictions=preds)
    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)

#! up2here


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default='./Finetune_PAWS-X_results/checkpoint-30000',
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--output_dir",
        default='./Pruning_PAWS-X_results',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_mask_all_iterations", action="store_true", help="Saves the masks and importance scores in all iterations"
    )
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )

    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--language", default="en", type=str, help="Language of the dataset")

    args = parser.parse_args()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    set_all_seeds(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
       torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.local_rank == 0:
       torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    #model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)    
    #model.to(args.device)

    def preprocess_function(ex_dataset):
        encodings = tokenizer(ex_dataset['sentence1'], ex_dataset['sentence2'])
        targets = torch.tensor(ex_dataset['label'],dtype=torch.long)
        encodings.update({'labels': targets})
        return encodings



    _,valid_dataset,_ = fetch_data(args) #! 10% of original size
    encoded_val_dataset = list(map(preprocess_function,valid_dataset))

    eval_dataloader = torch.utils.data.DataLoader(encoded_val_dataset, batch_size=args.batch_size, num_workers=2, collate_fn=collate_function(tokenizer,))

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if not os.path.exists(args.output_dir):
        os.mkdir(os.path.join(args.output_dir, "run_args.bin"))

    # Print/save training arguments
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    if args.try_masking and args.masking_threshold > 0.0 and args.masking_threshold < 1.0:
        head_mask = mask_heads(args, model, eval_dataloader)
        prune_heads(args, model, eval_dataloader, head_mask)
    else:
        #Compute head entropy and importance score
        compute_heads_importance(args, model, eval_dataloader)


if __name__ == "__main__":
    main()