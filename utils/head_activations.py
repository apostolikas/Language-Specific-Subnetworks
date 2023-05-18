import torch
from collections import defaultdict


def register_activation_saver(model, layer_i):

    # n_heads = model.config.num_attention_heads

    def save_activations(module, m_in, m_out):
        m_out = m_out[0][:, :1]
        batch_size = m_out.shape[0]
        m_out = m_out.view(batch_size, -1)
        model.cache[layer_i].append(m_out.detach().cpu())

    layer = model.roberta.encoder.layer[layer_i].attention
    layer.register_forward_hook(save_activations)


def register_hooks(model):

    n_layers = model.config.num_hidden_layers
    model.cache = [[] for _ in range(n_layers)]

    for layer_i in range(n_layers):
        register_activation_saver(model, layer_i)


def get_activations(model, dataloader, head_mask):

    register_hooks(model)

    for batch in dataloader:

        input_ids = batch['input_ids'].to(model.device)
        input_mask = batch['attention_mask'].to(model.device)
        label_ids = batch['labels'].to(model.device)

        with torch.no_grad():
            model(input_ids,
                  token_type_ids=None,
                  attention_mask=input_mask,
                  labels=label_ids,
                  head_mask=head_mask)

    activations = []
    for layer in model.cache:
        activations.append(torch.cat(layer, dim=0))
    activations = torch.stack(activations, dim=0)

    return activations
