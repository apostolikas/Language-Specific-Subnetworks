import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class StitchNet(nn.Module):

    def __init__(self, ckp1: str, ckp2: str, layer_idx: int):
        super().__init__()
        # Note: these models are in eval mode by default
        self.front_model = AutoModelForSequenceClassification.from_pretrained(ckp1)
        self.end_model = AutoModelForSequenceClassification.from_pretrained(ckp2)
        self.layer_idx = layer_idx
        self.hidden_size = self.front_model.config.hidden_size

        self.transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self._freeze_nets()
        self.front_model.hooks = []
        self.end_model.hooks = []
        self._register_default_hooks()

    def load_masks(self, mask1: str, mask2: str):
        self.front_mask = torch.load(mask1).to(self.front_model.device)
        self.end_mask = torch.load(mask2).to(self.end_model.device)

    def find_optimal_init(self, loader):
        """As suggested in Csiszarik et al (2020), we initialize the transformation
           matrix with pseudo inverse between activations
        """

        device = self.front_model.device

        # Set hooks for saving activation at layer
        self.remove_hooks()
        self._register_both_fw_hooks()

        # Save activations
        act1 = torch.empty(0, self.hidden_size).to(device)
        act2 = torch.empty(0, self.hidden_size).to(device)
        for batch in loader:
            batch = {k: v.to(device) for (k, v) in batch.items()}
            with torch.no_grad():
                self.front_model(**batch)
                self.end_model(**batch)
                act1 = torch.cat([act1, self.front_model.activation.view(-1, self.hidden_size)])
                act2 = torch.cat([act2, self.end_model.activation.view(-1, self.hidden_size)])
            if len(act1) > 2000:
                break

        # Initialize weights
        optimal_w = self._pseudo_inverse(act1, act2)
        with torch.no_grad():
            self.transform.weight.data = optimal_w

        # Reset hooks
        self.remove_hooks()
        self._register_default_hooks()

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            self.front_model(*args, head_mask=self.front_mask, **kwargs)
        activation = self.front_model.activation
        self.front_model.activation.activation = None
        transformed_activation = self.transform(activation)
        self.end_model.activation = transformed_activation
        outputs = self.end_model(*args, head_mask=self.end_mask, **kwargs)
        return outputs

    def _freeze_nets(self):
        for param in self.front_model.parameters():
            param.requires_grad = False
        for param in self.end_model.parameters():
            param.requires_grad = False

    # =======================================================================
    # Hook registrations
    # =======================================================================

    def _register_default_hooks(self):
        self.front_model.hooks.append(self._register_fw_hook(self.front_model))
        self.end_model.hooks.append(self._register_load_hook(self.end_model))

    def _register_both_fw_hooks(self):
        self.front_model.hooks.append(self._register_fw_hook(self.front_model))
        self.end_model.hooks.append(self._register_fw_hook(self.end_model))

    def _register_fw_hook(self, model):

        def _save_activations_hook(module, m_in, m_out):
            m_out = m_out[0]
            model.activation = m_out.detach()

        layer = model.roberta.encoder.layer[self.layer_idx].attention
        hook = layer.register_forward_hook(_save_activations_hook)
        return hook

    def _register_load_hook(self, model):

        def _override_activations_hook(module, m_in, m_out):
            activation = model.activation.clone()
            model.activation = None
            return (activation,)

        layer = model.roberta.encoder.layer[self.layer_idx].attention
        hook = layer.register_forward_hook(_override_activations_hook)
        return hook

    def remove_hooks(self):
        for hook in self.front_model.hooks:
            hook.remove()
        for hook in self.end_model.hooks:
            hook.remove()
        self.front_model.hooks = []
        self.end_model.hooks = []

    # =======================================================================
    # Hook registrations
    # =======================================================================

    def _pseudo_inverse(self, x1, x2):
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)

        if not x1.shape[0] == x2.shape[0]:
            raise ValueError('Spatial size of compared neurons must match when ' \
                            'calculating psuedo inverse matrix.')

        # Calculate pseudo inverse
        A_ones = torch.matmul(torch.linalg.pinv(x1), x2).T

        # Get weights and bias
        w = A_ones

        return w
