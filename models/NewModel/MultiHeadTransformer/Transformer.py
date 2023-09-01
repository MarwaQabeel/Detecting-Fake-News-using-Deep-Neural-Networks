# Build a full transformer model using Inception as the encoder.
from torch import nn
from torchvision.models import inception_v3
from torch.nn import Transformer
import torch.nn.functional as F
import torch


class InceptionTransformer(nn.Module):
    """ Build a model using Inception and transformer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inception = inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(2048, config["d_model"])
        # self.transformer = TransformerTranslator(config.input_size, config.output_size, config.hidden_dim, config.num_heads, config.dim_feedforward, config.max_length, config.device)
        self.transformer = Transformer(d_model=config["d_model"], nhead=config["n_heads"], num_encoder_layers=config["n_layers"], num_decoder_layers=config["n_layers"], dim_feedforward=config["dim_feedforward"], dropout=config["dropout"], activation=config["activation"], custom_encoder=None, custom_decoder=None)
        
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1]).to(self.config["device"])
        x = self.inception(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        x = F.log_softmax(x, dim=2)
        return x