from torch import nn

# Import the Inception model from torch
from torchvision.models import inception_v3
from Transformer import InceptionTransformer


class InceptionMultiHeadTransformer(nn.Module):
    """ Build a model using Inception and multi-head transformer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inception = inception_v3(pretrained=True)
        self.transformer = InceptionTransformer(config)
    
    def forward(self, inputs, mask):
        """ Forward pass.
        
        Args:
            inputs: A tensor of shape (batch_size, seq_len, input_size).
            mask: A tensor of shape (batch_size, seq_len).
        
        Returns:
            outputs: A tensor of shape (batch_size, seq_len, output_size).
        """
        outputs = self.inception(inputs)
        outputs = self.transformer(outputs, mask)
        outputs = nn.functional.log_softmax(outputs, dim=2)
        return outputs
        