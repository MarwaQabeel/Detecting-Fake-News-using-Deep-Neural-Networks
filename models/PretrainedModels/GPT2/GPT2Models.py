from torch import nn
from transformers import GPT2TokenizerFast
from transformers import AutoModel
import torch


MAX_LENGHT = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """

    def __init__(self, input_size, output_size, device, hidden_dim=768, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=MAX_LENGHT):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        # initialize word embedding layer
        self.embeddingL = nn.Embedding(input_size, hidden_dim).to(device)
        # initialize positional embedding layer
        self.posembeddingL = nn.Embedding(max_length, hidden_dim).to(device)

        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)

        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(
            self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.hidden_dim)).to(device)
        self.norm_ff = nn.LayerNorm(self.hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_size).to(device)
        self.dropout = nn.Dropout(0.5)


    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
        # embeds = self.embed(inputs)
        hidden_states = self.multi_head_attention(inputs)
        outputs = self.feedforward_layer(hidden_states)
        scores = self.final_layer(self.dropout(outputs))
        return scores

    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        # Word Embedding
        word_emb = self.embeddingL(inputs)
        # Positional Encoding
        positions = torch.arange(
            self.max_length, device=self.device).unsqueeze(0)
        pos_encode = self.posembeddingL(positions)
        x = pos_encode + word_emb
        return x

    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)

        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        # Multi-head Attention
        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)
        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)

        # print("Debug: q1_shape", q1.shape)
        # print("Debug: k1_shape", k1.shape)
        # print("Debug: q2_shape", q2.shape)
        # print("Debug: k2_shape", k2.shape)
        # print("Debug: v2_shape", v2.shape)
                
        z1 = torch.bmm(q1, k1.transpose(1, 2)) / np.sqrt(self.dim_k)
        z1 = torch.softmax(z1, dim=-1)
        z11 = torch.bmm(z1, v1)
        z2 = torch.bmm(q2, k2.transpose(1, 2)) / np.sqrt(self.dim_k)
        z2 = torch.softmax(z2, dim=-1)
        z22 = torch.bmm(z2, v2)
        z = torch.cat((z11, z22), dim=-1)
        z = self.attention_head_projection(z)
        z = self.norm_mh(z + inputs)
        return z

    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        outputs = self.feedforward(inputs)
        outputs = self.norm_ff(outputs + inputs)
        return outputs

    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        outputs = self.output_layer(inputs)
        return outputs


class GPT2Pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768).to(device)
        self.activation = nn.ReLU().to(device)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GPT2Pooler2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2, 2).to(device)
        self.activation = nn.ReLU().to(device)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GPT2Model_2FC(nn.Module):
    def __init__(self, model, dropout=0.2):
      super(GPT2Model_2FC, self).__init__()
      self.model = model.to(device)
      self.pooler = GPT2Pooler()

      self.dropout = nn.Dropout(dropout).to(device)
      self.relu =  nn.ReLU().to(device)
      self.fc1 = nn.Linear(768,512).to(device)

      self.sigmoid = nn.Sigmoid().to(device)
      self.fc2 = nn.Linear(512,2).to(device)
      self.softmax = nn.LogSoftmax(dim=1).to(device)
      self.tanh = nn.Tanh().to(device)
      return

    def forward(self, sent_id, mask):
      output_model = self.model(sent_id, attention_mask=mask)
      cls_hs = output_model["last_hidden_state"]
      x = self.pooler(cls_hs)
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x


class GPT2Model_1FC(nn.Module):
    def __init__(self, model, dropout=0.2):
      super(GPT2Model_1FC, self).__init__()
      self.model = model.to(device)
      self.pooler = GPT2Pooler()
      self.dropout = nn.Dropout(dropout).to(device)
      self.relu =  nn.ReLU().to(device)
      self.fc1 = nn.Linear(768,2).to(device)
      self.softmax = nn.LogSoftmax(dim=1).to(device)
      return

    def forward(self, sent_id, mask):
      output_model = self.model(sent_id, attention_mask=mask)
      cls_hs = output_model["last_hidden_state"]
      x = self.pooler(cls_hs)
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.softmax(x)
      return x


class GPT2Model_2FC_Norm(nn.Module):
    def __init__(self, model, dropout=0.2):
      super(GPT2Model_2FC_Norm, self).__init__()
      self.model = model.to(device)
      self.pooler = GPT2Pooler()
      self.norm_mh = nn.LayerNorm(768).to(device)
      self.dropout = nn.Dropout(dropout).to(device)
      self.relu =  nn.ReLU().to(device)
      self.fc1 = nn.Linear(768,512).to(device)
      # self.fc1 = nn.Linear(768,2).to(device)
      self.sigmoid = nn.Sigmoid().to(device)
      self.fc2 = nn.Linear(512,2).to(device)
      self.softmax = nn.LogSoftmax(dim=1).to(device)
      self.tanh = nn.Tanh().to(device)      
      return

    def forward(self, sent_id, mask):
      output_model = self.model(sent_id, attention_mask=mask)
      cls_hs = output_model["last_hidden_state"]
      x = self.pooler(cls_hs)
      x = self.norm_mh(x)
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x


class GPT2Model_1FC_Norm(nn.Module):
    def __init__(self, model, dropout=0.2):
      super(GPT2Model_1FC_Norm, self).__init__()
      self.model = model.to(device)
      self.pooler = GPT2Pooler()
      self.norm_mh = nn.LayerNorm(768).to(device)
      self.fc1 = nn.Linear(768,2).to(device)
      self.relu =  nn.ReLU().to(device)
      self.dropout = nn.Dropout(dropout).to(device)
      self.softmax = nn.LogSoftmax(dim=1).to(device)
      return

    def forward(self, sent_id, mask):
      output_model = self.model(sent_id, attention_mask=mask)
      cls_hs = output_model["last_hidden_state"]
      x = self.pooler(cls_hs)
      x = self.norm_mh(x)
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.softmax(x)
      return x


class GPT2Model_Trans(nn.Module):
    def __init__(self, model, dropout=0.2):
      super(GPT2Model_Trans, self).__init__()
      self.model = model.to(device)
      self.dropout = nn.Dropout(dropout).to(device)
      self.pooler2 = GPT2Pooler2()
      self.softmax = nn.LogSoftmax(dim=1).to(device)
      self.trans_model = TransformerTranslator(
            vocab_size, output_size, device, max_length=MAX_LENGHT).to(device)
      return

    def forward(self, sent_id, mask):
      output_model = self.model(sent_id, attention_mask=mask)
      cls_hs = output_model["last_hidden_state"]
      x = self.trans_model(cls_hs)
      x = self.dropout(x)
      x = self.pooler2(x)
      x = self.softmax(x)
      return x
  

gpt2 = AutoModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2.resize_token_embeddings(len(tokenizer))


vocab_size = len(tokenizer.get_vocab())
output_size = 2


model = GPT2Model_Trans(gpt2)

