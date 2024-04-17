from pathlib import Path
import torch
import tqdm
import math
import numpy as np
from torch.optim import Adam
from torch.nn import MultiheadAttention



class PositionalEmbedding(torch.nn.Module):
    
    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=13, dropout=0.1, device = "cuda"):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len).to(device)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.device = device
       
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence).to(self.device) + self.segment(segment_label)
        return self.dropout(x)
    




class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        d_model=768,
        heads=12, 
        feed_forward_hidden=768 * 4, 
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiheadAttention(d_model, heads, batch_first=True)
        self.feed_forward = FeedForward(d_model, middle_dim= d_model *4)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask = None):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        attention_res = self.self_multihead(embeddings, embeddings, embeddings, mask)
       

        interacted = self.dropout(attention_res[0])
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class BERT(torch.nn.Module):
    def __init__(self, vocab_size, d_model = 768, n_layers = 12, heads = 12, dropout = 0.1, seq_len = 33, device = "cuda"):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size (d_model * n)
        :param n_layers: numbers of Transformer blocks (Encoder layers)
        :param attn_heads: number of attention heads
        dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        
        self.feed_forward_hidden = d_model * 4

        self.embedding = BERTEmbedding(vocab_size = vocab_size, embed_size = d_model, seq_len = seq_len, device = device)

        self.encoder_blocks = torch.nn.ModuleList([
            EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, segment_info):
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = (x == 0)
        # print(type(x==0))

        x = self.embedding(x, segment_info)
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x


    
class WinnerPrediction(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))



class MaskedLanguageModel(torch.nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim = - 1)
    
    def forward(self, x):
        return self.softmax(self.linear(x))
    


class BERTLM(torch.nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.predict_winner = WinnerPrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)
    
    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.predict_winner(x), self.mask_lm(x)
    
    def embedding(self, x, segment_label):
        return self.bert(x, segment_label)



class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
            ])

    def update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



class BERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr= 1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device='cuda'
        ):

        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.criterion(next_sent_output, data["winner_label"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            
            loss = next_loss*3.16227766017*3 + mask_loss/3.16227766017 # normalize by sqrt(10) which is square root of the length of random variable


            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["winner_label"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["winner_label"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}, \
            total_acc={total_correct * 100.0 / total_element}"
        ) 

