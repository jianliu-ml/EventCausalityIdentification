import numpy as np
import pickle
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

from dataset import Dataset

from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention



class BertCausalModel(nn.Module):

    def __init__(self, y_num, model_dir):
        super(BertCausalModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_dir)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(1024 * 2, y_num)

        self.additional_fc = nn.Linear(1024 * 4, y_num)

    def forward(self, sentences_s, mask_s, event1, event1_mask, event2, event2_mask):

        if self.training:
            self.bert.train()
            encoded_layers_s, _ = self.bert(sentences_s, attention_mask=mask_s)
            enc_s = encoded_layers_s[-1]

            # encoded_layers_t, _ = self.bert(sentences_t, attention_mask=mask_t)
            # enc_t = encoded_layers_t[-1]

        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers_s, _ = self.bert(sentences_s, attention_mask=mask_s)
                enc_s = encoded_layers_s[-1]

                # encoded_layers_t, _ = self.bert(sentences_t, attention_mask=mask_t)
                # enc_t = encoded_layers_t[-1]

        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_s, event1)])
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_s, event2)])

        m1 = event1_mask.unsqueeze(-1).expand_as(event1).float()
        m2 = event2_mask.unsqueeze(-1).expand_as(event2).float()

        event1 = event1 * m1
        event2 = event2 * m2

        opt1 = torch.sum(event1, dim=1)
        opt2 = torch.sum(event2, dim=1)

        opt = torch.cat((opt1, opt2), 1)
        opt = self.drop(opt)
        opt = self.fc(opt)
        return opt

    def forward_logits(self, sentences_s, mask_s, event1, event1_mask, event2, event2_mask):

        if self.training:
            self.bert.train()
            encoded_layers_s, _ = self.bert(sentences_s, attention_mask=mask_s)
            enc_s = encoded_layers_s[-1]

            # encoded_layers_t, _ = self.bert(sentences_t, attention_mask=mask_t)
            # enc_t = encoded_layers_t[-1]

        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers_s, _ = self.bert(sentences_s, attention_mask=mask_s)
                enc_s = encoded_layers_s[-1]

                # encoded_layers_t, _ = self.bert(sentences_t, attention_mask=mask_t)
                # enc_t = encoded_layers_t[-1]

        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_s, event1)])
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_s, event2)])

        m1 = event1_mask.unsqueeze(-1).expand_as(event1).float()
        m2 = event2_mask.unsqueeze(-1).expand_as(event2).float()

        event1 = event1 * m1
        event2 = event2 * m2

        opt1 = torch.sum(event1, dim=1)
        opt2 = torch.sum(event2, dim=1)

        opt = torch.cat((opt1, opt2), 1)
        opt = self.drop(opt)
        return opt



if __name__ == '__main__':
    model = BertCausalModel(3)
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    print(data[250:251][0][2])

    dataset = Dataset(10, data[250:251])

    for batch in dataset.reader('cpu', True):
        sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y = batch
        print(sentences_s, sentences_t, event1, event2, data_y)
        opt = model(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
        print(opt)
        # print(a, b, c)
        break
