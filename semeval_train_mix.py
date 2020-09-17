import torch
import pickle
import random
import numpy as np
import os

from tqdm import tqdm

from semeval_dataset import Dataset, DatasetPair
from semeval_model import BertCausalModel
from pytorch_pretrained_bert import BertAdam


def split_train_test(dataset):
    train_set = []
    test_set = []

    test_topic = ['1', '3', '4', '5', '7', '8,'
              '12', '13', '14', '16', '18', '19', '20'
              '22', '23']
    for data in dataset:
        t = data[0]
        if t.split('/')[-2] in test_topic:
            test_set.append(data)
        else:
            train_set.append(data)
    return train_set, test_set


def compute_f1(gold, predicted):
    c_predict = 0
    c_correct = 0
    c_gold = 0

    for g, p in zip(gold, predicted):
        if g != 0:
            c_gold += 1
        if p != 0:
            c_predict += 1
        if g != 0 and p != 0 and p == g:
            c_correct += 1

    p = c_correct / (c_predict + 1e-100)
    r = c_correct / c_gold
    f = 2 * p * r / (p + r + 1e-100)

    print('correct', c_correct)
    print('predicted', c_predict)
    print('golden', c_gold)

    return p, r, f


def negative_sampling(data, ratio=0.7):
    result = []
    for d in data:
        if d[0][-1] == 'NULL':
            if random.random() < ratio:
                continue
        result.append(d)
    return result

def filter_dataset(datasets, length):
    return list(filter(lambda x: len(x[1])<length, datasets))


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    with open('train_sem_mask.pickle', 'rb') as f:
        train_dataeval_mask_set = pickle.load(f)

    with open('test_sem_mask.pickle', 'rb') as f:
        test_dataeval_mask_set = pickle.load(f)


    with open('train_sem.pickle', 'rb') as f:
        train_dataeval_set = pickle.load(f)

    with open('test_sem.pickle', 'rb') as f:
        test_dataeval_set = pickle.load(f)

    print(len(train_dataeval_set), len(test_dataeval_set))

    train_set = filter_dataset(train_dataeval_set, 40)
    train_set_mask = filter_dataset(train_dataeval_mask_set, 40)
    test_set = filter_dataset(test_dataeval_set, 1000)
    test_set_mask = filter_dataset(test_dataeval_mask_set, 1000)

    print(len(train_set), len(test_set))

    train_pair = list(zip(train_set, train_set_mask))
    #train_pair = negative_sampling(train_pair)
    train_dataset = DatasetPair(3, 40, train_pair)

    test_pair = list(zip(test_set, test_set_mask))
    test_dataset = DatasetPair(3, 80, test_pair)

    model_dir = '/home/jliu/data/BertModel/bert-large-uncased'
    model = BertCausalModel(20, model_dir).to(device)
    model_mask = BertCausalModel(20, model_dir).to(device)

    learning_rate = 1e-5
    optimizer = BertAdam(model.parameters(), lr=learning_rate)
    optimizer_mask = BertAdam(model_mask.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


    while True:
        idx = 0
        for batch in train_dataset.get_tqdm(device, True):
            idx += 1
            model.train()
            model_mask.train()
            sentences_s, mask_s, sentences_s_mask, event1, event1_mask, event2, event2_mask, data_y = batch

            opt = model.forward_logits(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
            opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s, event1, event1_mask, event2, event2_mask)

            opt_mix = torch.cat([opt, opt_mask], dim=-1)
            logits = model.additional_fc(opt_mix)
            loss = loss_fn(logits, data_y)

            optimizer.zero_grad()
            optimizer_mask.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_mask.step()


            if not idx % 200 == 0:
                continue
            model.eval()
            model_mask.eval()
            with torch.no_grad():
                predicted_all = []
                gold_all = []
                for batch in test_dataset.reader(device, True):
                    sentences_s, mask_s, sentences_s_mask, event1, event1_mask, event2, event2_mask, data_y = batch

                    opt = model.forward_logits(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
                    opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s, event1, event1_mask,
                                                         event2, event2_mask)

                    opt_mix = torch.cat([opt, opt_mask], dim=-1)
                    logits = model.additional_fc(opt_mix)

                    predicted = torch.argmax(logits, -1)
                    predicted = list(predicted.cpu().numpy())
                    predicted_all += predicted

                    gold = list(data_y.cpu().numpy())
                    gold_all += gold
                p, r, f = compute_f1(gold_all, predicted_all)
                print(p, r, f)
                print('Here')
