import torch
import pickle
import random
import numpy as np
import sys

from tqdm import tqdm

from dataset import Dataset
from model import BasicCausalModel, BertCausalModel
from pytorch_pretrained_bert import BertAdam

from util import get_topic, topic_c

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
        if g != 0 and p != 0:
            c_correct += 1

    p = c_correct / (c_predict + 1e-100)
    r = c_correct / c_gold
    f = 2 * p * r / (p + r + 1e-100)

    print('correct', c_correct, 'predicted', c_predict, 'golden', c_gold)
    return p, r, f


def negative_sampling(data, ratio=0.7):
    result = []
    for d in data:
        if d[0][-1] == 'NULL':
            if random.random() < ratio:
                continue
        result.append(d)
    return result


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def train_topic(st, tt):

    device = torch.device("cuda:1")

    # with open('train_sem_mask.pickle', 'rb') as f:
    #     train_dataeval_mask_set = pickle.load(f)

    # with open('test_sem_mask.pickle', 'rb') as f:
    #     test_dataeval_mask_set = pickle.load(f)

    # with open('train_sem.pickle', 'rb') as f:
    #     train_dataeval_set = pickle.load(f)

    # with open('test_sem.pickle', 'rb') as f:
    #     test_dataeval_set = pickle.load(f)

    # with open('framenet.pickle', 'rb') as f:
    #     test_framenet_set = pickle.load(f)

    # with open('framenet_mask.pickle', 'rb') as f:
    #     test_framenet_mask_set = pickle.load(f)

    # with open('data_seen.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # train_set, test_set = data['train'], data['test']

    # with open('data_seen_mask.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # train_set_mask, test_set_mask = data['train'], data['test']




    ### Reading data...
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    # train_set, test_set = split_train_test(data)
    train_set = get_topic(data, st)
    test_set = get_topic(data, tt)

    with open('data_mask.pickle', 'rb') as f:
        data_mask = pickle.load(f)
    #train_set_mask, test_set_mask = split_train_test(data)
    train_set_mask = get_topic(data_mask, st)
    test_set_mask = get_topic(data_mask, tt)


    test_set, test_set_mask = test_set, test_set_mask

    train_pair = list(zip(train_set, train_set_mask))
    train_pair = negative_sampling(train_pair, 0.8)
    train_set, train_set_mask = [d[0] for d in train_pair], [d[1] for d in train_pair]

    ###
    test_dataset = Dataset(10, test_set)
    test_dataset_mask = Dataset(10, test_set_mask)

    test_dataset_batch = [batch for batch in test_dataset.reader(device, False)]
    test_dataset_mask_batch = [batch for batch in test_dataset_mask.reader(device, False)]

    test_dataset_mix = list(zip(test_dataset_batch, test_dataset_mask_batch))


    ###
    train_dataset = Dataset(20, train_set)
    train_dataset_mask = Dataset(20, train_set_mask)

    train_dataset_batch = [batch for batch in train_dataset.reader(device, False)]
    train_dataset_mask_batch = [batch for batch in train_dataset_mask.reader(device, False)]

    train_dataset_mix = list(zip(train_dataset_batch, train_dataset_mask_batch))


    model = BertCausalModel(3).to(device)
    model_mask = BertCausalModel(3).to(device)


    learning_rate = 1e-5
    optimizer = BertAdam(model.parameters(), lr=learning_rate)
    optimizer_mask = BertAdam(model_mask.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


    for _ in range(0, 20):
        idx = 0
        for batch, batch_mask in tqdm(train_dataset_mix, mininterval=2, total=len(train_dataset_mix), file=sys.stdout, ncols=80):
            idx += 1
            model.train()
            model_mask.train()
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
            sentences_s_mask = batch_mask[0]

            opt = model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
            opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)

            opt_mix = torch.cat([opt, opt_mask], dim=-1)
            logits = model.additional_fc(opt_mix)
            loss = loss_fn(logits, data_y)

            optimizer.zero_grad()
            optimizer_mask.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_mask.step()



        model.eval()
        model_mask.eval()
        with torch.no_grad():
            predicted_all = []
            gold_all = []
            for batch, batch_mask in test_dataset_mix:
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                sentences_s_mask = batch_mask[0]

                opt = model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
                opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s, sentences_t, mask_t, event1, event1_mask,
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

if __name__ == '__main__':



    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)
    
    # all_topics = ['1', '3', '4', '5', '7', '8',
    #           '12', '13', '14', '16', '18', '19', '20',
    #           '22', '23', '24', '30', '32', '33', '35', '37', '41']

    # res = []
    # for i in all_topics:
    #     for j in all_topics:
    #         if int(j) != int(i):
    #             res.append([i, j, topic_c(data, [i], [j])])
    # res = sorted(res, key=lambda x: x[-1], reverse=True)

    # source_topics = ['8', '13', '18']
    # for st in source_topics:
    #     filt = list(filter(lambda x: x[0]==st, res))
    #     print(filt)
    #     tt = filt[-1]
    #     print(tt)
    #     train_topic([tt[0]], [tt[1]])

    #     tt = filt[-8]
    #     print(tt)
    #     train_topic([tt[0]], [tt[1]])

    #     tt = filt[0]
    #     print(tt)
    #     train_topic([tt[0]], [tt[1]])
    

    ### train
    device = torch.device("cuda:1")


    # with open('data/causaltb_train.pickle', 'rb') as f:
    #     train_set = pickle.load(f)
    
    # with open('data/causaltb_test.pickle', 'rb') as f:
    #     test_set = pickle.load(f)

    
    # with open('data/causaltb_train_mask.pickle', 'rb') as f:
    #     train_set_mask = pickle.load(f)
    
    # with open('data/causaltb_test_mask.pickle', 'rb') as f:
    #     test_set_mask = pickle.load(f)


    # with open('data/event_causality.pickle', 'rb') as f:
    #     data = pickle.load(f)

    #     test_set_file = ['2010.01.13.google.china.exit', '2010.01.03.japan.jal.airlines.ft',
    #         '2010.01.07.winter.weather', 
    #         '2010.01.12.turkey.israel', 
    #         '2010.01.12.uk.islamist.group.ban']
    
    # train_set = list(filter(lambda x: not x[0] in test_set_file, data))
    # test_set = list(filter(lambda x: x[0] in test_set_file, data))


    # with open('data/event_causality_mask.pickle', 'rb') as f:
    #     data = pickle.load(f)

    #     test_set_file = ['2010.01.13.google.china.exit', '2010.01.03.japan.jal.airlines.ft',
    #         '2010.01.07.winter.weather', 
    #         '2010.01.12.turkey.israel', 
    #         '2010.01.12.uk.islamist.group.ban']
    
    # train_set_mask = list(filter(lambda x: not x[0] in test_set_file, data))
    # test_set_mask = list(filter(lambda x: x[0] in test_set_file, data))

    
    ######## do
    with open('data/event_causality_do.pickle', 'rb') as f:
        data = pickle.load(f)

        # test_set_file = ['2010.01.13.google.china.exit', '2010.01.03.japan.jal.airlines.ft',
        #     '2010.01.07.winter.weather', 
        #     '2010.01.12.turkey.israel', 
        #     '2010.01.12.uk.islamist.group.ban']
        
        test_set_file = ['2010.02.03.cross.quake.resistant.housing', '2010.01.13.haiti.un.mission', '2010.03.02.health.care', '2010.01.03.japan.jal.airlines.ft', '2010.01.07.water.justice', '2010.01.18.sherlock.holmes.tourism.london', '2010.01.02.pakistan.attacks', '2010.03.22.africa.elephants.ivory.trade', '2010.02.06.iran.nuclear', '2010.02.26.census.redistricting', '2010.02.05.sotu.crowley.column', '2010.03.23.how.get.headhunted', '2010.03.17.france.eta.policeman', '2010.01.06.tennis.qatar.federer.nadal', '2010.01.01.iran.moussavi', '2010.02.07.japan.prius.recall.ft', '2010.01.07.winter.weather', '2010.03.02.japan.unemployment.ft', '2010.01.18.uk.israel.livni', '2010.01.12.uk.islamist.group.ban']
    
    train_set = list(filter(lambda x: not x[0] in test_set_file, data))
    test_set = list(filter(lambda x: x[0] in test_set_file, data))




    with open('data/event_causality_do_mask.pickle', 'rb') as f:
        data = pickle.load(f)

        # test_set_file = ['2010.01.13.google.china.exit', '2010.01.03.japan.jal.airlines.ft',
        #     '2010.01.07.winter.weather', 
        #     '2010.01.12.turkey.israel', 
        #     '2010.01.12.uk.islamist.group.ban']
        
        test_set_file = ['2010.02.03.cross.quake.resistant.housing', '2010.01.13.haiti.un.mission', '2010.03.02.health.care', '2010.01.03.japan.jal.airlines.ft', '2010.01.07.water.justice', '2010.01.18.sherlock.holmes.tourism.london', '2010.01.02.pakistan.attacks', '2010.03.22.africa.elephants.ivory.trade', '2010.02.06.iran.nuclear', '2010.02.26.census.redistricting', '2010.02.05.sotu.crowley.column', '2010.03.23.how.get.headhunted', '2010.03.17.france.eta.policeman', '2010.01.06.tennis.qatar.federer.nadal', '2010.01.01.iran.moussavi', '2010.02.07.japan.prius.recall.ft', '2010.01.07.winter.weather', '2010.03.02.japan.unemployment.ft', '2010.01.18.uk.israel.livni', '2010.01.12.uk.islamist.group.ban']
    
    train_set_mask = list(filter(lambda x: not x[0] in test_set_file, data))
    test_set_mask = list(filter(lambda x: x[0] in test_set_file, data))



    print('Train', len(train_set), 'Test', len(test_set))



    train_pair = list(zip(train_set, train_set_mask))
    train_pair = negative_sampling(train_pair)
    train_set, train_set_mask = [d[0] for d in train_pair], [d[1] for d in train_pair]

    ###
    test_dataset = Dataset(10, test_set)
    test_dataset_mask = Dataset(10, test_set_mask)

    test_dataset_batch = [batch for batch in test_dataset.reader(device, False)]
    test_dataset_mask_batch = [batch for batch in test_dataset_mask.reader(device, False)]

    test_dataset_mix = list(zip(test_dataset_batch, test_dataset_mask_batch))


    ###
    train_dataset = Dataset(20, train_set)
    train_dataset_mask = Dataset(20, train_set_mask)

    train_dataset_batch = [batch for batch in train_dataset.reader(device, False)]
    train_dataset_mask_batch = [batch for batch in train_dataset_mask.reader(device, False)]

    train_dataset_mix = list(zip(train_dataset_batch, train_dataset_mask_batch))


    model = BertCausalModel(3).to(device)
    model_mask = BertCausalModel(3).to(device)


    learning_rate = 1e-5
    optimizer = BertAdam(model.parameters(), lr=learning_rate)
    optimizer_mask = BertAdam(model_mask.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


    for _ in range(0, 100):
        idx = 0
        for batch, batch_mask in tqdm(train_dataset_mix, mininterval=2, total=len(train_dataset_mix), file=sys.stdout, ncols=80):
            idx += 1
            model.train()
            model_mask.train()
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
            sentences_s_mask = batch_mask[0]

            opt = model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
            opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)

            opt_mix = torch.cat([opt, opt_mask], dim=-1)
            logits = model.additional_fc(opt_mix)
            loss = loss_fn(logits, data_y)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model_mask.parameters(), 1)

            optimizer.zero_grad()
            optimizer_mask.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_mask.step()



        model.eval()
        model_mask.eval()
        with torch.no_grad():
            predicted_all = []
            gold_all = []
            for batch, batch_mask in test_dataset_mix:
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                sentences_s_mask = batch_mask[0]

                opt = model.forward_logits(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
                opt_mask = model_mask.forward_logits(sentences_s_mask, mask_s, sentences_t, mask_t, event1, event1_mask,
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