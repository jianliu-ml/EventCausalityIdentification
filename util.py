import pickle
import random
import copy
from pytorch_pretrained_bert import BertTokenizer


def all_events(datasets, positive=True, single_event=True):
    event_set = set()
    for data in datasets:
        _, token, _, e1, e2, _ = data
        e1 = [token[x] for x in e1]
        e2 = [token[x] for x in e2]
        if not positive or data[-1] != 'NULL':
            if single_event:
                event_set.add(tuple(e1))
                event_set.add(tuple(e2))
            else:
                event_set.add((tuple(e1), tuple(e2)))
    return event_set


def get_topic(dataset, topics):
    results = list()
    for data in dataset:
        t = data[0]
        if t.split('/')[-2] in topics:
            results.append(data)
    return results


def topic_c(dataset, topic1, topic2):
    data_topic1 = get_topic(dataset, topic1)
    data_topic2 = get_topic(dataset, topic2)
    
    events1 = all_events(data_topic1, positive=True, single_event=True)
    events2 = all_events(data_topic2, positive=True, single_event=True)

    inter_set = events1.intersection(events2)
    union_set = events1.union(events2)


    return len(inter_set) / len(union_set)






def select_datasets(datasets, seen_events, flag):
    seen = list()
    unseen = list()

    if flag == 'seen':
        for data in datasets:
            _, token, _, e1, e2, _ = data
            e1 = [token[x] for x in e1]
            e2 = [token[x] for x in e2]
            if tuple(e1) in seen_events and tuple(e2) in seen_events:
                seen.append(data)
            else:
                unseen.append(data) 

    if flag == 'and':
        for data in datasets:
            _, token, _, e1, e2, _ = data
            e1 = [token[x] for x in e1]
            e2 = [token[x] for x in e2]
            if tuple(e1) in seen_events and tuple(e2) in seen_events:
                seen.append(data)
            else:
                unseen.append(data)
    elif flag == 'or':
        for data in datasets:
            _, token, _, e1, e2, _ = data
            e1 = [token[x] for x in e1]
            e2 = [token[x] for x in e2]
            if (tuple(e1) in seen_events and not tuple(e2) in seen_events) or (tuple(e2) in seen_events and not tuple(e1) in seen_events):
                seen.append(data)
            else:
                unseen.append(data)
    return seen, unseen
    


def split_datasets(data, flag):
    l_data = len(data)
    training_data = data[:int(l_data/2)]
    testing_data = data[int(l_data/2):]

    if flag == 'seen' or flag == 'or':
        training_events = all_events(training_data, positive=True, single_event=True)
        seen, unseen = select_datasets(testing_data, training_events, flag)

        training_data = training_data
        testing_data = seen

    else:
        training_events = all_events(training_data, positive=True)
        seen, unseen = select_datasets(testing_data, training_events, flag)

        training_data = training_data
        testing_data = unseen

    training_data_mask = copy.deepcopy(training_data)
    testing_data_mask = copy.deepcopy(testing_data)


    for _, sen1, sen2, span1, span2, _ in training_data_mask:
        for i in span1 + span2:
            sen1[i] = 103

    for _, sen1, spen2, span1, span2, _ in testing_data_mask:
        for i in span1 + span2:
            sen1[i] = 103

    return training_data, testing_data, training_data_mask, testing_data_mask



if __name__ == '__main__':
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    data_topic1 = get_topic(data, ['18'])
    data_topic2 = get_topic(data, ['33'])
    
    events1 = all_events(data_topic1, positive=True, single_event=True)
    events2 = all_events(data_topic2, positive=True, single_event=True)

    inter_set = events1.intersection(events2)
    model_dir = '/home/jliu/data/BertModel/bert-large-uncased' # uncased better
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

    for elem in inter_set:
        print(tokenizer.convert_ids_to_tokens(list(elem)))
    print('--')
    for elem in events1:
        if elem not in inter_set:
            print(tokenizer.convert_ids_to_tokens(list(elem)))
    print('--')
    for elem in events2:
        if elem not in inter_set:
            print(tokenizer.convert_ids_to_tokens(list(elem)))

    
    with open('data/event_causality.pickle', 'rb') as f:
        data = pickle.load(f)

    print(len(data))


    # 3464, 4381, 1891
    
    # print(data[0])
    # random.seed(1234)
    # random.shuffle(data)
   
    # training_data, testing_data, training_data_mask, testing_data_mask = split_datasets(data, 'or')

    # print(len(training_data))
    # print(len(testing_data))
    
    # data = {
    #     'train': training_data,
    #     'test': testing_data
    # }
    
    # with open('data_one.pickle', 'wb') as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    
    # data = {
    #     'train': training_data_mask,
    #     'test': testing_data_mask
    # }
    
    # with open('data_one_mask.pickle', 'wb') as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)



    # all_topics = ['1', '3', '4', '5', '7', '8',
    #           '12', '13', '14', '16', '18', '19', '20',
    #           '22', '23', '24', '30', '32', '33', '35', '37', '41']

    # res = []
    # for i in all_topics:
    #     for j in all_topics:
    #         if int(j) != int(i):
    #             res.append([i, j, topic_c(data, [i], [j])])
    # res = sorted(res, key=lambda x: x[-1], reverse=True)
    # for st in all_topics:
    #     filt = list(filter(lambda x: x[0]==st, res))
    #     print(filt)
    #     print()
