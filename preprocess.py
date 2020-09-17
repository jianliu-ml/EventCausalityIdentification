import pickle
import random
import numpy as np

def build_embedding_table(word_map, wv_file, dim):
    res = [list()] * len(word_map)
    print('Reading word vector...')
    for idx, line in enumerate(open(wv_file, 'r')):
        if idx % 50000 == 0:
            print('\rReading...', idx)
        line = line.split()
        word = line[0]
        if len(line) != dim + 1 or not word in word_map:
            continue
        res[word_map[word]] = list(map(lambda t: float(t), line[1:]))

    lens = len(list(filter(lambda x: len(x)>0, res)))
    print('Hit', lens)
    print('Unknown', len(word_map) - lens)

    def _random_vector(x):
        if len(x) == 0:
            bias = 2 * np.sqrt(3.0 / dim)
            return [random.random() * bias - bias for _ in range(dim)]
        return x
    res = list(map(_random_vector, res))
    return res



if __name__ == '__main__':
    with open('document_raw.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        documents = pickle.load(f)

    word_map = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}
    word_list = ['<PAD>', '<UNK>', '<S>', '</S>']

    for doc in documents:
        [all_token, ecb_star_events, ecb_coref_relations,
        ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, 
        evaluation_data, evaluationcrof_data] = documents[doc]

        for token in all_token:
            _, _, _, word = token
            word = word.lower()
            if not word in word_map:
                word_map[word] = len(word_map)
                word_list.append(word)


    for doc in documents:
        [all_token, ecb_star_events, ecb_coref_relations,
        ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, 
        evaluation_data, evaluationcrof_data] = documents[doc]

        for i in range(len(all_token)):
            temp = list(all_token[i])
            temp[-1] = word_map[temp[-1].lower()]
            all_token[i] = temp
        

    wv_file = '/home/jliu/data/WordVector/GoogleNews-vectors-negative300.txt'
    dim = 300
    vec = build_embedding_table(word_map, wv_file, dim)

    with open('data.pickle', 'wb') as f:
        data = {
            'data': documents,
            'word_map': word_map,
            'word_list': word_list,
            'word_vector': vec
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)



    # for doc in documents:
    #     [all_token, ecb_star_events, ecb_coref_relations,
    #     ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, 
    #     evaluation_data, evaluationcrof_data] = documents[doc]

    #     for elem in ecb_star_events:
    #         print(ecb_star_events[elem])
    #     print()

    #     for elem in ecb_coref_relations:
    #         print(ecb_coref_relations[elem])
    #     print()

    #     for elem in ecb_star_time:
    #         print(ecb_star_time[elem])
    #     print()

    #     for elem in ecbstar_events_plotLink:
    #         print(elem, ecbstar_events_plotLink[elem])
    #     print()

    #     for elem in ecbstar_timelink:
    #         print(elem, ecbstar_timelink[elem])
    #     print()

    #     for elem in evaluation_data:
    #         print(elem)