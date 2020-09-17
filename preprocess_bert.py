import pickle
import random
from pytorch_pretrained_bert import BertTokenizer

def get_sentence_number(s, all_token):
    tid = s.split('_')[0]
    for token in all_token:
        if token[0] == tid:
            return token[1]


def nth_sentence(sen_no):
    res = []
    for token in all_token:
        if token[1] == sen_no:
            res.append(token[-1])
    return res


def get_sentence_offset(s, all_token):
    positions = []
    for c in s.split('_'):
        token = all_token[int(c) - 1]
        positions.append(token[2])
    return '_'.join(positions)


if __name__ == '__main__':
    with open('document_raw.pickle', 'rb') as f:
        documents = pickle.load(f)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    data_set = []
    for doc in documents:
        [all_token, ecb_star_events, ecb_coref_relations,
        ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink,
        evaluation_data, evaluationcrof_data] = documents[doc]

        for event1 in ecb_star_events:
            for event2 in ecb_star_events:

                if event1 == event2: # event ID
                    continue

                offset1 = ecb_star_events[event1]
                offset2 = ecb_star_events[event2]

                # # #Coreference Relation
                # rel = 'NULL'
                # if offset1 in ecb_coref_relations and offset2 in ecb_coref_relations[offset1]:
                #     rel = 'Coref'

                # #Causal Relation
                rel = 'NULL'
                for elem in evaluation_data:
                    e1, e2, value = elem
                    if e1 == offset1 and e2 == offset2:
                        rel = value

                sen_s = get_sentence_number(offset1, all_token)
                sen_t = get_sentence_number(offset2, all_token)

                if abs(int(sen_s) - int(sen_t)) == 0: # #
                    sentence_s = nth_sentence(sen_s)
                    sentence_t = nth_sentence(sen_t)

                    sen_offset1 = get_sentence_offset(offset1, all_token)
                    sen_offset2 = get_sentence_offset(offset2, all_token)

                    span1 = [int(x) for x in sen_offset1.split('_')]
                    span2 = [int(x) for x in sen_offset2.split('_')]

                    sentence_s = ['[CLS]'] + sentence_s + ['[SEP]']
                    sentence_t = ['[CLS]'] + sentence_t + ['[SEP]']

                    span1 = list(map(lambda x: x+1, span1))
                    span2 = list(map(lambda x: x+1, span2))

                    sentence_vec_s = []
                    sentence_vec_t = []

                    span1_vec = []
                    span2_vec = []
                    for i, w in enumerate(sentence_s):
                        tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = tokenizer.convert_tokens_to_ids(tokens)

                        if i in span1:
                            span1_vec.extend(list(range(len(sentence_vec_s), len(sentence_vec_s) + len(xx))))

                        sentence_vec_s.extend(xx)

                    for i, w in enumerate(sentence_t):
                        tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = tokenizer.convert_tokens_to_ids(tokens)

                        if i in span2:
                            span2_vec.extend(list(range(len(sentence_vec_t), len(sentence_vec_t) + len(xx))))

                        sentence_vec_t.extend(xx)
                    for i in span1_vec:
                        sentence_vec_s[i] = 103
                    for i in span2_vec:
                        sentence_vec_s[i] = 103

                    data_set.append([doc, sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel])
                        # print(tokenizer.ids_to_tokens(sentence_vec[span1_vec[0]:span1_vec[-1]+1]))
                        # print(tokenizer.ids_to_tokens(sentence_vec[span2_vec[0]:span2_vec[-1]+1]))

    print(len(data_set))
    print(data_set[0])
    with open('data_mask.pickle', 'wb') as f:
        pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
