import pickle
import random
from pytorch_pretrained_bert import BertTokenizer
from concept_net_util import crawl_concept_net

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def add_knowledge(elem):
    # ['annotated_data/v1.0/12/12_1ecbplus.xml.xml', [101, 1996, 2796, 3212, 24248, 1996, 6084, 1997, 16298, 2006, 9432, 28409, 1037, 4800, 1011, 4049, 2886, 2011, 2712, 16908, 29560, 2006, 6432, 6470, 1010, 10439, 2890, 22342, 2075, 2656, 16831, 8350, 1998, 9530, 8873, 15782, 3436, 2608, 1998, 9290, 1010, 1999, 1996, 3587, 3144, 3424, 1011, 24386, 3169, 2144, 2244, 1012, 102], [101, 1996, 2796, 3212, 24248, 1996, 6084, 1997, 16298, 2006, 9432, 28409, 1037, 4800, 1011, 4049, 2886, 2011, 2712, 16908, 29560, 2006, 6432, 6470, 1010, 10439, 2890, 22342, 2075, 2656, 16831, 8350, 1998, 9530, 8873, 15782, 3436, 2608, 1998, 9290, 1010, 1999, 1996, 3587, 3144, 3424, 1011, 24386, 3169, 2144, 2244, 1012, 102], [33, 34, 35, 36], [25, 26, 27, 28], 'NULL']
    doc, sen1, sen2, e1_offset, e2_offset, label = elem
    event1 = sen1[e1_offset[0]:e1_offset[-1]+1]
    event1_tokens = tokenizer.convert_ids_to_tokens(event1)
    event1_words = ' '.join(event1_tokens)

    event2 = sen1[e2_offset[0]:e2_offset[-1]+1]
    event2_tokens = tokenizer.convert_ids_to_tokens(event2)
    event2_words = ' '.join(event2_tokens)

    event1_knowledge = ' '.join(crawl_concept_net(event1_words)[:5]) ## max to 5 piece of knowledge
    event2_knowledge = ' '.join(crawl_concept_net(event2_words)[:5])

    xx = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(event1_knowledge))
    xx_len = len(xx)
    yy = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(event2_knowledge))
    yy_len = len(yy)

    e1_start, e1_end = e1_offset[0], e1_offset[-1]
    e2_start, e2_end = e2_offset[0], e2_offset[-1]

    if e1_start < e2_start:
        temp1 = sen1[:e1_end+1] + xx + sen1[e1_end+1:]  # add e1 knowledge
        e1_end += xx_len

        e2_start += xx_len
        e2_end += xx_len

        temp2 = temp1[:e2_end+1] + yy + temp1[e2_end+1:]  # add e2 knowledge
        e2_end += yy_len
    else:
        temp1 = sen1[:e1_end+1] + xx + sen1[e1_end+1:]  # add e1 knowledge
        e1_end += xx_len

        temp2 = temp1[:e2_end+1] + yy + temp1[e2_end+1:]  # add e2 knowledge
        e2_end += yy_len

        e1_start += yy_len
        e1_end += yy_len
    
    print('_')
    print(elem)
    print([doc, temp2, temp2, list(range(e1_start, e1_end+1)), list(range(e2_start, e2_end+1)), label])

    return [doc, temp2, temp2, list(range(e1_start, e1_end+1)), list(range(e2_start, e2_end+1)), label]
    

with open('data_bert.pickle', 'rb') as f:
    data = pickle.load(f)

data_knowledge = [add_knowledge(elem) for elem in data]

with open('data_knowledge.pickle', 'wb') as f:
    pickle.dump(data_knowledge, f, pickle.HIGHEST_PROTOCOL)


# def get_sentence_number(s, all_token):
#     tid = s.split('_')[0]
#     for token in all_token:
#         if token[0] == tid:
#             return token[1]


# def nth_sentence(sen_no):
#     res = []
#     for token in all_token:
#         if token[1] == sen_no:
#             res.append(token[-1])
#     return res


# def get_sentence_offset(s, all_token):
#     positions = []
#     for c in s.split('_'):
#         token = all_token[int(c) - 1]
#         positions.append(token[2])
#     return '_'.join(positions)


# if __name__ == '__main__':
#     with open('document_raw.pickle', 'rb') as f:
#         documents = pickle.load(f)



#     data_set = []
#     for doc in documents:
#         [all_token, ecb_star_events, ecb_coref_relations,
#         ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink,
#         evaluation_data, evaluationcrof_data] = documents[doc]

#         for event1 in ecb_star_events:
#             for event2 in ecb_star_events:

#                 if event1 == event2: # event ID
#                     continue

#                 offset1 = ecb_star_events[event1]
#                 offset2 = ecb_star_events[event2]

#                 # # #Coreference Relation
#                 # rel = 'NULL'
#                 # if offset1 in ecb_coref_relations and offset2 in ecb_coref_relations[offset1]:
#                 #     rel = 'Coref'

#                 # #Causal Relation
#                 rel = 'NULL'
#                 for elem in evaluation_data:
#                     e1, e2, value = elem
#                     if e1 == offset1 and e2 == offset2:
#                         rel = value

#                 sen_s = get_sentence_number(offset1, all_token)
#                 sen_t = get_sentence_number(offset2, all_token)

#                 if abs(int(sen_s) - int(sen_t)) == 0: # #

#                     sentence_s = nth_sentence(sen_s)
#                     sentence_t = nth_sentence(sen_t)

#                     sen_offset1 = get_sentence_offset(offset1, all_token)
#                     sen_offset2 = get_sentence_offset(offset2, all_token)

#                     span1 = [int(x) for x in sen_offset1.split('_')]
#                     span2 = [int(x) for x in sen_offset2.split('_')]

#                     sentence_s = ['[CLS]'] + sentence_s + ['[SEP]']
#                     sentence_t = ['[CLS]'] + sentence_t + ['[SEP]']

#                     span1 = list(map(lambda x: x+1, span1))
#                     span2 = list(map(lambda x: x+1, span2))

#                     sentence_vec_s = []
#                     sentence_vec_t = []

#                     span1_vec = []
#                     span2_vec = []
#                     for i, w in enumerate(sentence_s):
#                         tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
#                         xx = tokenizer.convert_tokens_to_ids(tokens)

#                         if i in span1:
#                             span1_vec.extend(list(range(len(sentence_vec_s), len(sentence_vec_s) + len(xx))))

#                         sentence_vec_s.extend(xx)

#                     for i, w in enumerate(sentence_t):
#                         tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
#                         xx = tokenizer.convert_tokens_to_ids(tokens)

#                         if i in span2:
#                             span2_vec.extend(list(range(len(sentence_vec_t), len(sentence_vec_t) + len(xx))))

#                         sentence_vec_t.extend(xx)

#                     data_set.append([doc, sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel])
#                         # print(tokenizer.ids_to_tokens(sentence_vec[span1_vec[0]:span1_vec[-1]+1]))
#                         # print(tokenizer.ids_to_tokens(sentence_vec[span2_vec[0]:span2_vec[-1]+1]))

#     with open('data_bert.pickle', 'wb') as f:
#         pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)


