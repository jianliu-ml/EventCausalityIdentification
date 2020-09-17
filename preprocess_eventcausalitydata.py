import pickle
import random
from pytorch_pretrained_bert import BertTokenizer
from util_eventcausalitydata import get_all_results, get_all_results2

model_dir = '/home/jliu/data/BertModel/bert-large-uncased' # uncased better
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)


def convert_to_bert_examples(datasets, opt_file_name, mask=True):
    data_set = []

    for data in datasets:
        doc_name, words, span1, span2, rel = data
        sentence_s = words[:]
        sentence_t = words[:]

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

        if mask:
            for i in span1_vec:
                sentence_vec_s[i] = 103
            for i in span2_vec:
                sentence_vec_s[i] = 103

        data_set.append([doc_name, sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel])

    with open(opt_file_name, 'wb') as f:
        pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    dataset = get_all_results()
    convert_to_bert_examples(dataset, 'data/event_causality_mask.pickle', mask=True)
    convert_to_bert_examples(dataset, 'data/event_causality.pickle', mask=False)


    dataset = get_all_results2()
    convert_to_bert_examples(dataset, 'data/event_causality_do_mask.pickle', mask=True)
    convert_to_bert_examples(dataset, 'data/event_causality_do.pickle', mask=False)