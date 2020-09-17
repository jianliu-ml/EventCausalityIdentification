import pickle
import random
from pytorch_pretrained_bert import BertTokenizer
from util_semeval import process_semeval

model_dir = '/home/jliu/data/BertModel/bert-large-uncased' # uncased better
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)


def get_sem_eval(datasets, opt_file_name, mask=True):
    data_set = []

    for data in datasets:
        words, span1, span2, rel = data
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

        data_set.append(['sem', sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel])

    with open(opt_file_name, 'wb') as f:
        pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    datasets = process_semeval('crossdomain_data/semeval_test.txt')
    def negative_sampling(data, ratio=0.8):
        result = []
        for d in data:
            if not d[-1].startswith('Cause-Effect') :
                if random.random() < ratio:
                    continue
            result.append(d)
        return result
    datasets = negative_sampling(datasets)
    get_sem_eval(datasets, 'sem_test_mask.pickle', mask=True)
    get_sem_eval(datasets, 'sem_test.pickle', mask=False)

    # datasets = process_semeval('crossdomain_data/semeval_train.txt')
    # get_sem_eval(datasets, 'sem_train_mask.pickle', mask=True)
    # get_sem_eval(datasets, 'sem_train.pickle', mask=False)