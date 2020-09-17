import pickle
import random
from pytorch_pretrained_bert import BertTokenizer
from util_causaltb import generate_data

model_dir = '/home/jliu/data/BertModel/bert-large-uncased' # uncased better
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)


def get_sem_eval(datasets, opt_file_name, mask=True):
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
    label_file = 'crossdomain_data/CausalTM/Causal-TempEval3-eval.txt'
    document_dir = 'crossdomain_data/CausalTM/TempEval3-eval_COL/'
    tempeval_results = generate_data(label_file, document_dir)
    
    get_sem_eval(tempeval_results, 'tempeval_mask.pickle', mask=True)
    get_sem_eval(tempeval_results, 'tempeval.pickle', mask=False)



    label_file = 'crossdomain_data/CausalTM/Causal-TimeBank.CLINK.txt'
    document_dir = 'crossdomain_data/CausalTM/Causal-TimeBank_COL/'
    causalTB_results = generate_data(label_file, document_dir)

    get_sem_eval(causalTB_results, 'causaltb_mask.pickle', mask=True)
    get_sem_eval(causalTB_results, 'causaltb.pickle', mask=False)


    import random 
    random.shuffle(causalTB_results)
    l = int(len(causalTB_results) / 10 * 9)

    train_set = causalTB_results[:l]
    test_set = causalTB_results[l:]

    get_sem_eval(train_set, 'data/causaltb_train_mask.pickle', mask=True)
    get_sem_eval(train_set, 'data/causaltb_train.pickle', mask=False)

    get_sem_eval(test_set, 'data/causaltb_test_mask.pickle', mask=True)
    get_sem_eval(test_set, 'data/causaltb_test.pickle', mask=False)