import pickle
from pytorch_pretrained_bert import BertTokenizer
import random

model_dir = '/home/jliu/data/BertModel/bert-large-uncased' # uncased better
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

def select_negative(datasets, length):
    negative_datasets = random.sample(datasets, length)
    results = list()
    for data in negative_datasets:
        name, lex, words, trigger_span, roles = data
        if len(roles) < 2:
            continue

        e1_spans = list(range(roles[0][1], roles[0][2]))
        e2_spans = list(range(roles[1][1], roles[1][2]))

        results.append([words, e1_spans, e2_spans, 'NULL'])
    return results


def get_cause_effect_pair(datasets):
    results = list()
    for data in datasets:
        name, lex, words, trigger_span, roles = data
        e1_spans = list()
        e2_spans = list()
        for role in roles:
            rolename = role[0]
            if 'Cause' in rolename:
                e1_spans = list(range(role[1], role[2]))
            if 'Effect' in rolename:
                e2_spans = list(range(role[1], role[2]))
        if len(e1_spans) + len(e2_spans) == 0:
            continue

        if len(e1_spans) * len(e2_spans) == 0:
            continue

        # if len(e1_spans) == 0:
        #     e1_spans = list(range(trigger_span[0], trigger_span[1]))

        # if len(e2_spans) == 0:
        #     e2_spans = list(range(trigger_span[0], trigger_span[1]))

        results.append([words, e1_spans, e2_spans, 'Cause-Effect'])
    return results


def get_framenet(datasets, opt_file_name, mask=True):

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

        data_set.append(['framenet', sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel])

    with open(opt_file_name, 'wb') as f:
        pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    with open('data_framenet.pl', 'rb') as f:
        dataset = pickle.load(f)
    cause_effect_datasets = get_cause_effect_pair(dataset)
    negative_datasets = select_negative(dataset, length=len(cause_effect_datasets))
    total_datasets = cause_effect_datasets + negative_datasets
    get_framenet(total_datasets, 'framenet.pickle', mask=False)
    get_framenet(total_datasets, 'framenet_mask.pickle', mask=True)