import json
import spacy
from tqdm import tqdm
import pickle

nlp = spacy.load('en')

def change_offset(text, target, fes):
    text = text.strip()
    sen = nlp(text)
    words = [token.text for token in sen]

    idx2i_map = {}
    for i, token in enumerate(sen):
        idx2i_map[token.idx] = i
    idx2i_map[len(text)+1] = len(sen)

    try:
        s, e = target
        target_span = [idx2i_map[s], idx2i_map[e+1]]

        roles = []
        for fe in fes:
            s, e, role = fe
            span = [idx2i_map[s], idx2i_map[e+1]]
            roles.append([role, span[0], span[1]])

        return words, target_span, roles
    except:
        print(text, target, fes)
        print(len(text))
        return None, None, None


def process_framenet(filename):
    results = []
    all_data = json.load(open(filename))
    for data in tqdm(all_data, mininterval=2, total=len(all_data)):
        framename = data['name']
        lexunit = data['lexunit']
        if not 'target' in data:
            continue
        text = data['text']
        target = data['target'][0]
        fe = data['fe'][0]

        words, target_span, roles = change_offset(text, target, fe)
        if words:
            results.append([framename, lexunit, words, target_span, roles])
    return results


if __name__ == '__main__':
    results = process_framenet('crossdomain_data/frame_examples.json')
    with open('data_framenet.pl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)