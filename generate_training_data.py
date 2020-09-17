import sys
import os
import os.path
from lxml import etree
import collections
import pickle


def get_feature(s, all_token):
    sens = []
    psns = []
    words = []
    
    for c in s.split('_'):
        token = all_token[int(c) - 1]

        sens.append(token[1])
        psns.append(token[2])
        words.append(str(token[3]))


    return '_'.join(sens), '_'.join(psns), '_'.join(words)


def get_sentence_no(s, all_token):
    tid = s.split('_')[0]
    for token in all_token:
        if token[0] == tid:
            return token[1]


def all_tokens(filename):
    ecbplus = etree.parse(filename, etree.XMLParser(remove_blank_text=True))
    root_ecbplus = ecbplus.getroot()
    root_ecbplus.getchildren()

    all_token = []

    for elem in root_ecbplus.findall('token'):
        temp = (elem.get('t_id'), elem.get('sentence'),
            elem.get('number'), elem.text)
        all_token.append(temp)
    return all_token





def main(argv=None):
    
    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        documents = data['data']
        word_map = data['word_map']
        word_list = data['word_list']
        word_vector = data['word_vector']


    # data format:
    # all_token
    # ecb_star_events
    # ecb_coref_relations
    # ecb_star_time
    # ecbstar_events_plotLink
    # ecbstar_timelink
    # evaluation_data
    # evaluationcrof_data

    f = open('training_data.txt', 'w')
    for key in documents:
        [all_token, ecb_star_events, ecb_coref_relations,
        ecb_star_time, ecbstar_events_plotLink, ecbstar_timelink, 
        evaluation_data, evaluationcrof_data] = documents[key]

        # for elem in evaluation_data:
        #     s, t, value = elem
        #     s_text = transfter_to_token(s, all_token)
        #     t_text = transfter_to_token(t, all_token)
        #     temp = [key, s_text, t_text, value]
        #     print(temp)


        for event1 in ecb_star_events:
            for event2 in ecb_star_events:

                if event1 == event2: # event ID
                    continue

                offset1 = ecb_star_events[event1]
                offset2 = ecb_star_events[event2]

                # every two pairs
                rel = 'NULL'
                for elem in evaluation_data:
                    e1, e2, value = elem
                    if e1 == offset1 and e2 == offset2:
                        rel = value

                sen_s = get_sentence_no(offset1, all_token)
                sen_t = get_sentence_no(offset2, all_token)

                if sen_s == sen_t:
                    s_sen, s_pos, s_word = get_feature(offset1, all_token)
                    t_sen, t_pos, t_word = get_feature(offset2, all_token)
                    print('\t'.join([key, offset1, s_sen, s_pos, s_word, 
                                          offset2, t_sen, t_pos, t_word, rel]), file=f)


if __name__ == '__main__':
    main()