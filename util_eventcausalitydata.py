from bs4 import BeautifulSoup, NavigableString
import re


def precess_sentence(sentence):
    soup = BeautifulSoup(sentence)
    soup = soup.find('p')
    
    tokens = list()
    tags = list()
    for elem in soup.contents:
        if isinstance(elem, NavigableString):
            tokens.extend(elem.strip().split())
        else:
            eid = elem.get('eid')
            tid = elem.get('tid')
            
            eid = eid if eid else tid

            s = len(tokens)
            tokens.extend(elem.text.split())
            e = len(tokens)
            tags.append((eid, list(range(s, e))))
        
    return tokens, tags


def process_document(doc_name):
    filein = open(doc_name)
    for _ in range(0, 5):
        filein.readline()
    results = list()
    for line in filein:
        if line.startswith('</TEXT>'): break
        sentence = '<p>' + line.strip() + '</p>'
        tokens, tags = precess_sentence(sentence)
        results.append([tokens, tags])
    return results

def get_causal_link():
    filein = open('crossdomain_data/EventCausalityData/allClinks.txt')
    d = {}
    for line in filein:
        fname, e1, e2 = line.strip().split('\t')[:3]
        d.setdefault(fname, set())
        d[fname].add(tuple(list([e1, e2])))
        d[fname].add(tuple(list([e2, e1])))
    return d


def get_all_results():
    from os import listdir
    from os.path import isfile, join
    mypath = 'crossdomain_data/EventCausalityData/rawdata/'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    documents = {}

    for f in onlyfiles:
        key = f.split('/')[-1][:-4]
        documents[key] = process_document(f)

    for key in documents:
        print(key)
    
    event_causal_dict = get_causal_link()

    results = list()
    for key in documents:
        sentences = documents[key]
        causal_pairs = event_causal_dict[key]
        for sentence in sentences:
            tokens, tags = sentence
            events = list(filter(lambda x: x[0][0] == 'e', tags))
            
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    e1, e1_span = events[i]
                    e2, e2_span = events[j]
                    rel = 'NULL'
                    if tuple(list([e1, e2])) in causal_pairs:
                        rel = 'Cause'
                    results.append([key, tokens, e1_span, e2_span, rel])
    
    return results















###############


def get_causal_link2():
    filein = open('crossdomain_data/EventCausalityData/keys/dev.keys')
    d = {}

    for line in filein:
        if line.startswith('<DOC'):
            key = line.split('"')[1]
        elif line.startswith('</DOC') or not line.strip():
            continue
        else:
            #R 1_3 1_10
            r, e1, e2 = line.strip().split()
            d.setdefault(key, set())
            d[key].add(tuple([e1, e2, r]))
            d[key].add(tuple([e2, e1, r]))
    
    
    filein = open('crossdomain_data/EventCausalityData/keys/eval.keys')
    for line in filein:
        if line.startswith('<DOC'):
            key = line.split('"')[1]
        elif line.startswith('</DOC') or not line.strip():
            continue
        else:
            #R 1_3 1_10
            r, e1, e2 = line.strip().split()
            d.setdefault(key, set())
            d[key].add(tuple([e1, e2, r]))
            d[key].add(tuple([e2, e1, r]))    
    return d



def read_document2(filename):
    filein = open(filename)
    soup = BeautifulSoup(filein.read())
    results = list()
    for elem in soup.find_all('s3'):
        temp = elem.text.strip().split(' ')
        temp = [x.split('/')[0] for x in temp]
        results.append(temp)
    return results




def read_all_docuemnt2():
    from os import listdir
    from os.path import isfile, join
    mypath = 'crossdomain_data/EventCausalityData/eval'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    mypath = 'crossdomain_data/EventCausalityData/dev'
    onlyfiles = onlyfiles + [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    d = {}

    for f in onlyfiles:
        key = f.split('/')[-1]
        results = read_document2(f)
        d[key] = results
    return d




def get_all_results2():
    from os import listdir
    from os.path import isfile, join
    mypath = 'crossdomain_data/EventCausalityData/rawdata/'
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    documents = {}

    for f in onlyfiles:
        key = f.split('/')[-1][:-4]
        documents[key] = process_document(f)

    for key in documents:
        print(key)

    documents_modified = read_all_docuemnt2()   ### Standford nlp
    event_causal_dict = get_causal_link2()

    results = list()
    for key in documents:

        sentences = documents[key]
        sentences_modefied = documents_modified[key] 
        causal_pairs = event_causal_dict[key]

        for idx, sentence in enumerate(sentences):
            tokens, tags = sentence
            events = list(filter(lambda x: x[0][0] == 'e', tags))
            events_text = set([tokens[elem[1][0]] for elem in events])

            tokens_modified = sentences_modefied[idx]

            all_events = list()
            for idx, t in enumerate(tokens_modified):
                if t in events_text:
                    all_events.append(idx)
            all_events = all_events[:5]
                        
            for elem in causal_pairs:
                e1, e2, r = elem
                sid, epos = e1.split('_')
                if sid == str(idx):
                    all_events.append(int(epos))
                
                sid, epos = e2.split('_')
                if sid == str(idx):
                    all_events.append(int(epos))

            all_events = sorted(list(set(all_events)))

            for i in range(len(all_events)):
                for j in range(i+1, len(all_events)):
                    e1 = all_events[i]
                    e2 = all_events[j]

                    e1_key = '%d_%d' % (idx, e1)
                    e2_key = '%d_%d' % (idx, e2)

                    rel = 'NULL'

                    temp = tuple([e1_key, e2_key, 'C'])
                    if temp in causal_pairs:
                        rel = 'Cause'
                        print('Here')
                    results.append([key, tokens_modified, [e1], [e2], rel])
    
    return results







if __name__ == '__main__':

    # results = get_all_results()

    # results2 = get_all_results2()

    pass