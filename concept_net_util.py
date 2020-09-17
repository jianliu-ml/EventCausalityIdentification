import requests
def crawl_concept_net(event):
    obj = requests.get('http://api.conceptnet.io/c/en/' + event).json()
    relations = ['CapableOf', 'IsA', 'HasProperty', 'Causes', 'MannerOf', 'CausesDesire', 'UsedFor', 'HasSubevent', 'HasPrerequisite', 'NotDesires', 'PartOf', 'HasA', 'Entails', 'ReceivesAction', 'UsedFor', 'CreatedBy', 'MadeOf', 'Desires']
    res = []
    for e in obj['edges']:
        if e['rel']['label'] in relations:
            res.append(' '.join([e['rel']['label'], e['end']['label']]))
    return res

if __name__ == '__main__':
    res = crawl_concept_net('tsunami')
    for r in res:
        print(r[2], r[0], r[1])

    res = crawl_concept_net('drive')
    for r in res:
        print(r[2], r[0], r[1])

    res = crawl_concept_net('smoke')
    for r in res:
        print(r[2], r[0], r[1])
