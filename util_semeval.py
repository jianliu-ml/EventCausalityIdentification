def split_sentence(text):
    sentences = []
    e1 = []
    e2 = []
    for idx, word in enumerate(text.split(' ')):
        if '<e1>' in word or '</e1>' in word:
            e1.append(idx)
        if '<e2>' in word or '</e2>' in word:
            e2.append(idx)
        word = word.replace('<e1>', '').replace('</e1>', '')
        word = word.replace('<e2>', '').replace('</e2>', '')
        sentences.append(word)
    return sentences, e1, e2

def process_semeval(filename):
    datasets = []
    lines = open(filename).readlines()
    sentences = [lines[i].strip()[lines[i].strip().find('\t')+1:] for i in range(0, len(lines), 4)]
    labels = [lines[i].strip() for i in range(1, len(lines), 4)]

    for sentence, label in zip(sentences, labels):
        words, e1, e2 = split_sentence(sentence)
        words[0] = words[0].replace('"', '')
        words[-1] = words[-1].replace('"', '')

        if label.find('e1') > label.find('e2'):
            t = '1'
        elif label.find('e1') < label.find('e2'):
            t = '2'
        else:
            t = ''

        label = label.split('(')[0] + t
        datasets.append([words, list(range(e1[0], e1[-1]+1)), list(range(e2[0], e2[-1]+1)), label])
    return datasets

if __name__ == '__main__':
    datasets = process_semeval('crossdomain_data/semeval_train.txt')
    print(datasets[0])
    sem_dict = {'Other': 0}
    for data in datasets:
        label = data[-1]
        if label not in sem_dict:
            sem_dict[label] = len(sem_dict)
    print(sem_dict)