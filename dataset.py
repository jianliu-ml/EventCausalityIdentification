import sys
import torch
import pickle
import random

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, dataset):
        super(Dataset, self).__init__()

        self.batch_size = batch_size
        self.y_label = {
            'NULL': 0,
            'null': 0,
            'FALLING_ACTION': 1,
            'PRECONDITION': 1,
            'Coref': 1,
            'Cause-Effect': 1,
            'Cause-Effect1': 1,
            'Cause-Effect2': 1,
            'Cause': 1
        }

        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))


    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        examples = list()
        sentence_len_s = [len(tup[1]) for tup in batch]
        sentence_len_t = [len(tup[2]) for tup in batch]

        max_sentence_len_s = max(sentence_len_s)
        max_sentence_len_t = max(sentence_len_t)

        event1_lens = [len(tup[2]) for tup in batch]
        event2_lens = [len(tup[3]) for tup in batch]

        sentences_s, sentences_t, event1, event2, data_y = list(), list(), list(), list(), list()
        for data in batch:
            sentences_s.append(data[1])
            sentences_t.append(data[2])
            event1.append(data[3])
            event2.append(data[4])
            y = self.y_label[data[5]] if data[5] in self.y_label else 0
            data_y.append(y)
            examples.append(data)

        sentences_s = list(map(lambda x: pad_sequence_to_length(x, max_sentence_len_s), sentences_s))
        sentences_t = list(map(lambda x: pad_sequence_to_length(x, max_sentence_len_t), sentences_t))

        event1 = list(map(lambda x: pad_sequence_to_length(x, 5), event1))
        event2 = list(map(lambda x: pad_sequence_to_length(x, 5), event2))

        mask_sentences_s = get_mask_from_sequence_lengths(torch.LongTensor(sentence_len_s), max_sentence_len_s)
        mask_sentences_t = get_mask_from_sequence_lengths(torch.LongTensor(sentence_len_t), max_sentence_len_t)

        mask_even1 = get_mask_from_sequence_lengths(torch.LongTensor(event1_lens), 5)
        mask_even2 = get_mask_from_sequence_lengths(torch.LongTensor(event2_lens), 5)

        return [torch.LongTensor(sentences_s).to(device), mask_sentences_s.to(device),
                torch.LongTensor(sentences_t).to(device), mask_sentences_t.to(device),
                torch.LongTensor(event1).to(device), mask_even1.to(device),
                torch.LongTensor(event2).to(device), mask_even2.to(device),
                torch.LongTensor(data_y).to(device), examples]


if __name__ == '__main__':
    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    dataset = Dataset(10, data[:20])
    for batch in dataset.reader('cpu', True):
        sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, y = batch
        print(sentences_s[0])
        print(mask_s[0])
        print(event1[0])
        print(event2[0])
        break

