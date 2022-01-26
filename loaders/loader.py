import os
import spacy
import torch

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

from torch.nn.utils.rnn import pad_sequence

class Dataset(TorchDataset):
    def __init__(self, option, mode, src_vocab = None, trg_vocab = None):
        super(Dataset, self).__init__()
        assert mode in ['train', 'valid', 'test']
        if mode != 'train':
            assert src_vocab != None
            assert trg_vocab != None
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm' )
        data_path = \
            os.path.join(option.targets_path, option.train_file) if mode == 'train' else \
            os.path.join(option.targets_path, option.valid_file) if mode == 'valid' else \
            os.path.join(option.targets_path, option.test_file )
        all_data = self.read_data(data_path)
        src_sent = [data[0] for data in all_data]
        trg_sent = [data[1] for data in all_data]
        src_word = [self.tokenize_de(sent) for sent in src_sent]
        trg_word = [self.tokenize_en(sent) for sent in trg_sent]
        self.src_vocab = self.build_vocab(src_word, option.min_freq, option.max_numb) if mode == 'train' else src_vocab
        self.trg_vocab = self.build_vocab(trg_word, option.min_freq, option.max_numb) if mode == 'train' else trg_vocab
        src_id = [self.encode(word, self.src_vocab) for word in src_word]
        trg_id = [self.encode(word, self.trg_vocab) for word in trg_word]
        self.datas = self.aggregate(src_id, trg_id)

    def read_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding = 'utf-8') as data_file:
            for line in data_file:
                src_sent, trg_sent = line.split('\t')
                src_sent = src_sent.strip()
                trg_sent = trg_sent.strip()
                data.append((src_sent, trg_sent))
        return data

    def tokenize_de(self, text):
        return [token.text for token in self.spacy_de.tokenizer(text)][ : :-1]

    def tokenize_en(self, text):
        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    def build_vocab(self, data, min_freq, max_numb):
        counter = {}
        for sent in data:
            for word in sent:
                counter[word] = counter.get(word, 0) + 1
        word_dict = {word: count for word, count in counter.items() if count >= min_freq}
        word_list = sorted(word_dict.items(), key = lambda x: x[1], reverse = True)[:max_numb - 4]

        words = [word for word, count in word_list]
        words.insert(0, self.SOS_TOKEN)
        words.insert(0, self.EOS_TOKEN)
        words.insert(0, self.UNK_TOKEN)
        words.insert(0, self.PAD_TOKEN)

        vocab = {}
        vocab['id2word'] = {idx: word for idx, word in enumerate(words)}
        vocab['word2id'] = {word: idx for idx, word in enumerate(words)}
        vocab['special'] = {
            'SOS_TOKEN': self.SOS_TOKEN,
            'EOS_TOKEN': self.EOS_TOKEN,
            'UNK_TOKEN': self.UNK_TOKEN,
            'PAD_TOKEN': self.PAD_TOKEN
        }
        return vocab

    def encode(self, tokens, vocab):
        tokens = [self.SOS_TOKEN] + tokens + [self.EOS_TOKEN]
        tokens = [vocab['word2id'].get(token) if vocab['word2id'].get(token) else vocab['word2id'].get(self.UNK_TOKEN) for token in tokens]
        return tokens

    def aggregate(self, src_id, trg_id):
        src_lens = [len(_id) for _id in src_id]
        trg_lens = [len(_id) for _id in trg_id]
        all_data = list(zip(src_id, src_lens, trg_id, trg_lens))
        all_data = sorted(all_data, key = lambda item: item[1], reverse = True)
        return all_data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return list(self.datas[index]) + [
            self.src_vocab['word2id'].get(self.PAD_TOKEN),
            self.trg_vocab['word2id'].get(self.PAD_TOKEN)
        ]

    def get_src_vocab(self):
        return self.src_vocab

    def get_trg_vocab(self):
        return self.trg_vocab

def collate_fn(batch_data):
    src = [data[0] for data in batch_data]
    trg = [data[2] for data in batch_data]
    src_len = [data[1] for data in batch_data]
    trg_len = [data[3] for data in batch_data]
    src_pad_idx = batch_data[0][4]
    trg_pad_idx = batch_data[0][5]
    src = [torch.tensor(seq, dtype = torch.long) for seq in src]
    trg = [torch.tensor(seq, dtype = torch.long) for seq in trg]
    src = pad_sequence(src, batch_first = True, padding_value = src_pad_idx)
    trg = pad_sequence(trg, batch_first = True, padding_value = trg_pad_idx)
    src_len = torch.tensor(src_len, dtype = torch.long)
    trg_len = torch.tensor(trg_len, dtype = torch.long)
    return src, src_len, trg, trg_len

def get_loader(option):
    train_dataset = Dataset(option, 'train')
    src_vocab = train_dataset.get_src_vocab()
    trg_vocab = train_dataset.get_trg_vocab()
    valid_dataset = Dataset(option, 'valid', src_vocab, trg_vocab)
    test_dataset  = Dataset(option, 'test' , src_vocab, trg_vocab)
    train_loader = TorchDataLoader(train_dataset, batch_size = option.batch_size, shuffle = False, collate_fn = collate_fn)
    valid_loader = TorchDataLoader(valid_dataset, batch_size = option.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader  = TorchDataLoader(test_dataset , batch_size = option.batch_size, shuffle = False, collate_fn = collate_fn)
    return src_vocab, trg_vocab, train_loader, valid_loader, test_loader

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    src_vocab, trg_vocab, train_loader, valid_loader, test_loader = get_loader(option)

    # vocab
    print(type(src_vocab), src_vocab.keys()) # <class 'dict'> dict_keys(['id2word', 'word2id', 'special'])
    print(type(trg_vocab), trg_vocab.keys()) # <class 'dict'> dict_keys(['id2word', 'word2id', 'special'])
    print(len(src_vocab['word2id']), len(src_vocab['id2word'])) # 30000 30000
    print(len(trg_vocab['word2id']), len(trg_vocab['id2word'])) # 22716 22716

    # dataloader
    print(len(train_loader.dataset)) # 304341
    print(len(valid_loader.dataset)) # 16907
    print(len(test_loader .dataset)) # 16907
    for mini_batch in train_loader:
        src, src_len, trg, trg_len = mini_batch
        print(src.shape) # (batch_size, max_seq_len)
        print(trg.shape) # (batch_size, max_seq_len)
        print(src_len.shape) # (batch_size)
        print(trg_len.shape) # (batch_size)
        break
