import pickle
import os
import sys

import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

from preprocess import load_dataset, WebNLGTokenizer


class WebNLG(Dataset):
    def __init__(self, mode="train", _path="dataset/webnlg_release_v3.0", sentence_length=32):
        self.mode = mode
        self.tokenizer = torch.load("cache/tokenizer.pkl") 
        
        cache_path = os.path.join("cache", mode+".pkl")
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            dataset = load_dataset(root_path=_path, mode=mode)
            self.data = []

            for sentence, triples in dataset:
                encoded_sentence = self.tokenizer.encode_sentence(sentence, sentence_length)
                if encoded_sentence is not None:
                    
                    triple_idx = [0] * self.tokenizer.n_triples
                    for t in triples:
                        triple_idx[self.tokenizer.triples[t]] = 1
                    self.data.append((torch.tensor(encoded_sentence), torch.tensor(triple_idx)))
            
            torch.save(self.data, cache_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_func(batch_tensor):
    batch_tensor = sorted(batch_tensor, key=lambda s: -sum(s[0] != 0))
    batch_len = list(map(lambda s: sum(s[0] != 0), batch_tensor))
    sentence_tensor = list(map(lambda x: x[0], batch_tensor))
    triple_tensor = list(map(lambda x: x[1], batch_tensor)) 

    return torch.stack(sentence_tensor, dim=0), torch.stack(triple_tensor, dim=0), torch.stack(batch_len, dim=0)


if __name__ == "__main__":
    dataset1 = WebNLG("train")
    dataset2 = WebNLG("dev")
