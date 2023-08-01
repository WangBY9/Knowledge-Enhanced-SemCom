import os
import sys
import random
import xml.dom.minidom

import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize


os.chdir(sys.path[0])


def read_xml_file(file_path):
    DOMTree = xml.dom.minidom.parse(file_path)
    entries = DOMTree.documentElement.getElementsByTagName('entry')

    ret = []
    for entry in entries:
        triple_list = []
        triples = entry.getElementsByTagName('modifiedtripleset')[0].getElementsByTagName('mtriple')
        for triple_node in triples:
            triple_raw_str = triple_node.firstChild.data
            triple_h, triple_r, triple_t = triple_raw_str.split(' | ')
            triple_list.append((triple_h, triple_r, triple_t))     
        
        sentences = entry.getElementsByTagName('lex')
        for sentence in sentences:
            ret.append([sentence.firstChild.data, triple_list])

    return ret


def load_dataset(root_path, mode="train"):
    assert mode in ["train", "dev"]
    real_path = os.path.join(root_path, "en", mode)
    files = os.listdir(real_path)
    ret = []

    for file_name in files:
        file_path = os.path.join(real_path, file_name)
        if os.path.isfile(file_path):
            if os.path.splitext(file_name)[-1] == ".xml":
                ret.extend(read_xml_file(file_path))
        else:
            for nf in os.listdir(file_path):
                files.append(os.path.join(file_name, nf))

    return ret


class WebNLGTokenizer:
    def __init__(self, dataset) -> None:
        self.vocab = {
            "[PAD]": 0,
            "[SOS]": 1,
            "[EOS]": 2
        }
        self.entites = {}
        self.relations = {}
        self.triples = {}   
        self.n_words = 3
        self.n_ents = 0
        self.n_rels = 0
        self.n_triples = 0
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        for sentence, triples in dataset:
            self.update_words(sentence)
            self.update_triples(triples)

    def update_words(self, sentence):
        words = word_tokenize(sentence)
        for word in words:
            if word.lower() not in self.vocab:
                self.vocab[word.lower()] = self.n_words
                self.n_words += 1
    
    def update_triples(self, triples):
        for triple in triples:
            if triple[0] not in self.entites:
                self.entites[triple[0]] = self.n_ents
                self.n_ents += 1
            if triple[2] not in self.entites:
                self.entites[triple[2]] = self.n_ents
                self.n_ents += 1
            if triple[1] not in self.relations:
                self.relations[triple[1]] = self.n_rels
                self.n_rels += 1
            if tuple(triple) not in self.triples:
                self.triples[tuple(triple)] = self.n_triples
                self.n_triples += 1

    def encode_sentence(self, sentence, sentence_length):
        words = word_tokenize(sentence)
        ret = [self.sos_token_id]

        if len(words) + 2 > sentence_length:
            return None
        
        for word in words:
            word = word.lower()
            ret.append(self.vocab[word])
        ret.append(self.eos_token_id)
        while len(ret) < sentence_length:
            ret.append(self.pad_token_id)       
        return ret



if __name__ == '__main__':
    full_dataset = load_dataset("dataset/webnlg_release_v3.0", "train")
    full_dataset.extend(load_dataset("dataset/webnlg_release_v3.0", "dev"))

    tokenizer = WebNLGTokenizer(full_dataset)
    torch.save(tokenizer, "cache/tokenizer.pkl")
