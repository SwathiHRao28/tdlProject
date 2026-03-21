import os
import json
import re
from collections import Counter
import torch
import torchvision.transforms as transforms
from PIL import Image

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        # Predefined tokens
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        self.itos = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        
    def __len__(self):
        return len(self.itos)
        
    @staticmethod
    def tokenize(text):
        # Basic tokenization handling NLTK-like word boundaries
        text = text.lower()
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
        return text.strip().split()
        
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi[self.unk_token]
            for token in tokenized_text
        ]

def get_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def build_vocab_from_captions(captions, freq_threshold=5):
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions)
    return vocab
