import json, re
import numpy as np
import nltk
from nltk import ngrams, word_tokenize, sent_tokenize, pos_tag
import gensim, logging
import os
import sys
import collections
from gensim import models
from gensim.models import Phrases
from gensim.models.keyedvectors import KeyedVectors
annotypes = ['Participants', 'Intervention', 'Outcome']
path = '/nlp/data/romap/set/'
#path = '/Users/romapatel/Desktop/set/'

def tokenize(s):
    """
    :param s: string of the abstract
    :return: list of word with original positions
    """
    def white_char(c):
        return c.isspace() or c in [',', '?']
    res = []
    i = 0
    while i < len(s):
        while i < len(s) and white_char(s[i]): i += 1
        l = i
        while i < len(s) and (not white_char(s[i])): i += 1
        r = i
        if s[r-1] == '.':       # consider . a token
            res.append( (s[l:r-1], l, r-1) )
            res.append( (s[r-1:r], r-1, r) )
        else:
            res.append((s[l:r], l, r))
    return res

def load_sentences(dirname):
    doc_count, fin_sentences = 0, []
    
    for fname in os.listdir(dirname):
        if fname == '.DS_Store': continue
        sentences = []
        if doc_count > 6000000: break
        for line in open(os.path.join(dirname, fname)):
            if len(line.strip()) == 0: continue
            flag = False
            while True:
                try:
                    sentence = line.decode('ascii').encode('ascii')
                    break
                except UnicodeDecodeError: flag = True
                except UnicodeEncodeError: flag = True
                break
            if flag: continue
            sents = sent_tokenize(line.strip())
            sentences = ([word_tokenize(sent) for sent in sents])
        doc_count +=1 
        fin_sentences.extend(sentences)
        
    return fin_sentences

def load_sentences_bigram(dirname):
    doc_count, fin_sentences = 0, []
    
    for fname in os.listdir(dirname):
        if fname == '.DS_Store': continue
        sentences = []
        if doc_count > 6000000: break
        for line in open(os.path.join(dirname, fname)):
            if len(line.strip()) == 0: continue
            flag = False
            while True:
                try:
                    sentence = line.decode('ascii').encode('ascii')
                    break
                except UnicodeDecodeError: flag = True
                except UnicodeEncodeError: flag = True
                break
            if flag: continue
            sents = sent_tokenize(line.strip())
            sentences = ([word_tokenize(sent) for sent in sents])
            for sent in sentences:
                sentence = []
                bigrams = ngrams(sent, 2)
                for bigram in bigrams:
                    sentence.append('_'.join(item for item in bigram))
                fin_sentences.append(sentence)
        doc_count +=1 
        
    return fin_sentences

def train_aslog():
    sentences = load_sentences('/nlp/data/romap/aslog_indexed_docs/')
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, size=200)
    model.wv.save_word2vec_format('/nlp/data/romap/naacl-pattern/w2v/Aslog-w2v.txt', binary=False)

def train_unigram():
    sentences = load_sentences('/nlp/data/romap/unigram_docs/')
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, size=200)
    model.wv.save_word2vec_format('/nlp/data/romap/naacl-pattern/w2v/PubMed-w2v-sub.txt', binary=False)

def train_bigram():
    sentences = load_sentences_bigram('/nlp/data/romap/unigram_docs/')
    print len(sentences)
    model = gensim.models.Word2Vec(sentences, size=200)
    model.wv.save_word2vec_format('/nlp/data/romap/naacl-pattern/w2v/Bigram-w2v.txt', binary=False)


def train_phrasal():
    sentences = load_sentences('/nlp/data/romap/unigram_docs/')
    bigram = Phrases()
    for sentence in sentences:
        bigram.add_vocab([sentence])
    print len(sentences)
    
    model = gensim.models.Word2Vec(bigram[sentences], size=200)
    model.wv.save_word2vec_format('/nlp/data/romap/naacl-pattern/w2v/Collocation-w2v.txt', binary=False)


if __name__ == '__main__':
    train_mode = sys.argv[1]
    if train_mode == '1': train_aslog()
    if train_mode == '2': train_unigram()
    if train_mode == '3': train_bigram()
    if train_mode == '4': train_phrasal()


        
    '''index_dict = index_patterns()
    patterns, rev = index_dict.keys(), {}
    for pattern in patterns: rev[index_dict[pattern]] = pattern
    
    for fname in os.listdir('/nlp/data/romap/aslog_docs/'):
        docid = fname.split('/')[-1]
        docid = docid.split('.')[0]
        convert_docs(rev, index_dict, '/nlp/data/romap/aslog_docs/' + fname, docid)'''

