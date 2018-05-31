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


def bigram_sim():
    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/PubMed-w2v-sub.txt'
    wv_pubmed = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(input_vector_file, binary=False)
    pubmed_vocab = []
    f = open(input_vector_file, 'r')
    for line in f:
        pubmed_vocab.append(line.strip().split(' ')[0])
        
    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/Bigram-w2v.txt'
    wv_bigram = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(input_vector_file, binary=False)
    bigram_vocab = []
    f = open(input_vector_file, 'r')
    for line in f:
        bigram_vocab.append(line.strip().split(' ')[0])


    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/Aslog-w2v.txt'
    wv_aslog = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(input_vector_file, binary=False)
    vocab = []
    f = open(input_vector_file, 'r')
    for line in f:
        vocab.append(line.strip().split(' ')[0])

    phrases = []
    for word in vocab:
        if '_' in word: phrases.append(word)
        
    f = open('/nlp/data/romap/naacl-pattern/experiments/new-space.txt', 'w+')
    f.write(str(len(phrases)) + '\n')
    f.write('Most similar words in Aslog-trained, Bigram-trained, PubMed-trained:\n\n')
    
    for word in phrases:
        tuples = wv_aslog.most_similar(positive=word, negative=None)
        f.write('Aslog: Most similar to: ' + word + '\n')

        for item in tuples:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.write('\n')

        if word in bigram_vocab:
            tuples = wv_bigram.most_similar(positive=word, negative=None)
        else:
            tuples = []
        f.write('Bigram: Most similar to: ' + word + '\n')

        for item in tuples:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.write('\n')

        words = word.split('_')

        if words[0] in pubmed_vocab:
            tuples = wv_aslog.most_similar(positive=words[0], negative=None)
        else: tuples = []
        f.write('PubMed: Most similar to: ' + words[0] + '\n')

        for item in tuples:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.write('\n')

        if words[1] in pubmed_vocab:
            tuples = wv_aslog.most_similar(positive=words[1], negative=None)
        else: tuples = []
        f.write('PubMed: Most similar to: ' + words[1] + '\n')

        for item in tuples:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.write('\n')
            
        f.write('\n\n')
        
def collocation_sim():
    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/Collocation-w2v.txt'
    wv_aslog = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(input_vector_file, binary=False)

    vocab = []
    f = open(input_vector_file, 'r')
    for line in f:
        vocab.append(line.strip().split(' ')[0])

    phrases = []
    for word in vocab:
        if '_' in word: phrases.append(word)
        
    f = open('/nlp/data/romap/naacl-pattern/experiments/collocation-space.txt', 'w+')
    f.write('Most similar words in Collocation-trained:\n\n')

    for word in phrases:
        if word not in vocab: continue
        tuples = wv_aslog.most_similar(positive=word, negative=None)
        f.write('Most similar to: ' + word + '\n')

        for item in tuples:
            f.write(item[0] + '\t' + str(item[1]) + '\n')
        f.write('\n\n')            
        
def vocab_count():
    f = open('/nlp/data/romap/naacl-pattern/experiments/vocab_count.txt', 'w+')
    
    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/PubMed-w2v-sub.txt'
    newf = open(input_vector_file, 'r')
    lines = newf.readlines()
    f.write('PubMed-w2v-sub vocab: ' + str(len(lines)) + '\n')

    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/Bigram-w2v.txt'
    newf = open(input_vector_file, 'r')
    lines = newf.readlines()
    f.write('Bigram-w2v vocab: ' + str(len(lines)) + '\n')
    
    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/Aslog-w2v.txt'
    newf = open(input_vector_file, 'r')
    lines = newf.readlines()
    f.write('Aslog-w2v vocab: ' + str(len(lines)) + '\n')

    input_vector_file = '/nlp/data/romap/naacl-pattern/w2v/Collocation-w2v.txt'
    newf = open(input_vector_file, 'r')
    lines = newf.readlines()
    f.write('Collocation-w2v vocab: ' + str(len(lines)) + '\n')

def print_collocations():
    f = open('/nlp/data/romap/naacl-pattern/experiments/collocation-space.txt', 'r')
    newf = open('/nlp/data/romap/naacl-pattern/experiments/w2v-collocations.txt', 'w+')

    for line in f:
        if 'Most similar to:' in line:
            collocation = line.strip().split(': ')[-1]
            newf.write(collocation + '\n')
            
if __name__ == '__main__':
    #print_collocations()
    #vocab_count()
    #collocation_sim()
    bigram_sim()
