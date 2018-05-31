FOLDERS:

data:


models/lstm: 

model_1: Original architecture on PubMed-w2v-sub.txt

model_2: Original architecture on Aslog-w2v.txt

model_3: Pattern features added with embeddings for each token on PubMed-w2v-sub.txt

model_4: Pattern features added in penultimate layer on PubMed-w2v-sub.txt

model_5: Pattern features added with embeddings for each token on PubMed-w2v.txt

model_6: Original architecture on Aslog-w2v.txt with pattern features for each token.

models/crf:

1: Original CRF

2: CRF with patterns

data:

AutoSlog-TS generated patterns:
intervention_caseframes.txt
participants_caseframes.txt
outcome_caseframes.txt

w2v-collocations.txt


w2v:

PubMed-w2v.txt: Original PubMed word vectors, trained over all articles from 1900s. Vocab ~ 23 million

Aslog-w2v.txt: Word vectors trained over Human RCT articles with Aslog patterns indexed as one unit. Vocab ~ 9 million.

PubMed-w2v-sub.txt: Word vectors trained over Human RCT articles, no changes made to articles. Vocab ~ 9 million.