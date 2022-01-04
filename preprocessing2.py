import pandas as pd
import stanza
from nltk import ParentedTree
import numpy as np

nlp = stanza.Pipeline(lang='en',processors='tokenize,pos,constituency')


def get_sentences():

    def parse_tree(tree,constituents):
        if type(tree) != str and tree.label() in ['NP','VP','S','SINV']:
            if len(tree.leaves()) > 1:
                if tree.label() in ['VP','S','SINV'] or (tree.label() == 'NP' and 'of' in tree.leaves()[0]):
                    if tree.leaves()[0] not in ['is','are','were','am','was','\'m','\'re','\'s','be','will','to','that']:
                        phrase = ' '.join(tree.leaves())
                        constituents.add(phrase.strip())

        if type(tree) != str:
            for child in tree:
                constituents = parse_tree(child,constituents)

        return constituents


    data = pd.read_csv('sentencesplits.csv')

    with open('sentences.txt','w') as out:
        with open('sentences_const.txt', 'w') as out2:
            for index,row in data.iterrows():
                lineid = int(row['lineid'])
                for sent in str(row['splits']).split('\t'):
                    sent = sent.replace(')','').replace('(','').replace('"','')
                    doc = nlp(sent)
                    for s in doc.sentences:
                        tree = str(s.constituency)
                        tree = ParentedTree.fromstring(tree)
                        constituents = set()
                        if tree[0].label() == 'NP': constituents.add(' '.join(tree.leaves()).strip())
                        else:
                            constituents = parse_tree(tree[0],constituents)
                        for const in constituents:
                            out2.write(str(lineid) + ',' + const + '\n')


                    out.write(str(lineid) + ',' + sent.strip() + '\n')

def add_label():

    labeledphrases = {}

    with open('/home/nitin/Desktop/dontpatronize/dontpatronize/data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv','r') as cats:
        for line in cats.readlines():
            lineid = int(line.split('\t')[0])
            if int(lineid) not in labeledphrases.keys():
                labeledphrases[lineid] = []

            phrase = line.split('\t')[-3]
            labeledphrases[lineid].append(phrase)


    with open('sentencesconstlabeled.txt','w') as out:
        with open('sentences_const.txt','r') as _in:
            for line in _in.readlines():
                if line.strip() != '':
                    lineid = int(line.strip().split(',')[0])
                    phrase = line.strip().split(',')[1]

                    if lineid not in labeledphrases.keys():
                        out.write(str(lineid) + ',' + phrase.strip() + ',' + str(0) + '\n')

            for k,v in labeledphrases.items():
                for ph in v:
                    out.write(str(k) + ',' + ph.strip() + ',' + str(1) + '\n')


#get_sentences()
add_label()
stats = []
with open('sentencesconstlabeled.txt','r') as st:
    for line in st.readlines():
        line = line.split(',')[1]
        stats.append(len(line.split(' ')))


print(np.mean(stats))
print(np.percentile(stats,50))
print(np.percentile(stats,90))
print(np.percentile(stats,95))
print(np.percentile(stats,99))



