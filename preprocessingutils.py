import pandas as pd
import stanza
from nltk import ParentedTree
import numpy as np
import re

from tqdm import tqdm

class PreprocessingUtils():

    def __init__(self,pclfile,categoriesfile,testfile):

        self.nlp = stanza.Pipeline(lang='en',processors='tokenize,pos,constituency')

        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile

        self.devfile = 'data/dev_ids.txt'
        self.devids = []
        self.labeledids = set()

        self.punctuation = r'@#^\*|~`<>\\$'
        self.regextag = r'<.+>'

        self.datafile = 'data/preprocessed.txt'

        self.constituentphrasecutoff = 5

        self.constexceptions = {'10:41 am', '16 xOSU' ,'ORLANDO , Fla .' ,'6 pm at Aotea Square'}

        self.get_categoriesdata()



    def get_categoriesdata(self):

        with open(self.categoriesfile,'r') as cats:
            for line in cats.readlines():
                lineid = int(line.split('\t')[0])

                self.labeledids.add(lineid)

    def get_devids(self):
        with open(self.devfile, 'r') as ds:
            for line in ds.readlines():
                line = int(line.split(',')[0])
                self.devids.append(line)


    def clean_string(self,line):

        line = line.strip()
        line = line.replace('&amp;', ' and ').replace('&apos;', '').replace('&gt;', '').replace('&lt;', '').replace('&quot;', '"')
        line = re.sub(self.regextag, ' ', line)

        for punct in self.punctuation:
            line = line.replace(punct, '')

        line = re.sub(r'(\! )+', '!', line)
        line = re.sub(r'\!+', ' ! ', line)
        line = re.sub(r'(\? )+', '?', line)
        line = re.sub(r'\?+', ' ? ', line)
        line = re.sub(r'\.\.+', ' ', line)
        line = re.sub(r'"+', '"', line)
        line = line.replace('"', ' ')
        line = re.sub(r'-+', ' ', line)
        line = line.replace('/', ' or ')
        line = line.replace(')', ' ')
        line = line.replace('(', ' ')
        line = line.replace('%',' percent ')
        line = re.sub(r'\.(?! )',' . ',line)
        if '6 . 30pm' in line: line = line.replace('6 . 30pm', '6:30pm')
        line = re.sub('([.])', r' \1 ', line)

        line = re.sub(r' +', ' ', line)

        return line.strip()

    def split_string(self,line):

        splits = line.split(' . ')
        splits = [s.split(' ? ') for s in splits]
        splits = [s for inner in splits for s in inner]
        splits = [s.split(' ! ') for s in splits]
        splits = [s for inner in splits for s in inner]

        splits = [s.strip() for s in splits if s]

        return splits

    def parse_tree(self,tree,constituents):

        if type(tree) != str and tree.label() in ['VP','S','SINV','NP']:
            if len(tree.leaves()) >= self.constituentphrasecutoff:
                if tree.leaves()[0] not in ['is','are','were','am','was','\'m','\'re','\'s','be','will','to','that']:
                    phrase = ' '.join(tree.leaves())
                    constituents.add(phrase.strip())

        if type(tree) != str:
            for child in tree:
                constituents = self.parse_tree(child,constituents)

        return constituents


    def get_const_parses(self,sent):

        doc = self.nlp(sent)
        constituents = set()

        for s in doc.sentences:
            tree = str(s.constituency)
            tree = ParentedTree.fromstring(tree)

            if tree[0].label() == 'NP' or len(tree[0].leaves()) < self.constituentphrasecutoff:
                constituents.add(' '.join(tree.leaves()).strip())
            else:
                constituents = self.parse_tree(tree[0], constituents)

        return constituents


    def preprocess(self):

        with open(self.datafile,'w') as dataf:
            dataf.write('lineid,text,phrase,startindex,label')

            with open(self.pclfile, 'r') as pclf:

                for line in tqdm(pclf.readlines()):

                    lineid = int(line.split('\t')[0])

                    if lineid not in self.labeledids:

                        line = line.strip()

                        if line != '':

                            label = int(line.split('\t')[-1])
                            if label in [0, 1]:
                                label = 0
                            else:
                                label = 1


                            line = line.split('\t')[4]

                            line = self.clean_string(line)
                            splits = self.split_string(line)

                            consts = set()
                            for split in splits:
                                consts.update(self.get_const_parses(split))

                            if len(consts) > 0:
                                for const in consts:
                                    if const in self.constexceptions: continue
                                    if '16 xOSU' in const: const = const.replace('16 xOSU','16xOSU')
                                    if '6 pm' in const: const = const.replace('6 pm','6pm')
                                    if '6:30 pm' in const: const = const.replace('6:30 pm', '6:30pm')

                                    startindex = line.find(const)

                                    assert startindex != -1

                                    dataf.write(str(lineid) + ',' + line + ',' + const + ',' + str(startindex) + ',' + str(0) + '\n')
                            else:
                                dataf.write(str(lineid) + ',' + line + ',' + line + ',' + str(0) + ',' + str(0) + '\n')



def main():

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'
    preprocess = PreprocessingUtils(pclfile,categoriesfile,None)
    preprocess.preprocess()


if __name__ == "__main__":
    main()