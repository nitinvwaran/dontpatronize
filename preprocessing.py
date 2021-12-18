import re
import math
import pandas as pd
import numpy as np

from collections import Counter
from transformers import RobertaTokenizer, RobertaModel

class PreProcessing():
    def __init__(self,pclfile,categoriesfile):
        self.pclfile = pclfile
        self.categoriesfile = categoriesfile

        self.punctuation = r'@#^\*|~`<>\\'
        self.regextag = r'<.+>'
        self.sentencedata = [] # all sentence level data
        self.lengths = []

        self.maxlen = float('-inf')

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')


    def encode_data(self):

        data = []
        for row in self.sentencedata:
            data.extend(row[0])

        inp = self.tokenizer(data,max_length=self.maxlen,padding='max_length',truncation=True,add_special_tokens=False,return_tensors='pt')
        output = self.model(**inp)
        print(output)

    def get_maxlength(self):

        distrib = []

        for row in self.sentencedata:
            splits = row[0]
            for s in splits:
                inp = self.tokenizer(s,add_special_tokens=False)
                length = len(inp['input_ids'])
                distrib.append(length)

        self.maxlen = int(math.floor(np.percentile(np.array(distrib),95)))

    def preprocess_data(self):
        with open(self.pclfile,'r') as pclf:
            for line in pclf.readlines():
                line = line.lower().strip()

                if line != '':

                    label = int(line.split('\t')[-1])
                    if label in [0,1]:
                        label = 0
                    else:
                        label = 1

                    line = line.split('\t')[4]
                    line = line.replace('&amp;',' and ').replace('&apos;','').replace('&gt;','').replace('&lt;','').replace('&quot;','"')

                    line = re.sub(self.regextag,' ',line)
                    line = re.sub(self.punctuation,' ',line)
                    line = re.sub(r'(\! )+', '!', line)
                    line = re.sub(r'\!+', ' ! ', line)
                    line = re.sub(r'(\? )+', '?', line)
                    line = re.sub(r'\?+', ' ? ', line)
                    line = re.sub(r'(\. )+', ' . ', line)
                    line = re.sub(r'\.\.+', '', line)
                    line = re.sub(r'"+','"',line)
                    line = line.replace('"',' " ')
                    line = re.sub(r'-+', '-', line)
                    line = line.replace('/',' or ')

                    line = re.sub(' +',' ',line)

                    splits = line.split(' . ')
                    splits = [s.split(' ? ') for s in splits]
                    splits = [s  for inner in splits for s in inner ]
                    splits = [s.split(' ! ') for s in splits]
                    splits = [s for inner in splits for s in inner ]
                    splits = [s.split(' : ') for s in splits]
                    splits = [s for inner in splits for s in inner]
                    splits = [s.split(' ; ') for s in splits]
                    splits = [s for inner in splits for s in inner]
                    splits = [s.split(' - ') for s in splits]
                    splits = [s for inner in splits for s in inner]

                    splits = [s.strip() for s in splits if s]
                    splits = [s for s in splits if s != '"']
                    splits = [s for s in splits if len(s.split()) >= 3]


                    self.sentencedata.append((splits,label))
                    self.lengths.append(len(splits))
            print(sum(self.lengths))



def main():

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None
    pcl = PreProcessing(pclfile,categoriesfile)
    pcl.preprocess_data()
    pcl.get_maxlength()
    print(pcl.maxlen)
    pcl.encode_data()
    print ('here')


if __name__ == "__main__":
    main()