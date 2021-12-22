import re
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class PreProcessing():
    def __init__(self,pclfile,categoriesfile):
        self.pclfile = pclfile
        self.categoriesfile = categoriesfile

        self.punctuation = r'@#^\*|~`<>\\'
        self.regextag = r'<.+>'
        self.sentencedata = None # to be dataframe of data
        self.traindata = None
        self.testdata = None
        self.lengths = []
        #self.maxlen = float('-inf')
        self.maxlen = 67


    """
    def encode_data(self):

        data = []
        for index, row in self.sentencedata:
            data.extend(row[0])

        inp = self.tokenizer(data,max_length=self.maxlen,padding='max_length',truncation=True,add_special_tokens=False,return_tensors='pt')
        output = self.model(**inp)
        print(output)
    """

    def get_maxlength(self):

        distrib = []

        for index, row in self.sentencedata.iterrows():
            splits = row[0]
            for s2 in splits.split('\t'):
                inp = self.tokenizer(s2,add_special_tokens=False)
                length = len(inp['input_ids'])
                distrib.append(length)

        self.maxlen = int(math.floor(np.percentile(np.array(distrib),99)))

    def preprocess_data(self):

        listdata = []

        with open(self.pclfile,'r') as pclf:
            for line in pclf.readlines():

                datum = []

                line = line.lower().strip()

                if line != '':

                    label = int(line.split('\t')[-1])
                    if label in [0,1]:
                        label = 0
                    else:
                        label = 1

                    lineid = int(line.split('\t')[0])

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

                    if len(splits) > 0:
                        if len(splits) == 42:
                            splits = splits[len(splits) // 2:] # discard first half of splits

                        datum.append(lineid)
                        datum.append('\t'.join(splits))
                        datum.append(len(splits))
                        datum.append(label)

                        listdata.append(datum)


        self.sentencedata = pd.DataFrame(listdata,columns=['lineid','splits','lengths','label'])
        self.sentencedata.to_csv('sentencesplits.csv',index=False)

        labels = self.sentencedata['label'].tolist()

        # extract train / test split - test data is in codalab!!
        self.traindata,self.testdata,_,_ = train_test_split(self.sentencedata,labels,stratify=labels,test_size=0.05)





def main():

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None
    pcl = PreProcessing(pclfile,categoriesfile)
    pcl.preprocess_data()
    #pcl.get_maxlength()
    #print(pcl.maxlen)


if __name__ == "__main__":
    main()