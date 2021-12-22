import re
import math
import pandas as pd
import numpy as np
import stanza

from sklearn.model_selection import train_test_split
from tqdm import tqdm


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

        self.nlp = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma,depparse')


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
        counter = 0

        with open(self.pclfile,'r') as pclf:
            for line in tqdm(pclf.readlines()):

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
                    splits = [s for inner in splits for s in inner]
                    splits = [s.split(' ! ') for s in splits]
                    splits = [s for inner in splits for s in inner]
                    splits = [s.split(' : ') for s in splits]
                    splits = [s for inner in splits for s in inner]
                    splits = [s.split(' ; ') for s in splits]
                    splits = [s for inner in splits for s in inner]
                    splits = [s.split(' - ') for s in splits]
                    splits = [s for inner in splits for s in inner]

                    splits = [s.strip() for s in splits if s]
                    splits = [s for s in splits if len(s.split()) >= 2]


                    if len(splits) > 0:
                        alltokens = []
                        alldeps = []

                        for split in splits:
                            doc = self.nlp(split)
                            if len(doc.sentences) > 1:
                                counter += 1
                                maxlen = float('-inf')
                                for sent in doc.sentences:
                                    if len(sent.words) > maxlen:
                                        maxlen = len(sent.words)
                                        tokens = [w.text for w in sent.words]
                                        deps = [w.deprel for w in sent.words]

                            elif len(doc.sentences) == 1:
                                tokens = [w.text for w in doc.sentences[0].words]
                                deps = [w.deprel for w in doc.sentences[0].words]

                            alltokens.append(' '.join(tokens))
                            alldeps.append(' '.join(deps))

                        assert len(alltokens) == len(splits)
                        assert len(alldeps) == len(alltokens)

                        if len(splits) > 40:
                            alltokens = alltokens[len(alltokens) // 2:] # discard first half of splits
                            alldeps = alldeps[len(alldeps) // 2: ]

                        datum.append(lineid)
                        datum.append('\t'.join(alltokens))
                        datum.append('\t'.join(alldeps))
                        datum.append(len(alltokens))
                        datum.append(label)

                        listdata.append(datum)
                    else:
                        datum.append(lineid)
                        d = self.nlp(line.lower().strip())
                        datum.append(' '.join([w.text for sent in d.sentences for w in sent.words]))
                        datum.append(' '.join([w.deprel for sent in d.sentences for w in sent.words]))
                        datum.append(1)
                        datum.append(label)

                        listdata.append(datum)

        self.sentencedata = pd.DataFrame(listdata,columns=['lineid','splits','deps','lengths','label'])
        print('paragraphs processed:' + str(len(self.sentencedata)))
        self.sentencedata.to_csv('sentencesplits.csv',index=False)

        labels = self.sentencedata['label'].tolist()

        # extract train / test split - test data is in codalab!!
        self.traindata,self.testdata,_,_ = train_test_split(self.sentencedata,labels,stratify=labels,test_size=0.05,random_state=42)
        print ('wrong splits:' + str(counter))



def main():

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None
    pcl = PreProcessing(pclfile,categoriesfile)
    pcl.preprocess_data()
    #pcl.get_maxlength()
    #print(pcl.maxlen)


if __name__ == "__main__":
    main()