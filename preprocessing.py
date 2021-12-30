import re
import math
import pandas as pd
import numpy as np
import stanza
import os

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
        self.devids = []
        self.devfile = 'data/dev_ids.txt'


    """
    def encode_data(self):

        data = []
        for index, row in self.sentencedata:
            data.extend(row[0])

        inp = self.tokenizer(data,max_length=self.maxlen,padding='max_length',truncation=True,add_special_tokens=False,return_tensors='pt')
        output = self.model(**inp)
        print(output)
    """

    def get_devids(self):
        with open(self.devfile,'r') as ds:
            for line in ds.readlines():
                line = int(line.split(',')[0])
                self.devids.append(line)


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
                    category = line.split('\t')[2].strip()
                    category = category.replace('-',' ')

                    line = line.split('\t')[4]
                    line = line.replace('&amp;',' and ').replace('&apos;','').replace('&gt;','').replace('&lt;','').replace('&quot;','"')

                    line = re.sub(self.regextag,' ',line)
                    for punct in self.punctuation:
                        line = line.replace(punct,'')

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

                    # features go here
                    theytokens = []

                    if len(splits) > 0:
                        alltokens = []
                        alldeps = []
                        allpos = []
                        allfeats = []

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
                                        pos = [w.xpos for w in sent.words]
                                        feats = [w.feats for w in sent.words]
                                        feats = [f if f is not None else '_' for f in feats]
                                        theytokens.extend([t for t in tokens if t.strip() in ['they','them','those','their','theirs']])

                            elif len(doc.sentences) == 1:
                                tokens = [w.text for w in doc.sentences[0].words]
                                deps = [w.deprel for w in doc.sentences[0].words]
                                pos = [w.xpos for w in doc.sentences[0].words]
                                feats = [w.feats for w in doc.sentences[0].words]
                                feats = [f if f is not None else '_' for f in feats ]
                                theytokens.extend(
                                    [t for t in tokens if t.strip() in ['they', 'them', 'those', 'their', 'theirs']])

                            alltokens.append(' '.join(tokens))
                            alldeps.append(' '.join(deps))
                            allpos.append(' '.join(pos))
                            allfeats.append(' '.join(feats))

                        assert len(alltokens) == len(splits)
                        assert len(alldeps) == len(alltokens)
                        assert len(allpos) == len(alldeps)
                        assert len(allfeats) == len(allpos)

                        theytokenscount = len(theytokens) / len([w for sent in alltokens for w in sent.strip(' ')])

                        if len(splits) > 40:
                            alltokens = alltokens[len(alltokens) // 2:] # discard first half of splits
                            alldeps = alldeps[len(alldeps) // 2: ]
                            allpos = allpos[len(allpos) // 2:]
                            allfeats = allfeats[len(allpos) // 2:]


                        datum.append(lineid)
                        datum.append('\t'.join(alltokens))
                        datum.append('\t'.join(alldeps))
                        datum.append('\t'.join(allpos))
                        datum.append('\t'.join(allfeats))
                        datum.append(len(alltokens))
                        datum.append(theytokenscount)
                        datum.append(category)
                        datum.append(label)

                        listdata.append(datum)
                    else:
                        datum.append(lineid)
                        d = self.nlp(line.lower().strip())

                        x = [w.text for sent in d.sentences for w in sent.words]
                        theytokens.extend([t for t in x if t.strip() in ['they', 'them', 'those', 'their', 'theirs']])

                        datum.append(' '.join([w.text for sent in d.sentences for w in sent.words]))
                        datum.append(' '.join([w.deprel for sent in d.sentences for w in sent.words]))
                        datum.append(' '.join([w.xpos for sent in d.sentences for w in sent.words]))
                        datum.append(' '.join([w.feats for sent in d.sentences for w in sent.words]))
                        datum.append(1)
                        datum.append(len(theytokens) / len(x))
                        datum.append(category)
                        datum.append(label)

                        listdata.append(datum)

        self.sentencedata = pd.DataFrame(listdata,columns=['lineid','splits','deps','xpos','feats','lengths','theytokens','category','label'])
        print('paragraphs processed:' + str(len(self.sentencedata)))
        self.sentencedata.to_csv('sentencesplits.csv',index=False)

        print('wrong splits:' + str(counter))



    def load_preprocessed_data(self):


        refinedlabelstrain = pd.read_csv('refinedlabelstrain.csv')
        refinedlabelstrain.set_index('lineid')

        refinedlabelsdev = pd.read_csv('refinedlabelsdev.csv')
        refinedlabelsdev.set_index('lineid')

        if os.path.exists('sentencesplits.csv'):
            self.sentencedata = pd.read_csv('sentencesplits.csv')
        else:
            self.preprocess_data()

        self.sentencedata.drop(['deps','xpos','feats','theytokens','category'],axis=1,inplace=True)

        self.get_devids()
        mask = self.sentencedata['lineid'].isin(self.devids)
        self.traindata = self.sentencedata.loc[~mask]
        self.testdata = self.sentencedata.loc[mask]

        self.traindata.set_index('lineid')
        self.testdata.set_index('lineid')

        self.traindata = self.traindata.merge(refinedlabelstrain, how='inner',on='lineid')
        self.testdata = self.testdata.merge(refinedlabelsdev, how='inner',on='lineid')

        trainfeatures = pd.read_csv('train_features.csv')
        testfeatures = pd.read_csv('test_features.csv')

        trainfeatures.set_index('lineid')
        testfeatures.set_index('lineid')

        self.traindata = self.traindata.merge(trainfeatures,how='inner',on='lineid')
        self.testdata = self.testdata.merge(testfeatures,how='inner',on='lineid')
        print ('Finished merging')








def main():

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None
    pcl = PreProcessing(pclfile,categoriesfile)
    pcl.load_preprocessed_data()
    #pcl.get_maxlength()
    #print(pcl.maxlen)


if __name__ == "__main__":
    main()