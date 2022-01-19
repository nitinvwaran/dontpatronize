import pandas as pd
import torch
import torch.nn as nn
import argparse

from preprocessingutils import PreprocessingUtils
from modules import LSTMAttention, BertModels,CNNBert
from tqdm import tqdm

class Inference():
    def __init__(self,bestmodelpath, pclfile,categoriesfile,testfile,modeltype='rnn',bertmodeltype='rawdistilbert',devbatchsize=500,rnntype='lstm',maxlenphrase=256,maxlentext=256,hiddensize=256,numlayers=2,forcnn=False):

        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile

        self.preprocess = PreprocessingUtils(pclfile,categoriesfile,testfile)
        self.preprocess.get_train_test_data(usetalkdown=False,testdata=True,forcnn=forcnn)

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.rnntype = rnntype
        self.devbatchsize = devbatchsize

        if modeltype == 'bert':
            self.model = BertModels(bertmodeltype=bertmodeltype, maxlen=maxlentext)
        elif modeltype == 'rnn':
            self.model = LSTMAttention(rnntype=rnntype,bertmodeltype=bertmodeltype,maxlentext=maxlentext,maxlenphrase=maxlenphrase,hiddensize=hiddensize,numlayers=numlayers)
        else:
            self.model = CNNBert(maxlen=maxlentext,bertmodeltype=bertmodeltype)

        checkpoint = torch.load(bestmodelpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.softmax = nn.Softmax()

    def inference_cnn(self):

        self.model.eval()

        with torch.no_grad():

            preds = pd.DataFrame()

            for j in tqdm(range(0, len(self.preprocess.testdata), self.devbatchsize)):

                if j + self.devbatchsize > len(self.preprocess.testdata):
                    df = self.preprocess.testdata.iloc[j:len(self.preprocess.testdata)]
                else:
                    df = self.preprocess.testdata.iloc[j:j + self.devbatchsize]

                df.reset_index(drop=True, inplace=True)

                logit = self.model(df)

                logitsdf = pd.DataFrame(logit.tolist(), columns=['zerologit', 'onelogit']).reset_index(drop=True)
                probdf = pd.DataFrame(self.softmax(logit).tolist(), columns=['zeroprob', 'oneprob']).reset_index(
                    drop=True)
                p = torch.argmax(logit, dim=1)

                df['preds'] = p.tolist()
                df = pd.concat([df, logitsdf, probdf], axis=1, ignore_index=True)

                preds = preds.append(df, ignore_index=True)

            preds.columns = ['lineid',  'text', 'preds', 'zerologit', 'onelogit',
                             'zeroprob', 'oneprob']


            preds = preds.set_index('lineid')

            preds.to_csv(
                'data/inference/inference_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',
                sep='\t', index=True)



    def inference(self):

        self.model.eval()

        with torch.no_grad():

            preds = pd.DataFrame()

            for j in tqdm(range(0, len(self.preprocess.testdata), self.devbatchsize)):

                if j + self.devbatchsize > len(self.preprocess.testdata):
                    df = self.preprocess.testdata.iloc[j:len(self.preprocess.testdata)]
                else:
                    df = self.preprocess.testdata.iloc[j:j + self.devbatchsize]

                df.reset_index(drop=True, inplace=True)
                if self.modeltype == 'bert':
                    _, logit = self.model(df,test=True)
                else:
                    logit = self.model(df)

                logitsdf = pd.DataFrame(logit.tolist(), columns=['zerologit', 'onelogit']).reset_index(drop=True)
                probdf = pd.DataFrame(self.softmax(logit).tolist(), columns=['zeroprob', 'oneprob']).reset_index(
                    drop=True)
                p = torch.argmax(logit, dim=1)

                df['preds'] = p.tolist()
                df = pd.concat([df, logitsdf, probdf], axis=1, ignore_index=True)

                preds = preds.append(df, ignore_index=True)

            preds.columns = ['lineid', 'category', 'text', 'phrase', 'preds', 'zerologit', 'onelogit',
                             'zeroprob', 'oneprob']

            preds.to_csv('data/proba/testproba/testproba_'+ self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',sep='\t',index=False)
            preds = preds.loc[preds.groupby(['lineid'])['preds'].idxmax()].reset_index(drop=True)
            preds.set_index('lineid')

            preds.to_csv('data/inference/inference_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',sep='\t',index=False)


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('--maxlentext', type=int, default=128)
    parser.add_argument('--maxlenphrase', type=int, default=64)
    parser.add_argument('--hiddensize', type=int, default=256)
    parser.add_argument('--numlayers', type=int, default=2)

    args = parser.parse_args()

    with open('bestmodel.txt','r') as i:
        for line in i.readlines():
            bestmodelpath = str(line).strip().split(',')[0]

    params = bestmodelpath.split('_')
    modeltype = params[1].strip()
    bertmodeltype = params[2].strip()
    rnntype = params[3].strip()

    print ('Model Type:' + modeltype)
    print ('Bert Model Type:' + bertmodeltype)
    print ('RNN Type:' + rnntype)

    if modeltype == 'cnn':
        forcnn = True
    else:
        forcnn = False


    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'

    inf = Inference(bestmodelpath=bestmodelpath,pclfile=pclfile,categoriesfile=categoriesfile,testfile=None,maxlentext=args.maxlentext,maxlenphrase=args.maxlenphrase,devbatchsize=500,modeltype=modeltype,bertmodeltype=bertmodeltype,rnntype=rnntype,hiddensize=args.hiddensize,numlayers=args.numlayers,forcnn=forcnn)
    if modeltype != 'cnn':
        inf.inference()
    else:
        inf.inference_cnn()


if __name__ == "__main__":
    main()

