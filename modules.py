import pandas as pd
import torch
import torch.nn as nn
import os,shutil
import numpy as np
import random
import argparse

from preprocessingutils import PreprocessingUtils
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from transformers import BertTokenizer,BertForSequenceClassification,DistilBertTokenizer,DistilBertForSequenceClassification



class BertModels(nn.Module):
    def __init__(self,bertmodeltype,maxlen=512):
        super(BertModels,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if bertmodeltype == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-cased')
        elif bertmodeltype == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

        self.model.to(self.device)

        self.maxlen = maxlen


    def forward(self,dataframe):

        texts = dataframe['text'].tolist()
        phrases = dataframe['phrase'].tolist()

        labels = dataframe['label'].tolist()
        labels = torch.LongTensor(labels)
        labels = labels.to(self.device)

        tokens = self.tokenizer(text=phrases,text_pair=texts,padding='max_length',return_tensors='pt',max_length=self.maxlen,truncation=True)
        tokens.to(self.device)

        output = self.model(labels=labels, **tokens)

        return output.loss, output.logits


class TrainEval():

    def __init__(self,pclfile,categoryfile,learningrate=1e-5,modeltype='bert',bertmodeltype='bert',maxlen=256,devbatchsize=1000,weightdecay=0):

        if os.path.isdir('tensorboarddir/'):
            shutil.rmtree('tensorboarddir/')
        os.mkdir('tensorboarddir/')

        self.writer = SummaryWriter('tensorboarddir/')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print ('Starting pre-processing')
        self.preprocess = PreprocessingUtils(pclfile,categoryfile,testfile=None)
        self.preprocess.get_train_test_data()

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.learningrate = learningrate

        self.devbatchsize=devbatchsize

        self.model = BertModels(bertmodeltype=bertmodeltype,maxlen=maxlen)


        self.optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=learningrate,weight_decay=weightdecay)

        self.epochs = 1000000
        self.samplesize = 32

        self.evalstep = 50
        self.earlystopgap = 30
        self.maxdevf1 = float('-inf')

        self.checkpointfile = 'data/checkpoint/model.pt'

    def set_seed(self,seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def train_eval_bertmodels(self):

        earlystopcounter = 0

        postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
        negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        torch.cuda.empty_cache()

        self.set_seed(42)
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs):

            self.model.train()

            possample = postraindata.sample(n=self.samplesize // 8)
            negsample = negtraindata.sample(n=(self.samplesize // 8) * 7)

            sample = pd.concat([possample, negsample], ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)

            loss, _ = self.model(sample)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar('train_loss', loss.item(), epoch)

            if epoch % self.evalstep == 0: # run evaluation

                earlystopcounter += 1

                self.model.eval()

                with torch.no_grad():

                    preds = pd.DataFrame()
                    devlabels = self.preprocess.refinedlabelsdev[['lineid','label']]

                    devloss = 0

                    for j in range(0, len(self.preprocess.devdata),self.devbatchsize):

                        if j + self.devbatchsize > len(self.preprocess.devdata):
                            df = self.preprocess.devdata.iloc[j:len(self.preprocess.devdata)]
                        else:
                            df = self.preprocess.devdata.iloc[j:j + self.devbatchsize]

                        loss, logit = self.model(df)

                        devloss += loss.item()
                        p = torch.argmax(logit, dim=1)

                        df['preds'] = p.tolist()
                        preds = preds.append(df)

                    preds.drop(['label'],axis=1,inplace=True)


                    preds = preds.loc[preds.groupby(['lineid'])['preds'].idxmax()].reset_index(drop=True)

                    preds.set_index('lineid')
                    devlabels.set_index('lineid')

                    devlabels = devlabels.merge(preds,how='inner',on='lineid')
                    devlabels.set_index('lineid')

                    f1score = f1_score(devlabels['label'].tolist(), devlabels['preds'].tolist())

                    self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep))
                    self.writer.add_scalar('dev_f1', f1score, int(epoch / self.evalstep))

                    print('dev f1 and loss: ' + str(f1score) + ',' + str(devloss))

                    if f1score > self.maxdevf1:

                        self.maxdevf1 = f1score
                        torch.save(self.model.state_dict(), self.checkpointfile.replace('.pt', '_' + str(type(self.model)) + '_' + str(f1score) + '.pt'))
                        devlabels.to_csv('data/errors_' + str(type(self.model)) + '_' + str(f1score) + '.csv')

                        earlystopcounter = 0

                    if earlystopcounter > self.earlystopgap:
                        print('early stop at epoch:' + str(epoch))
                        break

                    self.model.train()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--maxlen', type=int, default=224)
    parser.add_argument('--devbat', type=int, default=500)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--bertmodeltype',type=str, default='distilbert')

    args = parser.parse_args()

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'


    traineval = TrainEval(pclfile,categoriesfile,bertmodeltype=args.bertmodeltype,learningrate=args.lr,maxlen=args.maxlen,devbatchsize=args.devbat,weightdecay=args.wd)
    traineval.train_eval_bertmodels()


if __name__ == "__main__":
    main()


