import pandas as pd
import torch
import torch.nn as nn
import os,shutil
import numpy as np
import random
import argparse
import torch.nn.functional as F

from copy import deepcopy
from preprocessingutils import PreprocessingUtils
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from transformers import BertTokenizer,BertForSequenceClassification,DistilBertTokenizer,DistilBertForSequenceClassification,\
    DistilBertModel, BertModel, RobertaTokenizer,RobertaForSequenceClassification,XLNetTokenizer,XLNetForSequenceClassification, RobertaModel, XLNetModel


class CNNBert(nn.Module):
    def __init__(self,maxlen=512,bertmodeltype='rawdistilbert'):

        super(CNNBert, self).__init__()

        if bertmodeltype == 'rawdistilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-cased')
        elif bertmodeltype == 'rawbert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = BertModel.from_pretrained('bert-base-cased')
        elif bertmodeltype == 'rawroberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
        elif bertmodeltype == 'rawxlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = XLNetModel.from_pretrained('xlnet-base-cased')

        self.model.to(self.device)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout()
        self.dropoutinputs = nn.Dropout(p=0.1)

        self.maxlen = maxlen

        self.conv1 = nn.Conv1d(768, 128, 9)
        self.conv1.to(self.device)
        self.pool1 = nn.MaxPool1d(self.maxlen - 9)
        self.pool1.to(self.device)

        self.conv2 = nn.Conv1d(768, 128, 7)
        self.conv2.to(self.device)
        self.pool2 = nn.MaxPool1d(self.maxlen - 7)
        self.pool2.to(self.device)

        self.conv3 = nn.Conv1d(768, 128, 5)
        self.conv3.to(self.device)
        self.pool3 = nn.MaxPool1d(self.maxlen // 2)
        self.pool3.to(self.device)

        self.conv4 = nn.Conv1d(768, 128, 3)
        self.conv4.to(self.device)
        self.pool4 = nn.MaxPool1d(self.maxlen // 2)
        self.pool4.to(self.device)

        self.convavg = nn.AvgPool1d(5)
        self.convavg.to(self.device)

        self.linear1 = nn.Linear(128, 2)
        self.linear2 = nn.Linear(64, 2)
        self.linear1.to(self.device)
        self.linear2.to(self.device)


    def forward(self,dataframe):

        sentences = dataframe['phrase'].tolist()

        inp = self.bertmodel.tokenizer(sentences, max_length=self.maxlen, padding='max_length', truncation=True,
                                       add_special_tokens=False, return_tensors='pt')
        inp.to(self.device)
        output = self.model(**inp)
        lasthiddenstate = output.last_hidden_state
        lasthiddenstate.to(self.device)

        lasthiddenstate = lasthiddenstate.transpose(1,2)
        lasthiddenstate = self.dropoutinputs(lasthiddenstate)

        feats1 = self.conv1(lasthiddenstate)
        feats1 = self.pool1(feats1)

        feats2 = self.conv2(lasthiddenstate)
        feats2 = self.pool2(feats2)

        feats3 = self.conv3(lasthiddenstate)
        feats3 = self.pool3(feats3)

        feats4 = self.conv4(lasthiddenstate)
        feats4 = self.pool4(feats4)


        feats = torch.cat((feats1, feats2,feats3,feats4),dim=2)
        feats = self.convavg(feats)
        feats = torch.squeeze(feats,dim=2)

        #feats = torch.reshape(feats,(feats.size(dim=0),-1))
        feats.to(self.device)

        feats = self.relu(self.linear1(feats))
        feats = self.dropout(feats)
        logits = self.linear1(feats)


        return logits


class LSTMAttention(nn.Module):
    def __init__(self,rnntype='lstm',bertmodeltype='rawdistilbert',maxlentext=128,maxlenphrase=64):

        super(LSTMAttention,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if bertmodeltype == 'rawdistilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-cased')
        elif bertmodeltype == 'rawbert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = BertModel.from_pretrained('bert-base-cased')
        elif bertmodeltype == 'rawroberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
        elif bertmodeltype == 'rawxlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = XLNetModel.from_pretrained('xlnet-base-cased')

        self.model.to(self.device)

        self.hiddensize = 256
        self.numlayers = 2
        self.rnntype = rnntype

        self.maxlentext = maxlentext
        self.maxlenphrase = maxlenphrase

        self.rnndropout = 0.3

        if self.lstmtype == 'lstm':
            self.rnnphrase = nn.LSTM(input_size=768,dropout=self.rnndropout,hidden_size=self.hiddensize, num_layers=self.numlayers, batch_first=True,bidirectional=True)
            self.rnnphrase.to(self.device)

            self.rnntext = nn.LSTM(input_size=768, hidden_size=self.hiddensize, num_layers=self.numlayers, batch_first=True,bidirectional=True,dropout=self.rnndropout)
            self.rnntext.to(self.device)
        else: # gru
            self.rnnphrase = nn.GRU(input_size=768, dropout=self.rnndropout, hidden_size=self.hiddensize,
                                     num_layers=self.numlayers, batch_first=True, bidirectional=True)
            self.rnnphrase.to(self.device)

            self.rnntext = nn.GRU(input_size=768, hidden_size=self.hiddensize, num_layers=self.numlayers,
                                   batch_first=True, bidirectional=True, dropout=self.rnndropout)
            self.rnntext.to(self.device)



        self.attn = nn.Linear(self.hiddensize * 2, self.hiddensize * 2)
        self.attn.to(self.device)
        self.concat_linear = nn.Linear(self.hiddensize * 4, self.hiddensize * 2)
        self.concat_linear.to(self.device)

        self.logitlinear = nn.Linear(self.hiddensize * 2, 2)
        self.logitlinear.to(self.device)

        self.relu = nn.ReLU()

    def forward(self,dataframe):

        texts = dataframe['text'].tolist()
        phrases = dataframe['phrase'].tolist()

        tokenstext = self.tokenizer(texts, max_length=self.maxlentext, padding='max_length', truncation=True,
                                       add_special_tokens=False, return_tensors='pt')
        tokenstext.to(self.device)

        tokensphrases = self.tokenizer(phrases,max_length=self.maxlenphrase, padding='max_length', truncation=True,
                                       add_special_tokens=False, return_tensors='pt')
        tokensphrases.to(self.device)

        outputtext = self.model(**tokenstext)
        lasthiddenstatetext = outputtext['last_hidden_state']
        lasthiddenstatetext.to(self.device)

        outputphrase = self.model(**tokensphrases)
        lasthiddenstatephrase = outputphrase['last_hidden_state']
        lasthiddenstatephrase.to(self.device)

        lstmtext, _ = self.rnntext(lasthiddenstatetext)
        lstmphrase, (hn,cn) = self.rnnphrase(lasthiddenstatephrase)
        hn = hn[:-2]
        hn = torch.transpose(hn, 0, 1)
        hn = torch.reshape(hn, (len(dataframe), -1))

        attnweights = self.attn(lstmtext)
        attnweights = torch.bmm(attnweights, hn.unsqueeze(2))
        attnweights = F.softmax(attnweights.squeeze(2), dim=1)
        context = torch.bmm(lstmtext.transpose(1, 2), attnweights.unsqueeze(2)).squeeze(2)

        attnhidden = self.relu(self.concat_linear(torch.cat((context, hn), dim=1)))

        logits = self.logitlinear(attnhidden)

        return logits


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
        elif bertmodeltype == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            self.model = XLNetForSequenceClassification('xlnet-base-cased')
        elif bertmodeltype == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')

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

    def __init__(self,pclfile,categoryfile,learningrate=1e-5,modeltype='rnn',bertmodeltype='rawdistilbert',rnntype='lstm',maxlentext=256,maxlenphrase=64,devbatchsize=1000,weightdecay=0):

        if os.path.isdir('tensorboarddir/'):
            shutil.rmtree('tensorboarddir/')
        os.mkdir('tensorboarddir/')

        self.writer = SummaryWriter('tensorboarddir/')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print ('Starting pre-processing')
        self.preprocess = PreprocessingUtils(pclfile,categoryfile,testfile=None)
        self.preprocess.get_train_test_data(usetalkdown=False)

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.learningrate = learningrate
        self.rnntype = rnntype

        self.devbatchsize=devbatchsize

        if modeltype == 'bert':
            self.model = BertModels(bertmodeltype=bertmodeltype,maxlen=maxlentext)
        elif modeltype == 'rnn':
            self.model = LSTMAttention(rnntype=rnntype,bertmodeltype=bertmodeltype,maxlentext=maxlentext,maxlenphrase=maxlenphrase)
        else: # cnn
            self.model = CNNBert(bertmodeltype=bertmodeltype,maxlen=maxlentext)

        self.bestmodel = ''

        self.softmax = nn.Softmax()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learningrate,weight_decay=weightdecay)

        if self.modeltype != 'bert':
            self.loss = nn.CrossEntropyLoss()
            self.loss.to(self.device)


        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[1000,5000],gamma=0.1)

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

    def write_best_model(self,f1score):
        with open('bestmodel.txt','w') as o:
            o.write(self.bestmodel + ',' + str(f1score))

    def train_eval_models(self):

        earlystopcounter = 0

        postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
        negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        torch.cuda.empty_cache()

        self.set_seed(42)
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs):

            self.model.train()

            possample = postraindata.sample(n=1)
            negsample = negtraindata.sample(n=self.samplesize - 1)

            sample = pd.concat([possample, negsample], ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)

            if self.modeltype == 'bert':
                loss, _ = self.model(sample)

            else:
                logits = self.model(sample)
                labels = sample['label'].tolist()
                labels = torch.LongTensor(labels)
                labels = labels.to(self.device)

                loss = self.loss(logits,labels)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.scheduler.step()

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

                        df.reset_index(drop=True,inplace=True)

                        if self.modeltype == 'bert':
                            loss, logit = self.model(df)
                        else:
                            labels = df['label'].tolist()
                            labels = torch.LongTensor(labels)
                            labels = labels.to(self.device)

                            logit = self.model(df)

                            loss = self.loss(logit,labels)

                        logitsdf = pd.DataFrame(logit.tolist(),columns=['zerologit','onelogit']).reset_index(drop=True)
                        probdf = pd.DataFrame(self.softmax(logit).tolist(),columns=['zeroprob','oneprob']).reset_index(drop=True)

                        devloss += loss.item()
                        p = torch.argmax(logit, dim=1)

                        df['preds'] = p.tolist()
                        df = pd.concat([df,logitsdf,probdf],axis=1,ignore_index=True)

                        preds = preds.append(df,ignore_index=True)

                    preds.columns = ['lineid','category','text','phrase','label','preds','zerologit','onelogit','zeroprob','oneprob']
                    preds2 = deepcopy(preds)

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
                        self.bestmodel = self.checkpointfile.replace('.pt', '_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' + str(f1score) +  '.pt')

                        torch.save(self.model.state_dict(), self.bestmodel)
                        devlabels.to_csv('data/errors/errors_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(f1score) + '.csv')
                        preds2.to_csv('data/proba/blendeddata_'  + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' + str(f1score) +  '.csv')

                        earlystopcounter = 0

                        self.write_best_model(f1score)

                    if earlystopcounter > self.earlystopgap:
                        print('early stop at epoch:' + str(epoch))
                        break

                    self.model.train()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--maxlentext', type=int, default=224)
    parser.add_argument('--maxlenphrase', type=int, default=96)
    parser.add_argument('--devbat', type=int, default=500)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--bertmodeltype',type=str, default='bert')
    parser.add_argument('--modeltype', type=str, default='rnn')
    parser.add_argument('--rnntype', type=str, default='lstm')

    args = parser.parse_args()

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'


    traineval = TrainEval(pclfile=pclfile,categoryfile=categoriesfile,modeltype=args.modeltype, bertmodeltype=args.bertmodeltype,rnntype=args.rnntype,learningrate=args.lr,maxlentext=args.maxlentext,maxlenphrase=args.maxlenphrase,devbatchsize=args.devbat,weightdecay=args.wd)
    traineval.train_eval_models()


if __name__ == "__main__":
    main()



