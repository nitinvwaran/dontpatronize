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
    def __init__(self,maxlen=512,bertmodeltype='rawdistilbert',multilabel=0):

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

        self.multilabel = multilabel

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

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
        self.pool2 = nn.MaxPool1d(self.maxlen // 2)
        self.pool2.to(self.device)

        self.conv3 = nn.Conv1d(768, 128, 5)
        self.conv3.to(self.device)
        self.pool3 = nn.MaxPool1d(self.maxlen // 4)
        self.pool3.to(self.device)

        self.conv4 = nn.Conv1d(768, 128, 3)
        self.conv4.to(self.device)
        self.pool4 = nn.MaxPool1d(self.maxlen // 4)
        self.pool4.to(self.device)

        self.convavg = nn.AvgPool1d(5)
        self.convavg.to(self.device)

        self.linear1 = nn.Linear(128, 32)

        if multilabel == 0:
            self.linear2 = nn.Linear(32, 2)
        else:
            self.linear2 = nn.Linear(32, 7)

        self.linear1.to(self.device)
        self.linear2.to(self.device)


    def forward(self,dataframe):

        sentences = dataframe['text'].tolist()

        inp = self.tokenizer(sentences, max_length=self.maxlen, padding='max_length', truncation=True,
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
        #feats = torch.cat((feats2,feats3,feats4),dim=2)
        feats = self.convavg(feats)
        feats = torch.squeeze(feats,dim=2)

        #feats = torch.reshape(feats,(feats.size(dim=0),-1))
        feats.to(self.device)

        feats = self.relu(self.linear1(feats))
        feats = self.dropout(feats)
        logits = self.linear2(feats)


        return logits


class LSTMAttention(nn.Module):
    def __init__(self,rnntype='lstm',bertmodeltype='rawdistilbert',maxlentext=128,maxlenphrase=64,hiddensize=256,numlayers=2,multilabel=0):

        super(LSTMAttention,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.multilabel = multilabel

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

        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.rnntype = rnntype

        self.maxlentext = maxlentext
        self.maxlenphrase = maxlenphrase

        self.rnndropout = 0.5

        print ('Using lstm type:' + str(self.rnntype))
        if self.rnntype == 'lstm':
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


        self.inputdropout = nn.Dropout(p=0.1)

        self.attn = nn.Linear(self.hiddensize * 2, self.hiddensize * 2)
        self.attn.to(self.device)
        self.concat_linear = nn.Linear(self.hiddensize * 4, self.hiddensize * 2)
        self.concat_linear.to(self.device)

        if multilabel == 0:
            self.logitlinear = nn.Linear(self.hiddensize * 2, 2)
        else:
            self.logitlinear = nn.Linear(self.hiddensize * 2, 7)

        self.logitlinear.to(self.device)
        self.dropout = nn.Dropout(p=0.5)

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

        lasthiddenstatetext = self.inputdropout(lasthiddenstatetext)
        lasthiddenstatephrase = self.inputdropout(lasthiddenstatephrase)

        lstmtext, _ = self.rnntext(lasthiddenstatetext)
        if self.rnntype == 'lstm':
            lstmphrase, (hn,cn) = self.rnnphrase(lasthiddenstatephrase)
        else:
            lstmphrase, hn = self.rnnphrase(lasthiddenstatephrase)

        hn = hn[-2:]
        hn = torch.transpose(hn, 0, 1)
        hn = torch.reshape(hn, (len(dataframe), -1))

        attnweights = self.attn(lstmtext)
        attnweights = torch.bmm(attnweights, hn.unsqueeze(2))
        attnweights = F.softmax(attnweights.squeeze(2), dim=1)
        context = torch.bmm(lstmtext.transpose(1, 2), attnweights.unsqueeze(2)).squeeze(2)

        attnhidden = self.relu(self.concat_linear(torch.cat((context, hn), dim=1)))
        attnhidden = self.dropout(attnhidden)

        logits = self.logitlinear(attnhidden)

        return logits


class BertModels(nn.Module):
    def __init__(self,bertmodeltype,maxlen=512,multilabel=0):
        super(BertModels,self).__init__()

        self.multilabel=multilabel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if bertmodeltype == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            if self.multilabel == 0:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-cased')
            else:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=7)

        elif bertmodeltype == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            if self.multilabel == 0:
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')
            else:
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased',num_labels=7)
        elif bertmodeltype == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
            if self.multilabel == 0:
                self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
            else:
                self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased',num_labels=7)
        elif bertmodeltype == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            if self.multilabel == 0:
                self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
            else:
                self.model = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=7)

        self.model.to(self.device)

        self.maxlen = maxlen


    def forward(self,dataframe,test=False):

        texts = dataframe['text'].tolist()
        phrases = dataframe['phrase'].tolist()

        tokens = self.tokenizer(text=phrases,text_pair=texts,padding='max_length',return_tensors='pt',max_length=self.maxlen,truncation=True)
        tokens.to(self.device)

        if test == False:
            if self.multilabel == 0:
                labels = dataframe['label'].tolist()
                labels = torch.LongTensor(labels)
                labels = labels.to(self.device)
                output = self.model(labels=labels, **tokens)
            else:
                output = self.model(**tokens)

        else:
            output = self.model(**tokens)


        return output.loss, output.logits


class TrainEval():

    def __init__(self,pclfile,categoryfile,learningrate=1e-5,modeltype='rnn',bertmodeltype='rawdistilbert',rnntype='lstm',maxlentext=256,maxlenphrase=64,devbatchsize=1000,weightdecay=0.01,bestmodelname='bestmodel.txt',hiddensize=256,numlayers=2,forcnn=False,multilabel=0):

        if os.path.isdir('tensorboarddir/'):
            shutil.rmtree('tensorboarddir/')
        os.mkdir('tensorboarddir/')


        if not os.path.isdir('data/checkpoint/'):
            os.mkdir('data/checkpoint/')

        if not os.path.isdir('data/errors/'):
            os.mkdir('data/errors/')

        if not os.path.isdir('data/proba/'):
            os.mkdir('data/proba/')

        self.writer = SummaryWriter('tensorboarddir/')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print ('Starting pre-processing')
        self.preprocess = PreprocessingUtils(pclfile,categoryfile,testfile=None)
        if multilabel == 0:
            self.preprocess.get_train_test_data(usetalkdown=False,forcnn=forcnn)
        else:
            self.preprocess.get_train_test_multilabel(forcnn=forcnn)

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.learningrate = learningrate
        self.rnntype = rnntype
        self.multilabel = multilabel

        self.devbatchsize=devbatchsize

        self.bestmodelname = bestmodelname

        if modeltype == 'bert':
            self.model = BertModels(bertmodeltype=bertmodeltype,maxlen=maxlentext,multilabel=multilabel)
        elif modeltype == 'rnn':
            self.model = LSTMAttention(rnntype=rnntype,bertmodeltype=bertmodeltype,maxlentext=maxlentext,maxlenphrase=maxlenphrase,hiddensize=hiddensize,numlayers=numlayers,multilabel=multilabel)
        else: # cnn
            self.model = CNNBert(bertmodeltype=bertmodeltype,maxlen=maxlentext,multilabel=multilabel)

        self.bestmodel = ''

        self.softmax = nn.Softmax()
        self.sigm = nn.Sigmoid()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learningrate,weight_decay=weightdecay)

        if self.multilabel == 0:
            if self.modeltype != 'bert':
                self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,10]))
                self.loss.to(self.device)
        else:
            self.loss = nn.BCEWithLogitsLoss()
            self.loss.to(self.device)

        self.epochs = 1000000
        self.samplesize = 32

        self.evalstep = 500
        self.earlystopgap = 20
        self.maxdevf1 = float('-inf')
        self.mindevloss = float('inf')

        self.checkpointfile = 'data/checkpoint/model.pt'

        self.threshold = 0.5

    def set_seed(self,seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def write_best_model(self,f1score):
        with open(self.bestmodelname,'w') as o:
            o.write(self.bestmodel + ',' + str(f1score))

    def train_eval_cnn_models(self,checkpointfile=None):

        earlystopcounter = 0

        if self.multilabel == 0:
            postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
            negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        else:
            postraindata = self.preprocess.traindata.loc[(self.preprocess.traindata['unbalanced_power'] == 1) | (
                    self.preprocess.traindata['shallowsolution'] == 1) | (self.preprocess.traindata[
                                                                              'presupposition'] == 1) | (
                                                                 self.preprocess.traindata[
                                                                     'authorityvoice'] == 1) | (
                                                                 self.preprocess.traindata['metaphor'] == 1) | (
                                                                 self.preprocess.traindata['compassion'] == 1) | (
                                                                 self.preprocess.traindata['poorermerrier'] == 1)]

            negtraindata = self.preprocess.traindata.loc[(self.preprocess.traindata['unbalanced_power'] == 0) & (
                    self.preprocess.traindata['shallowsolution'] == 0) & (self.preprocess.traindata[
                                                                              'presupposition'] == 0) & (
                                                                 self.preprocess.traindata[
                                                                     'authorityvoice'] == 0) & (
                                                                 self.preprocess.traindata['metaphor'] == 0) & (
                                                                 self.preprocess.traindata['compassion'] == 0) & (
                                                                 self.preprocess.traindata['poorermerrier'] == 0)]

        torch.cuda.empty_cache()

        if checkpointfile is not None:
            checkpoint = torch.load(checkpointfile)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.set_seed(42)
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs):

            self.model.train()

            possample = postraindata.sample(n=self.samplesize // 2)
            negsample = negtraindata.sample(n=self.samplesize // 2)

            sample = pd.concat([possample, negsample], ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)

            logits = self.model(sample)

            if self.multilabel == 0:
                labels = sample['label'].tolist()
                labels = torch.LongTensor(labels)
            else:
                labels = sample[['unbalanced_power', 'shallowsolution', 'presupposition', 'authorityvoice', 'metaphor',
                                 'compassion', 'poorermerrier']].values.tolist()
                labels = torch.FloatTensor(labels)

            labels = labels.to(self.device)

            loss = self.loss(logits,labels)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar('train_loss', loss.item(), epoch)

            if epoch % self.evalstep == 0: # run evaluation

                earlystopcounter += 1

                self.model.eval()

                with torch.no_grad():

                    preds = pd.DataFrame()

                    devloss = 0

                    for j in range(0, len(self.preprocess.devdata),self.devbatchsize):

                        if j + self.devbatchsize > len(self.preprocess.devdata):
                            df = self.preprocess.devdata.iloc[j:len(self.preprocess.devdata)]
                        else:
                            df = self.preprocess.devdata.iloc[j:j + self.devbatchsize]

                        df.reset_index(drop=True,inplace=True)

                        if self.multilabel == 0:
                            labels = df['label'].tolist()
                            labels = torch.LongTensor(labels)
                        else:
                            labels = df[
                                ['unbalanced_power', 'shallowsolution', 'presupposition', 'authorityvoice', 'metaphor',
                                 'compassion', 'poorermerrier']].values.tolist()
                            labels = torch.FloatTensor(labels)

                        labels = labels.to(self.device)

                        logit = self.model(df)

                        loss = self.loss(logit,labels)

                        devloss += loss.item()


                        if self.multilabel == 0:
                            logitsdf = pd.DataFrame(logit.tolist(),columns=['zerologit','onelogit']).reset_index(drop=True)
                            probdf = pd.DataFrame(self.softmax(logit).tolist(),columns=['zeroprob','oneprob']).reset_index(drop=True)

                            p = torch.argmax(logit, dim=1)
                            df['preds'] = p.tolist()
                            df = pd.concat([df,logitsdf,probdf],axis=1,ignore_index=True)
                        else:
                            p = (self.sigm(logit) > self.threshold).type(torch.uint8)
                            p = pd.DataFrame(p.tolist(), columns=['unbalanced_power_pred', 'shallowsolution_pred',
                                                                  'presupposition_pred', 'authorityvoice_pred',
                                                                  'metaphor_pred', 'compassion_pred',
                                                                  'poorermerrier_pred']).reset_index(drop=True)
                            df = pd.concat([df, p], axis=1, ignore_index=True)
                            preds = preds.append(df, ignore_index=True)

                        preds = preds.append(df,ignore_index=True)

                    if self.multilabel == 0:
                        preds.columns = ['lineid','text','label','preds','zerologit','onelogit','zeroprob','oneprob']
                    else:
                        preds.columns = ['lineid', 'text', "unbalanced_power", "shallowsolution",
                                         "presupposition", "authorityvoice", "metaphor", "compassion", "poorermerrier",
                                         'unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                                         'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                                         'poorermerrier_pred']

                    preds = preds.set_index('lineid')

                    preds = preds.loc[self.preprocess.devids]


                    if self.multilabel == 0:
                        f1score = f1_score(preds['label'].tolist(), preds['preds'].tolist())

                        self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1', f1score, int(epoch / self.evalstep))

                        print('dev f1 and loss: ' + str(f1score) + ',' + str(devloss))

                        if f1score > self.maxdevf1:

                            self.maxdevf1 = f1score
                            self.bestmodel = self.checkpointfile.replace('.pt', '_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' + str(f1score) +  '.pt')

                            torch.save({'epoch':epoch,'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict()}, self.bestmodel)
                            preds.to_csv('data/errors/errors_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(f1score) + '.csv')

                            earlystopcounter = 0

                            self.write_best_model(f1score)

                        if earlystopcounter > self.earlystopgap:
                            print('early stop at epoch:' + str(epoch))
                            break

                    else:

                        f1score = f1_score(preds[['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor',"compassion","poorermerrier"]].values.tolist(),preds[['unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                             'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                             'poorermerrier_pred']].values.tolist(),average=None)

                        self.writer.add_scalar('dev_f1_avg', np.mean(f1score), int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_unbalanced', f1score[0], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_shallowsln', f1score[1], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_presupp', f1score[2], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_authority', f1score[3], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_metaphor', f1score[4], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_compassion', f1score[5], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_poorermerrier', f1score[6], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep))


                        print('f1 score and dev loss: ' + str(np.mean(f1score)) + ',' + str(devloss))

                        if np.mean(f1score) > self.maxdevf1:

                            self.maxdevf1 = np.mean(f1score)
                            self.bestmodel = self.checkpointfile.replace('.pt','_multilabel_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' + str(np.mean(f1score)) + '.pt')

                            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict()}, self.bestmodel)

                            preds.to_csv('data/errors/multilabel_preds_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(np.mean(f1score)) + '.csv')

                            earlystopcounter = 0

                            self.write_best_model(np.mean(f1score))

                            with open ('data/errors/bestscoresmulti_' +  self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(np.mean(f1score)),'w'  ) as best:
                                best.write('Unbalanced,' + str(f1score[0]) + '\n')
                                best.write('shallowsln,' + str(f1score[1]) + '\n')
                                best.write('Presupp,' + str(f1score[2]) + '\n')
                                best.write('Authority,' + str(f1score[3]) + '\n')
                                best.write('Metaphor,' + str(f1score[4]) + '\n')
                                best.write('Compassion,' + str(f1score[5]) + '\n')
                                best.write('PoorerMerrier,' + str(f1score[6]) + '\n')

                            if earlystopcounter > self.earlystopgap:
                                print('early stop at epoch:' + str(epoch))
                                break

    def train_eval_models(self,checkpointfile=None):

        earlystopcounter = 0

        if self.multilabel == 0:
            postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
            negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        else:
            postraindata = self.preprocess.traindata.loc[(self.preprocess.traindata['unbalanced_power'] == 1) | (
                        self.preprocess.traindata['shallowsolution'] == 1) | (self.preprocess.traindata[
                                                                                  'presupposition'] == 1) | (
                                                                     self.preprocess.traindata[
                                                                         'authorityvoice'] == 1) | (
                                                                     self.preprocess.traindata['metaphor'] == 1) | (
                                                                     self.preprocess.traindata['compassion'] == 1) | (
                                                                     self.preprocess.traindata['poorermerrier'] == 1)]

            negtraindata = self.preprocess.traindata.loc[(self.preprocess.traindata['unbalanced_power'] == 0) & (
                        self.preprocess.traindata['shallowsolution'] == 0) & (self.preprocess.traindata[
                                                                                  'presupposition'] == 0) & (
                                                                     self.preprocess.traindata[
                                                                         'authorityvoice'] == 0) & (
                                                                     self.preprocess.traindata['metaphor'] == 0) & (
                                                                     self.preprocess.traindata['compassion'] == 0) & (
                                                                     self.preprocess.traindata['poorermerrier'] == 0)]


        torch.cuda.empty_cache()

        if checkpointfile is not None:
            checkpoint = torch.load(checkpointfile)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.set_seed(42)
        self.optimizer.zero_grad()

        for epoch in range(1, self.epochs):

            self.model.train()

            possample = postraindata.sample(n=self.samplesize // 2)
            negsample = negtraindata.sample(n=self.samplesize // 2)

            sample = pd.concat([possample, negsample], ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)

            if self.modeltype == 'bert' and self.multilabel == 0:
                loss, _ = self.model(sample)
            elif self.modeltype == 'bert' and self.multilabel == 1:
                _, logits = self.model(sample)

                labels = sample[
                    ['unbalanced_power', 'shallowsolution', 'presupposition', 'authorityvoice', 'metaphor',
                     'compassion', 'poorermerrier']].values.tolist()
                labels = torch.FloatTensor(labels)

                labels = labels.to(self.device)

                loss = self.loss(logits, labels)

            else:
                logits = self.model(sample)

                if self.multilabel == 0:
                    labels = sample['label'].tolist()
                    labels = torch.LongTensor(labels)
                else:
                    labels = sample[['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']].values.tolist()
                    labels = torch.FloatTensor(labels)

                labels = labels.to(self.device)

                loss = self.loss(logits,labels)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar('train_loss', loss.item(), epoch)

            if epoch % self.evalstep == 0: # run evaluation

                earlystopcounter += 1

                self.model.eval()

                with torch.no_grad():

                    preds = pd.DataFrame()

                    if self.multilabel == 0:
                        devlabels = self.preprocess.refinedlabelsdev[['lineid','label']]
                    else:
                        devlabels = self.preprocess.refinedlabelsdev[['lineid','unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']]

                    devloss = 0


                    for j in range(0, len(self.preprocess.devdata),self.devbatchsize):

                        if j + self.devbatchsize > len(self.preprocess.devdata):
                            df = self.preprocess.devdata.iloc[j:len(self.preprocess.devdata)]
                        else:
                            df = self.preprocess.devdata.iloc[j:j + self.devbatchsize]

                        df.reset_index(drop=True,inplace=True)

                        if self.modeltype == 'bert' and self.multilabel == 0:
                            loss, logit = self.model(df)
                        elif self.modeltype == 'bert' and self.multilabel == 1:
                            _, logit = self.model(df)

                            labels = df[
                                ['unbalanced_power', 'shallowsolution', 'presupposition', 'authorityvoice', 'metaphor',
                                 'compassion', 'poorermerrier']].values.tolist()
                            labels = torch.FloatTensor(labels)

                            labels = labels.to(self.device)

                            loss = self.loss(logit, labels)

                        else:
                            if self.multilabel == 0:
                                labels = df['label'].tolist()
                                labels = torch.LongTensor(labels)
                            else:
                                labels = df[['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']].values.tolist()
                                labels = torch.FloatTensor(labels)

                            labels = labels.to(self.device)

                            logit = self.model(df)

                            loss = self.loss(logit,labels)

                        if self.multilabel == 0:
                            logitsdf = pd.DataFrame(logit.tolist(),columns=['zerologit','onelogit']).reset_index(drop=True)
                            probdf = pd.DataFrame(self.softmax(logit).tolist(),columns=['zeroprob','oneprob']).reset_index(drop=True)

                        devloss += loss.item()

                        if self.multilabel == 0:
                            p = torch.argmax(logit, dim=1)

                            df['preds'] = p.tolist()

                            df = pd.concat([df,logitsdf,probdf],axis=1,ignore_index=True)
                            preds = preds.append(df,ignore_index=True)

                        else:
                            p = (self.sigm(logit) > self.threshold).type(torch.uint8)
                            p = pd.DataFrame(p.tolist(),columns=['unbalanced_power_pred','shallowsolution_pred','presupposition_pred','authorityvoice_pred','metaphor_pred','compassion_pred','poorermerrier_pred']).reset_index(drop=True)
                            df = pd.concat([df,p], axis=1, ignore_index=True)
                            preds = preds.append(df, ignore_index=True)


                    if self.multilabel == 0:
                        preds.columns = ['lineid','category','text','phrase','label','preds','zerologit','onelogit','zeroprob','oneprob']
                    else:
                        preds.columns = ['lineid', 'category', 'text', 'phrase', "unbalanced_power","shallowsolution","presupposition","authorityvoice","metaphor","compassion","poorermerrier",
                                         'unbalanced_power_pred','shallowsolution_pred','presupposition_pred','authorityvoice_pred','metaphor_pred','compassion_pred','poorermerrier_pred']

                    if self.multilabel == 0:

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

                            torch.save({'epoch':epoch,'model_state_dict':self.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict()}, self.bestmodel)
                            devlabels.to_csv('data/errors/errors_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(f1score) + '.csv')
                            preds2.to_csv('data/proba/blendeddata_'  + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' + str(f1score) +  '.csv')

                            earlystopcounter = 0

                            self.write_best_model(f1score)

                        if earlystopcounter > self.earlystopgap:
                            print('early stop at epoch:' + str(epoch))
                            break

                    else:

                        preds = preds.groupby(['lineid'])[
                            ['lineid','unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                             'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                             'poorermerrier_pred']].max().reset_index(drop=True)

                        preds = preds.set_index('lineid')

                        devlabels = devlabels.set_index('lineid')
                        devlabels = devlabels.merge(preds,on='lineid',how='inner')
                        devlabels = devlabels.loc[self.preprocess.devids]

                        f1score = f1_score(devlabels[['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor',"compassion","poorermerrier"]].values.tolist(),devlabels[['unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                             'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                             'poorermerrier_pred']].values.tolist(),average=None)

                        self.writer.add_scalar('dev_f1_avg', np.mean(f1score), int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_unbalanced', f1score[0], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_shallowsln', f1score[1], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_presupp', f1score[2], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_authority', f1score[3], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_metaphor', f1score[4], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_compassion', f1score[5], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_f1_poorermerrier', f1score[6], int(epoch / self.evalstep))
                        self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep))


                        print('f1 score and dev loss: ' + str(np.mean(f1score)) + ',' + str(devloss))

                        if np.mean(f1score) > self.maxdevf1:

                            self.maxdevf1 = np.mean(f1score)
                            self.bestmodel = self.checkpointfile.replace('.pt','_multilabel_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' + str(np.mean(f1score)) + '.pt')

                            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict()}, self.bestmodel)

                            devlabels.to_csv('data/errors/multilabel_preds_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(np.mean(f1score)) + '.csv')

                            earlystopcounter = 0

                            self.write_best_model(np.mean(f1score))

                            with open ('data/errors/bestscoresmulti_' +  self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '_' +  str(np.mean(f1score)),'w'  ) as best:
                                best.write('Unbalanced,' + str(f1score[0]) + '\n')
                                best.write('shallowsln,' + str(f1score[1]) + '\n')
                                best.write('Presupp,' + str(f1score[2]) + '\n')
                                best.write('Authority,' + str(f1score[3]) + '\n')
                                best.write('Metaphor,' + str(f1score[4]) + '\n')
                                best.write('Compassion,' + str(f1score[5]) + '\n')
                                best.write('PoorerMerrier,' + str(f1score[6]) + '\n')

                            if earlystopcounter > self.earlystopgap:
                                print('early stop at epoch:' + str(epoch))
                                break



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-6, help="learning rate")
    parser.add_argument('--maxlentext', type=int, default=192, help = "Max. length of the context")
    parser.add_argument('--maxlenphrase', type=int, default=32,help="Max. length of the phrase")
    parser.add_argument('--devbat', type=int, default=250,help="Pick the batch size to process dev data")
    parser.add_argument('--wd', type=float, default=0.01,help="weight decay parameter for AdamW")
    parser.add_argument('--bertmodeltype',type=str, default='distilbert',help="type of pre-trained model to use")
    parser.add_argument('--modeltype', type=str, default='bert',help="one of bert,rnn,cnn")
    parser.add_argument('--rnntype', type=str, default='gru',help="one of lstm,gru")
    parser.add_argument('--bestmodelname', type=str, default='bestmodel.txt')
    parser.add_argument('--hiddensize', type=int, default=256,help="hidden size param for RNN")
    parser.add_argument('--numlayers', type=int, default=2,help="number of layers param for RNN")
    parser.add_argument('--multilabel', type=int, default=0,help="0 for Task 1,  1 for Task 2 (multi-label)")
    parser.add_argument('--chkpoint', type=str, default='',help="path to checkpoint file")


    args = parser.parse_args()

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'

    if args.modeltype == 'cnn':
        forcnn = True
    else:
        forcnn = False

    traineval = TrainEval(pclfile=pclfile,categoryfile=categoriesfile,modeltype=args.modeltype, bertmodeltype=args.bertmodeltype,rnntype=args.rnntype,learningrate=args.lr,maxlentext=args.maxlentext,maxlenphrase=args.maxlenphrase,devbatchsize=args.devbat,weightdecay=args.wd,bestmodelname=args.bestmodelname,hiddensize=args.hiddensize,numlayers=args.numlayers,forcnn=forcnn,multilabel=args.multilabel)

    if args.multilabel == 0:
        print('Training single classification')
    else:
        print('Training multi-label classification')

    if args.chkpoint != '':
        chkpoint = args.chkpoint
    else:
        chkpoint = None

    print ('checkpointfile is:' + str(chkpoint))
    if args.modeltype != 'cnn':
        traineval.train_eval_models(checkpointfile=chkpoint)
    else:
        print ('training for cnn')
        traineval.train_eval_cnn_models(checkpointfile=chkpoint)


if __name__ == "__main__":
    main()



