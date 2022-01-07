import pandas as pd
import torch
import torch.nn as nn
import os,shutil
import math
import numpy as np
import re
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from copy import  deepcopy
from sklearn.metrics import f1_score, auc, roc_curve
from transformers import RobertaTokenizer, RobertaModel
from preprocessing import PreProcessing
from nltk.corpus import stopwords


torch.backends.cudnn.deterministic = True


class RobertaWrapper():
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        #return self.dropout(x)
        return x

class ModelLSTMAttn():


    def __init__(self):
        super(ModelLSTMAttn,self).__init__()

        self.maxlen = 64
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()

        # for param in self.bertmodel.model.parameters():
        #    param.requires_grad = False

        self.bertmodel.model.to(self.device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.hiddensize = 256
        self.numlayers = 1

        self.rnn = nn.LSTM(input_size=768,hidden_size=self.hiddensize,num_layers=self.numlayers,batch_first=True,bidirectional=True)

        self.attn = nn.Linear(self.hiddensize * 2,self.hiddensize * 2)
        self.concat_linear = nn.Linear(self.hiddensize * 4, self.hiddensize * 2)

        self.logitlinear = nn.Linear(self.hiddensize * 2,2)




    def forward(self,dataframe):

        sentences = dataframe['splits'].tolist()

        inp = self.bertmodel.tokenizer(sentences, max_length=self.maxlen, padding='max_length', truncation=True,
                                       add_special_tokens=False, return_tensors='pt')
        inp.to(self.device)
        # inp['output_hidden_states'] = True
        # attentionmask = inp['attention_mask']

        output = self.bertmodel.model(**inp)
        lasthiddenstate = output['last_hidden_state']
        lasthiddenstate.to(self.device)

        lstmoutput,(hn,cn) = self.rnn(lasthiddenstate)
        finalhiddenstate = torch.add(hn[0,:,:],hn[1,:,:])

        attnweights = self.attn(lstmoutput)
        attnweights = torch.bmm(attnweights,finalhiddenstate.unsqueeze(2))

        attnweights = F.softmax(attnweights.squeeze(2),dim=1)

        context = torch.bmm(lstmoutput.transpose(1, 2), attnweights.unsqueeze(2)).squeeze(2)

        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, finalhiddenstate), dim=1)))

        logits = self.logitlinear(attnweights)

        return logits

"""
# best performing CNN v1
class ModelRobertaCNN(nn.Module):
    def __init__(self):
        super(ModelRobertaCNN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()

        #for param in self.bertmodel.model.parameters():
        #    param.requires_grad = False

        self.bertmodel.model.to(self.device)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout()
        self.stopwords = set(stopwords.words('english'))

        self.maxlen = 64
             
        
        
        self.conv3 = nn.Conv1d(768, 256, 5)
        self.conv3.to(self.device)
        self.pool3 = nn.MaxPool1d(self.maxlen // 2)
        self.pool3.to(self.device)

        self.conv4 = nn.Conv1d(768, 256, 3)
        self.conv4.to(self.device)
        self.pool4 = nn.MaxPool1d(self.maxlen // 4)
        self.pool4.to(self.device)

        self.conv5 = nn.Conv1d(768, 256, 2)
        self.conv5.to(self.device)
        self.pool5 = nn.MaxPool1d(self.maxlen // 4)
        self.pool5.to(self.device)

        self.convavg = nn.AvgPool1d(5)
        self.convavg.to(self.device)

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 2)
        self.linear1.to(self.device)
        self.linear2.to(self.device)



    def forward(self,dataframe):


        #data = []
        sentences = dataframe['splits'].tolist()

        #features = dataframe[['nmod:npmod','obl:npmod','det:predet','acl','acl:relcl','advcl','advmod','advmod:emph','advmod:lmod','amod','appos','aux','aux:pass','case','cc','cc:preconj','ccomp','clf','compound','compound:lvc','compound:prt','compound:redup','compound:svc','conj','cop','csubj','csubj:pass','dep','det','det:numgov','det:nummod','det:poss','discourse','dislocated','expl','expl:impers','expl:pass','expl:pv','fixed','flat','flat:foreign','flat:name','goeswith','iobj','list','mark','nmod','nmod:poss','nmod:tmod','nsubj','nsubj:pass','nummod','nummod:gov','obj','obl','obl:agent','obl:arg','obl:lmod','obl:tmod','orphan','parataxis','punct','reparandum','root','vocative','xcomp','subjectivity','positive','negative','subjectivewords','authoritytokens','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','COMMA','HYPH','.','``','TWOQUOT','$','-RRB-','-LRB-',':','NFP','ADD','AFX','ind','cnd','imp','pot','sub','jus','prp','qot','opt','des','nec','irr','adm','conv','fin','gdv','ger','inf','part','sup','vnoun','fut','past','pqp','pres','CC_category','CD_category','DT_category','EX_category','FW_category','IN_category','JJ_category','JJR_category','JJS_category','LS_category','MD_category','NN_category','NNS_category','NNP_category','NNPS_category','PDT_category','POS_category','PRP_category','PRP$_category','RB_category','RBR_category','RBS_category','RP_category','SYM_category','TO_category','UH_category','VB_category','VBD_category','VBG_category','VBN_category','VBP_category','VBZ_category','WDT_category','WP_category','WP$_category','WRB_category','COMMA_category','HYPH_category','._category','``_category','TWOQUOT_category','$_category','-RRB-_category','-LRB-_category',':_category','NFP_category','ADD_category','AFX_category']]
        #features = torch.FloatTensor(features.values)
        #features = features.to(self.device)


        # flatten sentences
        #for s in sentences:
        #    s = ' '.join([t for t in s.split(' ') if t not in self.stopwords])
        #    data.append(s)



        inp = self.bertmodel.tokenizer(sentences, max_length=self.maxlen, padding='max_length', truncation=True,
                                       add_special_tokens=False, return_tensors='pt')
        inp.to(self.device)
        # inp['output_hidden_states'] = True
        # attentionmask = inp['attention_mask']

        output = self.bertmodel.model(**inp)
        lasthiddenstate = output['last_hidden_state']
        lasthiddenstate.to(self.device)

        lasthiddenstate = lasthiddenstate.transpose(1,2)
        lasthiddenstate = self.dropout(lasthiddenstate)


        feats3 = self.conv3(lasthiddenstate)
        feats3 = self.pool3(feats3)

        feats4 = self.conv4(lasthiddenstate)
        feats4 = self.pool4(feats4)

        feats5 = self.conv5(lasthiddenstate)
        feats5 = self.pool5(feats5)

        #feats = torch.cat((feats1,feats2,feats3,feats4,feats5),dim=2)
        feats = torch.cat((feats3,feats4,feats5),dim=2)
        feats = self.convavg(feats)
        feats = torch.squeeze(feats,dim=2)

        #feats = torch.reshape(feats,(feats.size(dim=0),-1))
        feats.to(self.device)

        #feats = torch.cat((feats,features),dim=1)

        feats = self.relu(self.linear1(feats))
        feats = self.dropout(feats)
        logits = self.linear2(feats)


        return logits
"""


class ModelRobertaCNN(nn.Module):
    def __init__(self):
        super(ModelRobertaCNN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()

        #for param in self.bertmodel.model.parameters():
        #    param.requires_grad = False

        self.bertmodel.model.to(self.device)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout()

        self.maxlen = 64

        self.conv3 = nn.Conv1d(768, 256, 5)
        self.conv3.to(self.device)
        self.pool3 = nn.MaxPool1d(self.maxlen // 2)
        self.pool3.to(self.device)

        self.conv4 = nn.Conv1d(768, 256, 3)
        self.conv4.to(self.device)
        self.pool4 = nn.MaxPool1d(self.maxlen // 4)
        self.pool4.to(self.device)

        self.conv5 = nn.Conv1d(768, 256, 2)
        self.conv5.to(self.device)
        self.pool5 = nn.MaxPool1d(self.maxlen // 4)
        self.pool5.to(self.device)

        self.convavg = nn.AvgPool1d(5)
        self.convavg.to(self.device)

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 2)
        self.linear1.to(self.device)
        self.linear2.to(self.device)



    def forward(self,dataframe):


        #data = []
        sentences = dataframe['splits'].tolist()

        #features = dataframe[['nmod:npmod','obl:npmod','det:predet','acl','acl:relcl','advcl','advmod','advmod:emph','advmod:lmod','amod','appos','aux','aux:pass','case','cc','cc:preconj','ccomp','clf','compound','compound:lvc','compound:prt','compound:redup','compound:svc','conj','cop','csubj','csubj:pass','dep','det','det:numgov','det:nummod','det:poss','discourse','dislocated','expl','expl:impers','expl:pass','expl:pv','fixed','flat','flat:foreign','flat:name','goeswith','iobj','list','mark','nmod','nmod:poss','nmod:tmod','nsubj','nsubj:pass','nummod','nummod:gov','obj','obl','obl:agent','obl:arg','obl:lmod','obl:tmod','orphan','parataxis','punct','reparandum','root','vocative','xcomp','subjectivity','positive','negative','subjectivewords','authoritytokens','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','COMMA','HYPH','.','``','TWOQUOT','$','-RRB-','-LRB-',':','NFP','ADD','AFX','ind','cnd','imp','pot','sub','jus','prp','qot','opt','des','nec','irr','adm','conv','fin','gdv','ger','inf','part','sup','vnoun','fut','past','pqp','pres','CC_category','CD_category','DT_category','EX_category','FW_category','IN_category','JJ_category','JJR_category','JJS_category','LS_category','MD_category','NN_category','NNS_category','NNP_category','NNPS_category','PDT_category','POS_category','PRP_category','PRP$_category','RB_category','RBR_category','RBS_category','RP_category','SYM_category','TO_category','UH_category','VB_category','VBD_category','VBG_category','VBN_category','VBP_category','VBZ_category','WDT_category','WP_category','WP$_category','WRB_category','COMMA_category','HYPH_category','._category','``_category','TWOQUOT_category','$_category','-RRB-_category','-LRB-_category',':_category','NFP_category','ADD_category','AFX_category']]
        #features = torch.FloatTensor(features.values)
        #features = features.to(self.device)


        """
        # flatten sentences
        for s in sentences:
            s = s.replace('(','').replace(')','').replace('"','').replace('-',' ').replace(" ' ",' ')
            s = re.sub(r'(\. )+', ' . ', s)
            s = re.sub(r'\.\.+', '', s)
            s = re.sub(' +','',s)
            s = s.lower().strip()
            data.append(s)
        """



        inp = self.bertmodel.tokenizer(sentences, max_length=self.maxlen, padding='max_length', truncation=True,
                                       add_special_tokens=False, return_tensors='pt')
        inp.to(self.device)
        # inp['output_hidden_states'] = True
        # attentionmask = inp['attention_mask']

        output = self.bertmodel.model(**inp)
        lasthiddenstate = output['last_hidden_state']
        lasthiddenstate.to(self.device)

        lasthiddenstate = lasthiddenstate.transpose(1,2)
        lasthiddenstate = self.dropout(lasthiddenstate)

        feats3 = self.conv3(lasthiddenstate)
        feats3 = self.pool3(feats3)

        feats4 = self.conv4(lasthiddenstate)
        feats4 = self.pool4(feats4)

        feats5 = self.conv5(lasthiddenstate)
        feats5 = self.pool5(feats5)

        #feats = torch.cat((feats2,feats3,feats4,feats5),dim=2)
        feats = torch.cat((feats3,feats4,feats5),dim=2)
        feats = self.convavg(feats)
        feats = torch.squeeze(feats,dim=2)

        #feats = torch.reshape(feats,(feats.size(dim=0),-1))
        feats.to(self.device)

        #feats = torch.cat((feats,features),dim=1)

        feats = self.relu(self.linear1(feats))
        feats = self.dropout(feats)
        logits = self.linear2(feats)


        return logits


class ModelRoberta(nn.Module):
    def __init__(self):
        super(ModelRoberta, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()
        self.bertmodel.model.to(self.device)

        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.posencoder = PositionalEncoding(d_model=768).to(self.device)
        self.encoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, num_layers=8).to(self.device)

        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(128, 2)
        #self.linear3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout()
        self.dropoutinputs = nn.Dropout(p=0.1)
        self.dropout.to(self.device)
        self.dropoutinputs.to(self.device)

        self.linear1.to(self.device)
        self.linear2.to(self.device)

        self.relu = nn.ReLU()

        self.maxlen = 33
        #self.stopwords = set(stopwords.words('english'))

    def forward(self,dataframe):

        data = dataframe['splits'].tolist()

        """
        # flatten sentences
        for sentence in sentences:
            s = str(sentence).replace('\t', ' . ')
            #s = ' '.join([t for t in s.split(' ') if t not in self.stopwords])
            data.append(s)
        """

        inp = self.bertmodel.tokenizer(data, max_length=self.maxlen, padding='max_length', truncation=True,
                                       add_special_tokens=True, return_tensors='pt')
        inp.to(self.device)
        #inp['output_hidden_states'] = True
        attentionmask = inp['attention_mask']

        output = self.bertmodel.model(**inp)
        lasthiddenstate = output['last_hidden_state']

        src = self.posencoder(lasthiddenstate)
        feats = self.encoder(src,src_key_padding_mask = attentionmask)

        feats = self.dropoutinputs(feats)
        feats.to(self.device)

        feats = feats[:,0]

        logits = self.relu(self.linear1(feats))
        logits - self.dropout(logits)
        logits = self.linear2(logits)


        return logits

class TrainEval():
    def __init__(self,pclfile,categoryfile):

        if os.path.isdir('tensorboarddir/'):
            shutil.rmtree('tensorboarddir/')
        os.mkdir('tensorboarddir/')

        self.writer = SummaryWriter('tensorboarddir/')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print ('Starting pre-processing')
        self.preprocess = PreProcessing(pclfile,categoryfile)
        #self.preprocess.load_preprocessed_data()
        self.preprocess.preprocess_phrase_data()
        print('Completed preprocessing')

        #self.model = ModelRoberta()
        self.model = ModelRobertaCNN()


        self.lossmultilabel = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([15,52,52,44,58,23,288]))
        #self.lossmultilabel = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()

        self.loss.to(self.device)
        self.lossmultilabel.to(self.device)

        self.sigm = nn.Sigmoid()

        self.optimizer = torch.optim.AdamW(self.model.parameters(),weight_decay=0.01)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[4000],gamma=0.1)
        self.epochs = 1000000
        self.samplesize = 32
        self.cutoff = 0.5

        self.evalstep = 50
        self.earlystopgap = 100
        self.maxdevf1  = float('-inf')


        self.checkpointfile = 'data/checkpoint/model.pt'



    def train_eval_phrase(self):

        earlystopcounter = 0

        torch.cuda.empty_cache()

        postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
        negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        self.model.train()

        for epoch in range(1, self.epochs):

            #possample = postraindata.sample(n=self.samplesize // 4)
            #negsample = negtraindata.sample(n=(self.samplesize // 4) * 3)

            possample = postraindata.sample(n=self.samplesize // 2)
            negsample = negtraindata.sample(n=self.samplesize // 2)

            sample = pd.concat([possample, negsample], ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)

            self.optimizer.zero_grad()

            logits = self.model(sample)

            labels = sample['label'].tolist()
            labels = torch.LongTensor(labels)
            labels = labels.to(self.device)

            loss = self.loss(logits, labels)

            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            self.writer.add_scalar('train_loss', loss.item(), epoch)

            if epoch % self.evalstep == 0: # run evaluation

                earlystopcounter += 1

                torch.cuda.empty_cache()
                self.model.eval()


                with torch.no_grad():

                    preds = pd.DataFrame()
                    devlabels = self.preprocess.refineddevlabels[['lineid','label']]

                    devloss = 0

                    for j in range(0, len(self.preprocess.testdata),1000):

                        if j + 1000 > len(self.preprocess.testdata):
                            df = self.preprocess.testdata.iloc[j:len(self.preprocess.testdata)]
                        else:
                            df = self.preprocess.testdata.iloc[j:j + 1000]

                        logit = self.model(df)

                        l = df['label'].tolist()
                        l = torch.LongTensor(l)
                        l = l.to(self.device)

                        devloss += self.loss(logit, l).item()
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

                    if f1score > self.maxdevf1:

                        self.maxdevf1 = f1score
                        torch.save(self.model.state_dict(), self.checkpointfile.replace('.pt', '_' + str(type(self.model)) + '_' + str(f1score) + '.pt'))
                        devlabels.to_csv('data/errorsphrase_' + str(type(self.model)) + '_' + str(f1score) + '.csv')

                        earlystopcounter = 0

                    if earlystopcounter > self.earlystopgap:
                        print('early stop at epoch:' + str(epoch))
                        break

                    self.model.train()


    def train_eval(self,labeltype):

        torch.cuda.empty_cache()

        postraindata = self.preprocess.traindata.loc[self.preprocess.traindata[labeltype] == 1]
        negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata[labeltype] == 0]

        self.model.train()
        for epoch in range(1,self.epochs):

            possample = postraindata.sample(n=int(self.samplesize / 2))
            negsample = negtraindata.sample(n=int(self.samplesize / 2))

            sample = pd.concat([possample,negsample],ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)
            self.model.zero_grad()

            logits = self.model(sample)

            labels = sample[labeltype].tolist()
            labels = torch.LongTensor(labels)
            labels = labels.to(self.device)

            loss = self.loss(logits,labels)

            #self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(logits,dim=1).tolist()
            #preds = torch.squeeze((probs > self.cutoff).float()).tolist()
            labels = torch.squeeze(labels).tolist()
            f1score = f1_score(labels,preds)

            self.writer.add_scalar('train_loss',loss.item(),epoch)
            self.writer.add_scalar('train_f1',f1score,epoch)


            if epoch % self.evalstep == 0: # run evaluation
                torch.cuda.empty_cache()
                self.model.eval()
                self.model.batchlen = 1

                with torch.no_grad():
                    preds = []
                    labels = self.preprocess.testdata[labeltype].tolist()

                    devloss = 0
                    for j in range(0,len(self.preprocess.testdata)):
                        df = self.preprocess.testdata.iloc[[j]]

                        logit = self.model(df)

                        label = df[labeltype].tolist()
                        label = torch.LongTensor(label)
                        label = label.to(self.device)

                        devloss += self.loss(logit,label).item()
                        p = torch.argmax(logit,dim=1)
                        preds.append(p.item())

                devloss /= len(self.preprocess.testdata)
                f1score = f1_score(labels,preds)

                self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep ))
                self.writer.add_scalar('dev_f1', f1score, int(epoch / self.evalstep))

                #print ('dev f1 score:' + str(f1score))
                #print ('dev loss:' + str(devloss.item()))


                lineids = self.preprocess.testdata['lineid'].tolist()
                splits = self.preprocess.testdata['splits'].tolist()

                errors = pd.concat([pd.Series(lineids), pd.Series(splits), pd.Series(preds), pd.Series(labels)],
                                   axis=1, ignore_index=True)

                cols = ['lineid','splits','preds']
                cols.append(labeltype)
                errors.columns = cols

                errors = errors.set_index('lineid')
                errors = errors.loc[self.preprocess.devids]
                errors.to_csv('data/errors.csv')

                torch.save(self.model.state_dict(), self.checkpointfile.replace('.pt','_' + labeltype + '.pt'))

                self.model.batchlen = 16
                self.model.train()

    def train_eval_multilabel(self):

        torch.cuda.empty_cache()

        postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
        negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        self.model.train()
        for epoch in range(1,self.epochs):

            possample = postraindata.sample(n=int(self.samplesize / 2))
            negsample = negtraindata.sample(n=int(self.samplesize / 2))

            sample = pd.concat([possample,negsample],ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)
            #self.model.zero_grad()
            self.optimizer.zero_grad()

            logits = self.model(sample)

            mainlabel = sample['label'].tolist()
            labels = np.array(sample[['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']])
            labels = torch.FloatTensor(labels)
            labels = labels.to(self.device)

            loss = self.lossmultilabel(logits,labels)
            loss.backward()
            self.optimizer.step()

            preds = self.sigm(logits)
            preds = (preds > self.cutoff).type(torch.int)
            preds = torch.max(preds,dim=1)[0]
            preds = preds.tolist()
            f1score = f1_score(mainlabel,preds)

            self.writer.add_scalar('train_loss',loss.item(),epoch)
            self.writer.add_scalar('train_f1',f1score,epoch)


            if epoch % self.evalstep == 0: # run evaluation
                torch.cuda.empty_cache()
                self.model.eval()
                self.model.batchlen = 1

                with torch.no_grad():
                    preds = []
                    multilabelpreds = []
                    mainlabel = self.preprocess.testdata['label'].tolist()

                    devloss = 0
                    for j in range(0,len(self.preprocess.testdata)):
                        df = self.preprocess.testdata.iloc[[j]]

                        logit = self.model(df)

                        label = np.array(df[['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']])
                        label = torch.FloatTensor(label)
                        label = label.to(self.device)

                        devloss += self.lossmultilabel(logit,label).item()
                        p = self.sigm(logit)
                        p = (p > self.cutoff).type(torch.int)
                        multilabelpreds.append(p[0].tolist())
                        p = torch.max(p,dim=1)[0]
                        preds.append(p.item())

                devloss /= len(self.preprocess.testdata)
                f1score = f1_score(mainlabel,preds)

                self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep ))
                self.writer.add_scalar('dev_f1', f1score, int(epoch / self.evalstep))

                #print ('dev f1 score:' + str(f1score))
                #print ('dev loss:' + str(devloss.item()))


                lineids = self.preprocess.testdata['lineid'].tolist()
                splits = self.preprocess.testdata['splits'].tolist()


                errors = pd.concat([pd.Series(lineids), pd.Series(splits), pd.Series(preds), pd.Series(mainlabel)],
                                   axis=1, ignore_index=True)

                cols = ['lineid','splits','preds','label']
                errors.columns = cols

                mlabels = pd.DataFrame(data=multilabelpreds,columns=['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier'])
                errors = pd.concat([errors,mlabels],axis=1)

                errors = errors.set_index('lineid')
                errors = errors.loc[self.preprocess.devids]
                errors.to_csv('data/errors.csv')

                torch.save(self.model.state_dict(), self.checkpointfile.replace('.pt','_' + 'label' + '.pt'))

                self.model.batchlen = 16
                self.model.train()


def main():
    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None

    traineval = TrainEval(pclfile,categoriesfile)
    labeltypes = ['label','unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']
    #traineval.train_eval(labeltypes[0])
    #traineval.train_eval_multilabel()
    traineval.train_eval_phrase()

if __name__ == "__main__":
    main()




