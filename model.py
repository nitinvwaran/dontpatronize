import pandas as pd
import torch
import torch.nn as nn
import os,shutil
import math

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, auc, roc_curve
from transformers import RobertaTokenizer, RobertaModel
from preprocessing import PreProcessing


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

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()
        self.bertmodel.model.to(self.device)

        self.maxlen = 67
        self.linear1 = nn.Linear(768,64)
        self.linear2 = nn.Linear(64,1)

        self.linear1.to(self.device)
        self.linear2.to(self.device)

        self.relu = nn.ReLU()

        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(d_model=768,nhead=8,batch_first=True)
        self.posencoder = PositionalEncoding(d_model=768).to(self.device)
        self.encoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, num_layers=4).to(self.device)

        self.pooling = 'max' # or max



    def forward(self,dataframe):

        try:
            data = [] # holds flattened sentences

            sentences = dataframe['splits'].tolist()
            lengths = dataframe['lengths'].tolist()

            for sent in sentences:
                s = sent.split('\t')
                data.extend(s)


            inp = self.bertmodel.tokenizer(data, max_length=self.maxlen, padding='max_length', truncation=True,add_special_tokens=False,return_tensors='pt')
            inp.to(self.device)
            inp['output_hidden_states'] = True
            attentionmask = inp['attention_mask']

            output = self.bertmodel.model(**inp)
            lasthiddenstate = output['last_hidden_state']

            src = self.posencoder(lasthiddenstate)
            feats = self.encoder(src)

            if self.pooling == 'max':
                input_mask_expanded = attentionmask.unsqueeze(-1).expand(feats.size()).float()
                lasthiddenstate[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                resultvectors = torch.max(lasthiddenstate, 1)[0]
            elif self.pooling == 'avg':
                input_mask_expanded = attentionmask.unsqueeze(-1).expand(feats.size()).float()
                sum_embeddings = torch.sum(feats * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                resultvectors = sum_embeddings / sum_mask


            i = 0
            for l in lengths:
                if i == 0:
                    squeezedvectors = torch.max(resultvectors[i:i + l,:],0)[0]
                    squeezedvectors = torch.unsqueeze(squeezedvectors,0)
                else:
                    temp = torch.max(resultvectors[i:i + l, :], 0)[0]
                    temp = torch.unsqueeze(temp, 0)
                    squeezedvectors = torch.cat((squeezedvectors,temp),0)

                i += l

            assert squeezedvectors.size(dim=0) == len(dataframe)

            logits = self.relu(self.linear1(squeezedvectors))
            logits = self.linear2(logits)


        except Exception as ex:
            print(ex)
            print(len(data))
            print(max(lengths))
            print(data)
            raise

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
        self.preprocess.preprocess_data()
        print('Completed preprocessing')

        self.model = Model()

        self.loss = nn.BCEWithLogitsLoss()
        self.loss.to(self.device)

        self.sigm = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.epochs = 1000000
        self.samplesize = 16
        self.cutoff = 0.5

        self.evalstep = 20

        self.checkpointfile = 'data/checkpoint/model_v1.pt'



    def train_eval(self):

        torch.cuda.empty_cache()

        devf1 = float('-inf')
        postraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 1]
        negtraindata = self.preprocess.traindata.loc[self.preprocess.traindata['label'] == 0]

        self.model.train()
        for epoch in range(1,self.epochs):


            possample = postraindata.sample(n=int(self.samplesize / 2))
            negsample = negtraindata.sample(n=int(self.samplesize / 2))

            sample = pd.concat([possample,negsample],ignore_index=True)
            sample = sample.sample(frac=1).reset_index(drop=True)
            self.model.zero_grad()

            logits = self.model(sample)

            labels = sample['label'].tolist()
            labels = torch.unsqueeze(torch.FloatTensor(labels),1)
            labels = labels.to(self.device)

            loss = self.loss(logits,labels)

            #self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            probs = self.sigm(logits)
            preds = torch.squeeze((probs > self.cutoff).float()).tolist()
            labels = torch.squeeze(labels).tolist()
            probs = torch.squeeze(probs).tolist()

            f1score = f1_score(labels,preds)
            fpr,tpr,_ = roc_curve(labels,probs,pos_label=1)
            aucscore = auc(fpr,tpr)


            self.writer.add_scalar('train_loss',loss.item(),epoch)
            self.writer.add_scalar('train_f1',f1score,epoch)
            self.writer.add_scalar('train_auc',aucscore,epoch)

            if epoch % 100 == 0:
                torch.save(self.model.state_dict(),self.checkpointfile)

            if epoch % self.evalstep == 0: # run evaluation
                torch.cuda.empty_cache()
                self.model.eval()

                with torch.no_grad():
                    preds = []
                    probs = []
                    labels = self.preprocess.testdata['label'].tolist()

                    devloss = 0
                    for j in range(0,len(self.preprocess.testdata)):
                        df = self.preprocess.testdata.iloc[[j]]
                        logit = self.model(df)

                        label = df['label'].tolist()
                        label = torch.unsqueeze(torch.FloatTensor(label),1)
                        label = label.to(self.device)

                        devloss += self.loss(logit,label).item()
                        prob = self.sigm(logit)

                        probs.append(prob.item())
                        if prob > self.cutoff: preds.append(1)
                        else: preds.append(0)

                devloss /= len(self.preprocess.testdata)
                f1score = f1_score(labels,preds)
                fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
                aucscore = auc(fpr, tpr)

                self.writer.add_scalar('dev_loss', devloss, int(epoch / self.evalstep ))
                self.writer.add_scalar('dev_f1', f1score, int(epoch / self.evalstep))
                self.writer.add_scalar('dev_auc', aucscore, int(epoch / self.evalstep))

                #print ('dev f1 score:' + str(f1score))
                #print ('dev loss:' + str(devloss.item()))
                #print ('dev auc:' + str(aucscore))

                lineids = self.preprocess.testdata['lineid'].tolist()
                splits = self.preprocess.testdata['splits'].tolist()

                if f1score > devf1:
                    errors = pd.concat([pd.Series(lineids), pd.Series(splits), pd.Series(preds), pd.Series(labels)],
                                       axis=1, ignore_index=True)
                    devf1 = f1score
                    errors.to_csv('data/errors.csv',index=False)

                self.model.train()

        torch.save(self.model.state_dict(), self.checkpointfile)


def main():
    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None

    traineval = TrainEval(pclfile,categoriesfile)
    traineval.train_eval()

if __name__ == "__main__":
    main()




