import pandas as pd
import torch
import torch.nn as nn
import os,shutil
import math

from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from copy import  deepcopy
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



class ModelRoberta(nn.Module):
    def __init__(self):
        super(ModelRoberta, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()
        self.bertmodel.model.to(self.device)

        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

        self.linear1.to(self.device)
        self.linear2.to(self.device)
        self.linear3.to(self.device)

        self.relu = nn.ReLU()

        self.maxlen = 136

    def forward(self,dataframe):

        data = []
        sentences = dataframe['splits'].tolist()

        # flatten sentences
        for sentence in sentences:
            data.append(str(sentence).replace('\t',' '))

        inp = self.bertmodel.tokenizer(data, max_length=self.maxlen, padding='max_length', truncation=True,
                                       add_special_tokens=True, return_tensors='pt')
        inp.to(self.device)
        #inp['output_hidden_states'] = True
        #attentionmask = inp['attention_mask']

        output = self.bertmodel.model(**inp)
        lasthiddenstate = output['last_hidden_state']
        lasthiddenstate.to(self.device)

        feats = lasthiddenstate[:,0]

        logits = self.relu(self.linear1(feats))
        logits = self.relu(self.linear2(logits))
        logits = self.linear3(logits)

        return logits




class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()
        self.bertmodel.model.to(self.device)


        self.maxlen = 67
        self.linear1 = nn.Linear(768,128)
        self.linear2 = nn.Linear(128,1)
        self.linear1.to(self.device)
        self.linear2.to(self.device)

        self.relu = nn.ReLU()

        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(d_model=768,nhead=8,batch_first=True)
        self.posencoder = PositionalEncoding(d_model=768).to(self.device)
        self.encoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, num_layers=4).to(self.device)

        self.pooling = 'cls' # or max

        self.deprelfeats = {'nmod:npmod': 0, 'obl:npmod': 0, 'det:predet': 0, 'acl': 0, 'acl:relcl': 0, 'advcl': 0,
                            'advmod': 0, 'advmod:emph': 0, 'advmod:lmod': 0, 'amod': 0, 'appos': 0, 'aux': 0,
                            'aux:pass': 0, 'case': 0, 'cc': 0, 'cc:preconj': 0, 'ccomp': 0, 'clf': 0, 'compound': 0,
                            'compound:lvc': 0, 'compound:prt': 0, 'compound:redup': 0, 'compound:svc': 0, 'conj': 0,
                            'cop': 0, 'csubj': 0, 'csubj:pass': 0, 'dep': 0, 'det': 0, 'det:numgov': 0, 'det:nummod': 0,
                            'det:poss': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'expl:impers': 0, 'expl:pass': 0,
                            'expl:pv': 0, 'fixed': 0, 'flat': 0, 'flat:foreign': 0, 'flat:name': 0, 'goeswith': 0,
                            'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nmod:poss': 0, 'nmod:tmod': 0, 'nsubj': 0,
                            'nsubj:pass': 0, 'nummod': 0, 'nummod:gov': 0, 'obj': 0, 'obl': 0, 'obl:agent': 0,
                            'obl:arg': 0, 'obl:lmod': 0, 'obl:tmod': 0, 'orphan': 0, 'parataxis': 0, 'punct': 0,
                            'reparandum': 0, 'root': 0, 'vocative': 0, 'xcomp': 0}


    def create_dep_vectors(self,sentences,deps):

        newdeps = []

        for i in range(0,len(sentences)):
            tokenized = self.bertmodel.tokenizer.tokenize(sentences[i])
            deptokens = deps[i].split()

            newdeptokens = []
            latesttoken = deptokens[0]
            depindex = 0
            for j in range(0,len(tokenized)):
                if j == 0:
                    newdeptokens.append(latesttoken)
                else:
                    if 'Ġ' in tokenized[j]:
                        depindex += 1
                        latesttoken = deptokens[depindex]
                    newdeptokens.append(latesttoken)

            assert len(tokenized) == len(newdeptokens)
            newdeps.append(newdeptokens)

        # now convert the BPE tokenized deps to one hot vectors
        onehotdeps = []

        for newdep in newdeps:
            depvector = []
            for dep in newdep:
                onehot = deepcopy(self.deprelfeats)
                onehot[dep] = 1
                onehot = OrderedDict(sorted(onehot.items()))
                onehot = list(onehot.values())

                depvector.append(torch.IntTensor(onehot))

            if len(newdep) >= self.maxlen:
                depvector = depvector[:self.maxlen]
            else:
                for k in range(len(newdep),self.maxlen):
                    onehot = deepcopy(self.deprelfeats)
                    depvector.append(torch.IntTensor(list(onehot.values())))

            onehotdeps.append(depvector)

        return onehotdeps



    def forward(self,dataframe):

        try:
            data = [] # holds flattened sentences
            #depdata = []

            sentences = dataframe['splits'].tolist()
            lengths = dataframe['lengths'].tolist()
            #deps = dataframe['deps'].tolist()

            for sent in sentences:
                if str(sent).strip() != '':
                    s = str(sent).split('\t')
                    data.extend(s)

            #for dep in deps:
            #    d = dep.split('\t')
            #    depdata.extend(d)

            """
            onehotdeps = self.create_dep_vectors(data,depdata)
            for i in range(0,len(onehotdeps)):
                if i == 0:
                    onehottensor = torch.unsqueeze(torch.stack(onehotdeps[i]),dim=0)
                else:
                    onehottensor = torch.cat((onehottensor,torch.unsqueeze(torch.stack(onehotdeps[i]),dim=0)),dim=0)

            onehottensor = onehottensor.to(self.device)
            """


            inp = self.bertmodel.tokenizer(data, max_length=self.maxlen, padding='max_length', truncation=True,add_special_tokens=True,return_tensors='pt')
            inp.to(self.device)
            #inp['output_hidden_states'] = True
            attentionmask = inp['attention_mask']

            output = self.bertmodel.model(**inp)
            feats = output['last_hidden_state']
            feats.to(self.device)
            #concatvectors = torch.cat((lasthiddenstate,onehottensor),dim=2)

            #src = self.posencoder(lasthiddenstate)
            #feats = self.encoder(src,src_key_padding_mask = attentionmask)

            if self.pooling == 'max':
                input_mask_expanded = attentionmask.unsqueeze(-1).expand(feats.size()).float()
                feats[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                resultvectors = torch.max(feats, 1)[0]
            elif self.pooling == 'avg':
                input_mask_expanded = attentionmask.unsqueeze(-1).expand(feats.size()).float()
                sum_embeddings = torch.sum(feats * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                resultvectors = sum_embeddings / sum_mask
            elif self.pooling == 'cls':
                resultvectors = feats[:,0]


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
        self.preprocess.load_preprocessed_data()
        print('Completed preprocessing')

        self.model = Model()
        #self.model = ModelRoberta()

        self.loss = nn.BCEWithLogitsLoss()
        self.loss.to(self.device)

        self.sigm = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.epochs = 1000000
        self.samplesize = 16
        self.cutoff = 0.5

        self.evalstep = 20

        self.checkpointfile = 'data/checkpoint/model_v2.pt'

    def train_eval(self,labeltype):

        torch.cuda.empty_cache()

        devf1 = float('-inf')
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


            if epoch % self.evalstep == 0: # run evaluation
                torch.cuda.empty_cache()
                self.model.eval()

                with torch.no_grad():
                    preds = []
                    probs = []
                    labels = self.preprocess.testdata[labeltype].tolist()

                    devloss = 0
                    for j in range(0,len(self.preprocess.testdata)):
                        df = self.preprocess.testdata.iloc[[j]]

                        logit = self.model(df)


                        label = df[labeltype].tolist()
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
                    devf1 = f1score
                    errors = pd.concat([pd.Series(lineids), pd.Series(splits), pd.Series(preds), pd.Series(labels)],
                                       axis=1, ignore_index=True)

                    errors.columns = ['lineid','splits','preds']
                    errors.columns.append(labeltype)
                    errors = errors.set_index('lineid')
                    errors = errors.loc[self.preprocess.devids]
                    errors.to_csv('data/errors.csv')

                    torch.save(self.model.state_dict(), self.checkpointfile)

                self.model.train()




def main():
    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None

    traineval = TrainEval(pclfile,categoriesfile)
    labeltypes = ['unbalanced_power','shallowsolution','presupposition','authorityvoice','metaphor','compassion','poorermerrier']
    traineval.train_eval(labeltypes[0])

if __name__ == "__main__":
    main()




