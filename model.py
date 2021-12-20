import torch
import torch.nn as nn
import os,shutil

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, auc, roc_curve
from transformers import RobertaTokenizer, RobertaModel
from preprocessing import PreProcessing


class RobertaWrapper():
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()
        self.bertmodel.model.to(self.device)

        self.maxlen = 67
        self.linear = nn.Linear(768,1)
        self.linear.to(self.device)



    def forward(self,dataframe):

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

        input_mask_expanded = attentionmask.unsqueeze(-1).expand(lasthiddenstate.size()).float()
        lasthiddenstate[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        maxvectors = torch.max(lasthiddenstate, 1)[0]

        i = 0
        for l in lengths:
            if i == 0:
                try:
                    squeezedvectors = torch.max(maxvectors[i:i + l,:],0)[0]
                    squeezedvectors = torch.unsqueeze(squeezedvectors,0)
                except Exception as ex:
                    print (ex)
                    print(dataframe)
            else:
                temp = torch.max(maxvectors[i:i + l, :], 0)[0]
                temp = torch.unsqueeze(temp, 0)
                squeezedvectors = torch.cat((squeezedvectors,temp),0)

            i += l

        assert squeezedvectors.size(dim=0) == len(dataframe)
        logits = self.linear(squeezedvectors)

        return logits


class TrainEval():
    def __init__(self,pclfile,categoryfile):

        if os.path.isdir('tensorboarddir/'):
            shutil.rmtree('tensorboarddir/')
        os.mkdir('tensorboarddir/')

        self.writer = SummaryWriter('tensorboarddir/')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.preprocess = PreProcessing(pclfile,categoryfile)
        self.preprocess.preprocess_data()
        self.model = Model()

        self.loss = nn.BCEWithLogitsLoss()
        self.loss.to(self.device)

        self.sigm = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.epochs = 1000
        self.samplesize = 32
        self.cutoff = 0.5

        self.evalstep = 20

    def train_eval(self):

        self.model.train()
        for epoch in range(1,self.epochs):
            sample = self.preprocess.traindata.sample(n=self.samplesize)
            logits = self.model(sample)

            labels = sample['label'].tolist()
            labels = torch.unsqueeze(torch.FloatTensor(labels),1)
            labels = labels.to(self.device)

            loss = self.loss(logits,labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            probs = self.sigm(logits)
            #preds = torch.squeeze((probs > self.cutoff).float()).tolist()
            labels = torch.squeeze(labels).tolist()
            probs = torch.squeeze(probs).tolist()

            #f1score = f1_score(labels,preds)
            fpr,tpr,_ = roc_curve(labels,probs,pos_label=1)
            #aucscore = auc(fpr,tpr)


            self.writer.add_scalar('train_loss',loss.item(),epoch)
            #self.writer.add_scalar('train_f1',f1score,i)
            #self.writer.add_scalar('train_auc',aucscore,i)

            if epoch % self.evalstep == 0: # run evaluation
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

                        devloss += self.loss(logit,label)
                        prob = self.sigm(logit)

                        probs.append(prob.item())
                        if prob > self.cutoff: preds.append(1)
                        else: preds.append(0)

                devloss /= len(self.preprocess.testdata)
                f1score = f1_score(labels,preds)
                fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
                aucscore = auc(fpr, tpr)

                self.writer.add_scalar('dev_loss', devloss.item(), int(epoch / self.evalstep ))
                self.writer.add_scalar('dev_f1', f1score, int(epoch / self.evalstep))
                self.writer.add_scalar('dev_auc', aucscore, int(epoch / self.evalstep))

                print ('dev f1 score:' + str(f1score))
                print ('dev loss:' + str(devloss.item()))
                print ('dev auc:' + str(aucscore))
                self.model.train()



def main():
    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None

    traineval = TrainEval(pclfile,categoriesfile)
    traineval.train_eval()






if __name__ == "__main__":
    main()




