import pandas as pd
import torch
import torch.nn as nn
import os,shutil
import math
import numpy as np
import re
import torch.nn.functional as F
import random

from preprocessingutils import PreprocessingUtils
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from nltk.corpus import stopwords


class


class TrainEval():

    def __init__(self,pclfile,categoryfile,learningrate=1e-5,modeltype='bert',bertmodeltype='roberta'):

        if os.path.isdir('tensorboarddir/'):
            shutil.rmtree('tensorboarddir/')
        os.mkdir('tensorboarddir/')

        self.writer = SummaryWriter('tensorboarddir/')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print ('Starting pre-processing')
        self.preprocess = PreprocessingUtils(pclfile,categoryfile)
        self.preprocess.get_train_test_data()

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.learningrate = learningrate

        self.model = None

        if self.modeltype == 'bert':
            self.optimizer = torch.optim.AdamW(list(self.model.bertmodel.model.named_parameters()), lr=learningrate)

        self.epochs = 1000000
        self.samplesize = 32

        self.evalstep = 10
        self.earlystopgap = 25
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

                    for j in range(0, len(self.preprocess.devdata),1000):

                        if j + 1000 > len(self.preprocess.devdata):
                            df = self.preprocess.devdata.iloc[j:len(self.preprocess.devdata)]
                        else:
                            df = self.preprocess.devdata.iloc[j:j + 1000]

                        loss, logit = self.model(df)
                        devloss += loss.item()
                        p = torch.argmax(logit, dim=1)

                        df['preds'] = p.tolist()
                        preds = preds.append(df)

                    preds.to_csv('devpreds.csv', index=False)
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



