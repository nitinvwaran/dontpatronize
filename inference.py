import pandas as pd
import torch
import torch.nn as nn
import argparse, os

from preprocessingutils import PreprocessingUtils
from modules import LSTMAttention, BertModels,CNNBert
from tqdm import tqdm

class Inference():
    def __init__(self,bestmodelpath, pclfile,categoriesfile,testfile,modeltype='rnn',bertmodeltype='rawdistilbert',devbatchsize=250,rnntype='lstm',maxlenphrase=256,maxlentext=256,hiddensize=256,numlayers=2,forcnn=False,multilabel=0):


        if not os.path.isdir('data/inference/'):
            os.mkdir('data/inference/')

        if not os.path.isdir('data/proba/testproba/'):
            os.mkdir('data/proba/testproba/')


        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile

        self.preprocess = PreprocessingUtils(pclfile,categoriesfile,testfile)
        if multilabel == 0:
            self.preprocess.get_train_test_data(usetalkdown=False,testdata=True,forcnn=forcnn)
        else:
            self.preprocess.get_train_test_multilabel(testdata=True, forcnn=forcnn)

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.rnntype = rnntype
        self.devbatchsize = devbatchsize

        if modeltype == 'bert':
            self.model = BertModels(bertmodeltype=bertmodeltype, maxlen=maxlentext,multilabel=multilabel)
        elif modeltype == 'rnn':
            self.model = LSTMAttention(rnntype=rnntype,bertmodeltype=bertmodeltype,maxlentext=maxlentext,maxlenphrase=maxlenphrase,hiddensize=hiddensize,numlayers=numlayers,multilabel=multilabel)
        else:
            self.model = CNNBert(maxlen=maxlentext,bertmodeltype=bertmodeltype,multilabel=multilabel)

        checkpoint = torch.load(bestmodelpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.softmax = nn.Softmax()

        self.multilabel = multilabel
        self.threshold = 0.5
        self.sigm = nn.Sigmoid()

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

                if self.multilabel == 0:
                    logitsdf = pd.DataFrame(logit.tolist(), columns=['zerologit', 'onelogit']).reset_index(drop=True)
                    probdf = pd.DataFrame(self.softmax(logit).tolist(), columns=['zeroprob', 'oneprob']).reset_index(
                        drop=True)
                    p = torch.argmax(logit, dim=1)

                    df['preds'] = p.tolist()
                    df = pd.concat([df, logitsdf, probdf], axis=1, ignore_index=True)
                else:
                    p = (self.sigm(logit) > self.threshold).type(torch.uint8)
                    p = pd.DataFrame(p.tolist(),
                                     columns=['unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                                              'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                                              'poorermerrier_pred']).reset_index(drop=True)
                    df = pd.concat([df, p], axis=1, ignore_index=True)


                preds = preds.append(df, ignore_index=True)

            if self.multilabel == 0:
                preds.columns = ['lineid',  'text', 'preds', 'zerologit', 'onelogit',
                                 'zeroprob', 'oneprob']


                preds = preds.set_index('lineid')

                preds.to_csv(
                    'data/inference/inference_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',
                    sep='\t', index=True)
            else:
                preds.columns = ['lineid', 'text',
                                 'unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                                 'authorityvoice_pred', 'metaphor_pred', 'compassion_pred', 'poorermerrier_pred']

                preds.to_csv(
                    'data/inference/inference_multilabel_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',
                    sep='\t', index=False)

    def inference(self):

        self.model.eval()

        with torch.no_grad():

            preds = pd.DataFrame()

            for j in tqdm(range(0, len(self.preprocess.testdata), self.devbatchsize)):

                if j + self.devbatchsize > len(self.preprocess.testdata):
                    df = self.preprocess.testdata.iloc[j:len(self.preprocess.testdata)]
                else:
                    df = self.preprocess.testdata.iloc[j:j + self.devbatchsize]


                if self.modeltype == 'bert':
                    _, logit = self.model(df,test=True)
                else:
                    logit = self.model(df)

                if self.multilabel == 0:
                    logitsdf = pd.DataFrame(logit.tolist(), columns=['zerologit', 'onelogit']).reset_index(drop=True)
                    probdf = pd.DataFrame(self.softmax(logit).tolist(), columns=['zeroprob', 'oneprob']).reset_index(drop=True)
                    p = torch.argmax(logit, dim=1)

                    df['preds'] = p.tolist()
                    df = pd.concat([df, logitsdf, probdf], axis=1, ignore_index=True)

                else:
                    df.reset_index(drop=False, inplace=True)
                    p = (self.sigm(logit) > self.threshold).type(torch.uint8)
                    p = pd.DataFrame(p.tolist(),
                                     columns=['unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                                              'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                                              'poorermerrier_pred']).reset_index(drop=True)
                    df = pd.concat([df, p], axis=1, ignore_index=True)



                preds = preds.append(df, ignore_index=True)


            if self.multilabel == 0:
                preds.columns = ['lineid', 'category', 'text', 'phrase', 'preds', 'zerologit', 'onelogit',
                             'zeroprob', 'oneprob']
                preds.to_csv(
                    'data/proba/testproba/testproba_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',
                    sep='\t', index=False)
                preds = preds.loc[preds.groupby(['lineid'])['preds'].idxmax()].reset_index(drop=True)
                preds.set_index('lineid')

                preds.to_csv(
                    'data/inference/inference_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',
                    sep='\t', index=False)
            else:
                preds.columns = ['lineid', 'category', 'text', 'phrase',
                                 'unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                                 'authorityvoice_pred', 'metaphor_pred', 'compassion_pred', 'poorermerrier_pred']
                preds = preds.groupby(['lineid'])[
                    ['lineid', 'unbalanced_power_pred', 'shallowsolution_pred', 'presupposition_pred',
                     'authorityvoice_pred', 'metaphor_pred', 'compassion_pred',
                     'poorermerrier_pred']].max().reset_index(drop=True)

                preds.to_csv('data/inference/inference_multilabel_'  + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',
                    sep='\t', index=False)


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('--maxlentext', type=int, default=192)
    parser.add_argument('--maxlenphrase', type=int, default=64)
    parser.add_argument('--hiddensize', type=int, default=256)
    parser.add_argument('--numlayers', type=int, default=2)
    parser.add_argument('--multilabel', type=int, default=1)

    args = parser.parse_args()

    with open('bestmodel.txt','r') as i:
        for line in i.readlines():
            bestmodelpath = str(line).strip().split(',')[0]
            break

    if args.multilabel == 0:
        params = bestmodelpath.split('_')
        modeltype = params[1].strip()
        bertmodeltype = params[2].strip()
        rnntype = params[3].strip()
    else:
        params = bestmodelpath.split('_')
        modeltype = params[2].strip()
        bertmodeltype = params[3].strip()
        rnntype = params[4].strip()


    print ('Model Type:' + modeltype)
    print ('Bert Model Type:' + bertmodeltype)
    print ('RNN Type:' + rnntype)
    print ('Multilabel:' + str(bool(args.multilabel)))

    if modeltype == 'cnn':
        forcnn = True
    else:
        forcnn = False


    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'
    testfile = 'data/dontpatronizeme_v1.4/testdata.tsv'

    inf = Inference(bestmodelpath=bestmodelpath,pclfile=pclfile,categoriesfile=categoriesfile,testfile=testfile,maxlentext=args.maxlentext,maxlenphrase=args.maxlenphrase,devbatchsize=500,modeltype=modeltype,bertmodeltype=bertmodeltype,rnntype=rnntype,hiddensize=args.hiddensize,numlayers=args.numlayers,forcnn=forcnn,multilabel=args.multilabel)

    if args.multilabel == 1:
        print ('Inference for multilabel')
    else:
        print ('Inference for single label')

    if modeltype != 'cnn':
        inf.inference()
    else:
        print ('inference for cnn')
        inf.inference_cnn()


if __name__ == "__main__":
    main()

