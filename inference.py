import pandas as pd
import torch
import argparse

from preprocessingutils import PreprocessingUtils
from modules import LSTMAttention, BertModels,CNNBert
from tqdm import tqdm

class Inference():
    def __init__(self,bestmodelpath, pclfile,categoriesfile,testfile,modeltype='rnn',bertmodeltype='rawdistilbert',devbatchsize=1000,rnntype='lstm',maxlenphrase=96,maxlentext=512):

        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile

        self.preprocess = PreprocessingUtils(pclfile,categoriesfile,testfile)
        self.preprocess.get_train_test_data(usetalkdown=False,testdata=True)

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.rnntype = rnntype
        self.devbatchsize = devbatchsize

        if modeltype == 'bert':
            self.model = BertModels(bertmodeltype=bertmodeltype, maxlen=maxlentext)
        elif modeltype == 'rnn':
            self.model = LSTMAttention(rnntype=rnntype,bertmodeltype=bertmodeltype,maxlentext=maxlentext,maxlenphrase=maxlenphrase)
        else:
            self.model = CNNBert(maxlen=maxlentext,bertmodeltype=bertmodeltype)

        checkpoint = torch.load(bestmodelpath)
        self.model.load_state_dict(checkpoint)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
                    _, logit = self.model(df)
                else:
                    logit = self.model(df)

                p = torch.argmax(logit, dim=1)

                df['preds'] = p.tolist()

                preds = preds.append(df, ignore_index=True)

            preds = preds.loc[preds.groupby(['lineid'])['preds'].idxmax()].reset_index(drop=True)
            preds.set_index('lineid')

            preds.to_csv('inference_' + self.modeltype + '_' + self.bertmodeltype + '_' + self.rnntype + '.tsv',sep='\t')


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('--maxlentext', type=int, default=224)
    parser.add_argument('--maxlenphrase', type=int, default=96)

    args = parser.parse_args()

    with open('bestmodel.txt','r') as i:
        for line in i.readlines():
            bestmodelpath = str(line).strip().split()[0]

    params = bestmodelpath.split('_')
    modeltype = params[1].strip()
    bertmodeltype = params[2].strip()
    rnntype = params[3].strip()


    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'

    inf = Inference(bestmodelpath=bestmodelpath,pclfile=pclfile,categoriesfile=categoriesfile,testfile=None,maxlentext=args.maxlentext,maxlenphrase=args.maxlenphrase,devbatchsize=500,modeltype=modeltype,bertmodeltype=bertmodeltype,rnntype=rnntype)
    inf.inference()


if __name__ == "__main__":
    main()

