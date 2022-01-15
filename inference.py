import pandas as pd
import torch
import argparse

from preprocessingutils import PreprocessingUtils
from modules import LSTMAttention
from tqdm import tqdm

class Inference():
    def __init__(self,pclfile,categoriesfile,testfile,modeltype='rnn',bertmodeltype='rawdistilbert',maxlen=256,devbatchsize=1000):

        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile

        self.preprocess = PreprocessingUtils(pclfile,categoriesfile,testfile)
        self.preprocess.get_train_test_data(usetalkdown=False,testdata=True)

        self.modeltype = modeltype
        self.bertmodeltype = bertmodeltype
        self.devbatchsize = devbatchsize

        #if modeltype == 'bert':
        #    self.model = BertModels(bertmodeltype=bertmodeltype, maxlen=maxlen)
        #elif modeltype == 'rnn':
        #    self.model = LSTMAttention(lstmtype='lstm', bertmodeltype=bertmodeltype)

        self.model = LSTMAttention(bertmodeltype=bertmodeltype,maxlen=maxlen)
        checkpoint = torch.load('/home/nitin/Desktop/dontpatronize/dontpatronize/data/checkpoint/model_test20.3863080684596577.pt')
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

            preds.to_csv('inference.csv')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--maxlen', type=int, default=224)
    parser.add_argument('--devbat', type=int, default=500)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--bertmodeltype', type=str, default='rawdistilbert')
    parser.add_argument('--modeltype', type=str, default='rnn')

    args = parser.parse_args()

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'

    inf = Inference(pclfile,categoriesfile,None,maxlen=224,devbatchsize=500)
    inf.inference()


if __name__ == "__main__":
    main()

