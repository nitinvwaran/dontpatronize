import pandas as pd

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

class Blender():
    def __init__(self,probfile):
        self.probfile = probfile
        self.data = pd.read_csv(probfile)

        self.y = self.data['label']
        self.preds = self.data['lineid']
        self.X = self.data.drop(['lineid','label'],axis=1)

        self.probthreshold = 0.3

        self.devids = []
        self.devfile = 'data/dev_ids.txt'

        self.get_devids()

    def get_devids(self):
        with open(self.devfile, 'r') as ds:
            for line in ds.readlines():
                line = int(line.split(',')[0])
                self.devids.append(line)

    def svcblend(self):
        clf = SVC(gamma='scale',probability=True, random_state=42,class_weight={0:1,1:5})
        pipe = make_pipeline(StandardScaler(), clf)

        proba = cross_val_predict(pipe,self.X,self.y,cv=10,method='predict_proba')
        proba = proba[:,1].squeeze()
        preds = (proba > self.probthreshold).astype(int)

        self.preds = pd.concat([self.preds, pd.Series(preds.tolist())], axis=1)
        self.preds.columns = ['lineid', 'blendedpreds']
        self.preds = self.preds.loc[self.preds.groupby(['lineid'])['blendedpreds'].idxmax()].reset_index(drop=True)

        self.preds = self.preds.set_index('lineid')
        self.preds = self.preds.loc[self.devids]

        self.preds.to_csv('blendedpreds.csv', index=True)


    def logisticblend(self):
        clf = LogisticRegression(random_state=42,max_iter=500,class_weight={0: 1, 1: 3})
        pipe = make_pipeline(StandardScaler(),clf)

        proba = cross_val_predict(pipe, self.X, self.y, cv=10, method='predict_proba')
        proba = proba[:, 1].squeeze()
        preds = (proba > self.probthreshold).astype(int)


        self.preds = pd.concat([self.preds,pd.Series(preds.tolist())],axis=1)
        self.preds.columns = ['lineid','blendedpreds']
        self.preds = self.preds.loc[self.preds.groupby(['lineid'])['blendedpreds'].idxmax()].reset_index(drop=True)

        self.preds = self.preds.set_index('lineid')
        self.preds = self.preds.loc[self.devids]

        self.preds.to_csv('blendedpreds.csv',index=True)


def main():
    probfile = '/home/nitin/Desktop/dontpatronize/dontpatronize/data/proba/blendedcombo.csv'
    blender = Blender(probfile=probfile)

    #blender.logisticblend()
    #blender.randomforestblend()
    blender.svcblend()
    #blender.gettreeblend()

if __name__ == "__main__":
    main()