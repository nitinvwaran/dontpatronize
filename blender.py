import pandas as pd

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV

class Blender():
    def __init__(self,probfile):
        self.probfile = probfile
        self.data = pd.read_csv(probfile)

        self.y = self.data['label']
        self.preds = self.data['lineid']
        self.X = self.data.drop(['lineid','label'],axis=1)

        self.probthreshold = 0.5

        self.devids = []
        self.devfile = 'data/dev_ids.txt'

        self.get_devids()

    def get_devids(self):
        with open(self.devfile, 'r') as ds:
            for line in ds.readlines():
                line = int(line.split(',')[0])
                self.devids.append(line)

    def svcblend(self):

        parameters = {'C':[0.1,1,10],'kernel':['rbf','poly']}
        scaler = StandardScaler()
        data = scaler.fit_transform(self.X)

        svc = SVC(gamma='scale',probability=True, random_state=42,class_weight='balanced')
        clf = GridSearchCV(estimator=svc,param_grid=parameters,n_jobs=-1,cv=10,scoring='f1_micro').fit(data,self.y)
        print(clf.best_params_)


        proba = clf.best_estimator_.predict_proba(data)[:,1].squeeze()
        preds = (proba > self.probthreshold).astype(int)

        self.preds = pd.concat([self.preds, pd.Series(preds.tolist())], axis=1)
        self.preds.columns = ['lineid', 'blendedpreds']
        self.preds = self.preds.loc[self.preds.groupby(['lineid'])['blendedpreds'].idxmax()].reset_index(drop=True)

        self.preds = self.preds.set_index('lineid')
        self.preds = self.preds.loc[self.devids]

        self.preds.to_csv('blendedpreds.csv', index=True)


    def logisticblend(self):
        parameters = {'C': [0.1, 1, 10]}
        lr = LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced')
        clf = GridSearchCV(estimator=lr,param_grid=parameters,n_jobs=-1,cv=10,scoring='f1_micro').fit(self.X,self.y)
        print (clf.best_params_)

        proba = clf.best_estimator_.predict_proba(self.X)[:, 1].squeeze()
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