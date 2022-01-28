import pandas as pd

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV

class Blender():
    def __init__(self,probfile,cnnprobfile,testprobs):
        self.probfile = probfile
        self.cnnprobfile = cnnprobfile

        self.data = pd.read_csv(probfile)
        self.datacnn = pd.read_csv(cnnprobfile)

        self.test = pd.read_csv(testprobs)

        self.y = self.data['label']
        self.preds = self.data['lineid']
        self.X = self.data.drop(['lineid','label'],axis=1)

        self.testpreds = self.test['lineid']
        self.Xtest = self.test.drop(['lineid'],axis=1)

        self.ycnn = self.datacnn['label']
        self.predscnn = self.datacnn['lineid']
        self.Xcnn = self.datacnn.drop(['lineid','label'],axis=1)

        for i in range(0,len(self.X.columns)):
            assert self.X.columns[i] == self.Xtest.columns[i]


        self.probthreshold = 0.5 # magic threshold

        self.devids = []
        self.devfile = 'data/dev_ids.txt'

        self.get_devids()

    def get_devids(self):
        with open(self.devfile, 'r') as ds:
            for line in ds.readlines():
                line = int(line.split(',')[0])
                self.devids.append(line)


    def svcblendscombo(self):

        combo = self.preds.merge(self.predscnn,on='lineid',how='inner')
        combo.to_csv('combopreds.csv',index=True)

    def svcblendcnn(self):
        parameters = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly']}
        scaler = StandardScaler()
        data = scaler.fit_transform(self.Xcnn)

        svc = SVC(gamma='scale',probability=True,random_state=42,class_weight='balanced')
        clf = GridSearchCV(estimator=svc,param_grid=parameters,n_jobs=-1,cv=10,scoring='f1_micro').fit(data,self.ycnn)
        print(clf.best_params_)

        proba = clf.best_estimator_.predict_proba(data)[:, 1].squeeze()
        preds = (proba > self.probthreshold).astype(int)

        self.predscnn = pd.concat([self.predscnn, pd.Series(preds.tolist())], axis=1)
        self.predscnn.columns = ['lineid', 'blendedpredscnn']

        self.predscnn = self.predscnn.set_index('lineid')
        self.predscnn = self.predscnn.loc[self.devids]

        self.predscnn.to_csv('blendedpredscnn.csv', index=True)


    def svcblend(self):

        parameters = {'C':[0.1,1,10],'kernel':['rbf','poly']}
        scaler = StandardScaler()
        data = scaler.fit_transform(self.X)

        testdata = scaler.fit_transform(self.Xtest)

        #svc = SVC(gamma='scale',probability=True, random_state=42,class_weight='balanced')
        #clf = GridSearchCV(estimator=svc,param_grid=parameters,n_jobs=-1,cv=10,scoring='f1_micro').fit(data,self.y)
        clf = SVC(gamma='scale', probability=True, random_state=42, class_weight='balanced',C=1,kernel='rbf').fit(data,self.y)
        #print(clf.best_params_)


        #proba = clf.best_estimator_.predict_proba(data)[:,1].squeeze()
        proba = clf.predict_proba(data)[:,1].squeeze()

        preds = (proba > self.probthreshold).astype(int)

        #testproba = clf.best_estimator_.predict_proba(testdata)[:,1].squeeze()
        testproba = clf.predict_proba(testdata)[:,1].squeeze()
        testpreds = (testproba > self.probthreshold).astype(int)

        self.preds = pd.concat([self.preds, pd.Series(preds.tolist())], axis=1)
        self.preds.columns = ['lineid', 'blendedpreds']
        self.preds = self.preds.loc[self.preds.groupby(['lineid'])['blendedpreds'].idxmax()].reset_index(drop=True)

        self.preds = self.preds.set_index('lineid')
        self.preds = self.preds.loc[self.devids]

        self.preds.to_csv('blendedpreds.csv', index=True)

        self.testpreds = pd.concat([self.testpreds,pd.Series(testpreds.tolist())],axis=1)
        self.testpreds.columns = ['lineid','blendedpreds']
        self.testpreds = self.testpreds.loc[self.testpreds.groupby(['lineid'])['blendedpreds'].idxmax()].reset_index(drop=True)

        self.testpreds = self.testpreds.set_index('lineid')
        self.testpreds.to_csv('testpreds.csv',index=True)




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
    cnnprobfile = '/home/nitin/Desktop/dontpatronize/dontpatronize/data/errors/cnn_proba.csv'

    testprobs = '/home/nitin/Desktop/dontpatronize/dontpatronize/data/proba/testproba/testprobacombo.csv'

    blender = Blender(probfile=probfile,cnnprobfile=cnnprobfile,testprobs=testprobs)

    #blender.logisticblend()
    #blender.randomforestblend()
    blender.svcblend()
    #blender.svcblendcnn()
    #blender.svcblendscombo()
    #blender.gettreeblend()

if __name__ == "__main__":
    main()