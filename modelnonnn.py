import pandas as pd
import re

from copy import deepcopy
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from tqdm import tqdm


deprelfeats = {'nmod:npmod': 0, 'obl:npmod': 0, 'det:predet': 0, 'acl': 0, 'acl:relcl': 0, 'advcl': 0,
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

postagfeats = {'CC': 0, 'CD': 0, 'DT': 0, 'EX': 0, 'FW': 0, 'IN': 0, 'JJ': 0, 'JJR': 0, 'JJS': 0, 'LS': 0,
                    'MD': 0, 'NN': 0, 'NNS': 0, 'NNP': 0, 'NNPS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0,
                    'RB': 0, 'RBR': 0, 'RBS': 0, 'RP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0,
                    'VBG': 0, 'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0,',':0,'HYPH':0,'.':0,'``':0,"''":0,"$":0,"-RRB-":0,"-LRB-":0,
                     ':':0,'NFP':0,'ADD':0,'AFX':0}

moodfeats = {'ind':0,'cnd':0,'imp':0,'pot':0,'sub':0,'jus':0,'prp':0,'qot':0,'opt':0,'des':0,'nec':0,'irr':0,'adm':0}

devids = []
devfile = 'data/dev_ids.txt'

traindata = None
testdata = None

def get_devids():
    with open(devfile,'r') as ds:
        for line in ds.readlines():
            line = int(line.split(',')[0])
            devids.append(line)

def load_preprocessed_data():


    refinedlabelstrain = pd.read_csv('refinedlabelstrain.csv')
    refinedlabelstrain.set_index('lineid')

    refinedlabelsdev = pd.read_csv('refinedlabelsdev.csv')
    refinedlabelsdev.set_index('lineid')

    sentencedata = pd.read_csv('sentencesplits.csv')

    #get_devids()
    mask = sentencedata['lineid'].isin(devids)
    traindata = sentencedata.loc[~mask]
    testdata = sentencedata.loc[mask]

    traindata.set_index('lineid')
    testdata.set_index('lineid')

    traindata = traindata.merge(refinedlabelstrain, how='inner',on='lineid')
    testdata = testdata.merge(refinedlabelsdev, how='inner',on='lineid')

    return traindata,testdata

def build_features(dataframe):

    alldata = {}
    counter = 0



    for index, row in tqdm(dataframe.iterrows()):
        depcounts = deepcopy(deprelfeats)
        poscounts = deepcopy(postagfeats)
        moodcounts = deepcopy(moodfeats)

        deps = str(row['deps']).replace('\t',' punct ')
        pos = str(row['xpos']).replace('\t',' . ')
        feats = str(row['feats'].replace('\t',' _ ')).lower().strip()

        counts = dict(Counter(deps.split(' ')))
        for key,value in counts.items():
            #depcounts[key] += value
            depcounts[key] += value / len(deps.split(' '))
            #depcounts[key] = 1

        poslist = dict(Counter(pos.split(' ')))
        for k, v in poslist.items():
            poscounts[k] += v / len(pos.split(' '))

        feats = feats.split(' ')
        for feat in feats:
            moods = [x.group() for x in re.finditer(r'mood=[a-z]{3}',feat)]
            for mood in moods:
                m = mood.split('=')[1]
                moodcounts[m] = 1

        #label = int(row['label'])
        label = int(row['label'])
        depcounts['label'] = label

        lineid = int(row['lineid'])
        depcounts['lineid'] = lineid
        depcounts['splits'] = str(row['splits'])
        #depcounts['lengths'] = int(row['lengths'])
        depcounts.update(poscounts)
        depcounts.update(moodcounts)

        counter += 1
        alldata[counter] = depcounts

    return alldata




def build_dataframes_run_models():

    get_devids()
    traindata,testdata = load_preprocessed_data()

    trainfeatures = build_features(traindata)
    traindatadf = pd.DataFrame.from_dict(data = trainfeatures,orient='index')

    testfeatures = build_features(testdata)
    testdatadf = pd.DataFrame.from_dict(data=testfeatures,orient='index')

    trainlabels = traindatadf['label'].tolist()
    traindatadf.drop(['splits','lineid','label'],axis=1,inplace=True)

    testlabels = testdatadf[['lineid','label','splits']]
    testdatadf = testdatadf.set_index('lineid')
    testlabels = testlabels.set_index('lineid')
    testdatadf = testdatadf.loc[devids]
    testlabels = testlabels.loc[devids]
    testdatadf.drop(['label','splits'],axis=1,inplace=True)

    clf = svm.SVC(class_weight={1:10,0:1},random_state=43,C=1)
    clf.fit(traindatadf,trainlabels)

    preds = clf.predict(testdatadf)
    labs = testlabels['label'].tolist()
    testdatadf['preds'] = preds
    testdatadf['splits'] = testlabels['splits'].tolist()
    testdatadf['label'] =  labs

    testdatadf.to_csv('error_svc.csv')

    print(f1_score(labs,preds))
    print(precision_score(labs,preds))
    print(recall_score(labs, preds))



def main():
    build_dataframes_run_models()

if __name__ == "__main__":
    main()