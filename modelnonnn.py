import pandas as pd
import re
import nltk

from copy import deepcopy
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from tqdm import tqdm
from textblob import TextBlob
from nltk.corpus import sentiwordnet
nltk.download('omw-1.4')
nltk.download('sentiwordnet')

from allennlp.predictors.predictor import Predictor


predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",cuda_device=0)


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

postagcategoryfeats = {k + '_category':v for k,v in postagfeats.items()}
depcategoryfeats = {k + '_category':v for k,v in deprelfeats.items()}

moodfeats = {'ind':0,'cnd':0,'imp':0,'pot':0,'sub':0,'jus':0,'prp':0,'qot':0,'opt':0,'des':0,'nec':0,'irr':0,'adm':0}
personfeats = {1:0,2:0,3:0}

verbformfeats = {'conv':0,'fin':0,'gdv':0,'ger':0,'inf':0,'part':0,'sup':0,'vnoun':0}
voicefeats = {'act':0,'antip':0,'bfoc':0,'cau':0,'dir':0,'inv':0,'lfoc':0,'mid':0,'pass':0,'rcp':0}
polarityfeats = {'neg':0,'pos':0}
casefeats = {'abs':0,'erg':0,'acc':0,'nom':0}
tensefeats = {'fut':0,'imp':0,'past':0,'pqp':0,'pres':0}

additiveadverbs = ['still','even','again','hardly','sadly','otherwise','always','particularly','never']

devids = []
devfile = 'data/dev_ids.txt'

traindata = None
testdata = None

sentimentlexicon = {}
subjectivelexicon = set()

def read_subjectivity_lexicon():
    with open ('/home/nitin/Downloads/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjectivity.tff','r') as subj:
        for line in subj.readlines():
            line = line.split(' ')
            key = line[2].split('=')[1]
            #subjectivity = line[0].split('=')[1]
            #if subjectivity == 'strongsubj':
            subjectivelexicon.add(key)

def read_sentiment_lexicon():
    with open('/home/nitin/Desktop/sentiwordnet/SentiWordNet/data/SentiWordNet_3.0.0.txt','r') as senti:

        for line in senti.readlines():

            if line.startswith('#'): continue
            line = line.split('\t')
            synsets = line[4].split(' ')
            synsets = [s[:-2] for s in synsets]
            for syn in synsets:
                if syn not in sentimentlexicon.keys():
                    sentimentlexicon[syn] = {}
                    sentimentlexicon[syn]['pos'] = float(line[2])
                    sentimentlexicon[syn]['neg'] = float(line[3])
                else:
                    sentimentlexicon[syn]['pos'] += float(line[2])
                    sentimentlexicon[syn]['neg'] += float(line[3])


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

def count_authority_tokens(dataframe):

    tokens = []
    for _,row in dataframe.iterrows():
        if int(row['authorityvoice']) == 1:
            splits = str(row['splits']).replace('\t',' . ')
            splits = splits.split(' ')
            tokens.extend(splits)

    counts = dict(Counter(tokens))
    with open('authoritytokens.csv','w') as aut:
        for k,v in counts.items():
            if k.strip() != '':
                aut.write(str(k.strip()) + ',' + str(v) + '\n')



def build_features(dataframe):

    alldata = {}
    counter = 0
    read_sentiment_lexicon()
    read_subjectivity_lexicon()

    for index, row in tqdm(dataframe.iterrows()):


        depcounts = deepcopy(deprelfeats)
        poscounts = deepcopy(postagfeats)
        moodcounts = deepcopy(moodfeats)
        personcounts = deepcopy(personfeats)
        verbformcounts = deepcopy(verbformfeats)
        voicecounts = deepcopy(voicefeats)
        polarcounts = deepcopy(polarityfeats)
        casecounts = deepcopy(casefeats)
        tensecounts = deepcopy(tensefeats)

        depcats = deepcopy(depcategoryfeats)
        poscats = deepcopy(postagcategoryfeats)

        category = str(row['category'])
        category = category.split()
        if len(category) > 1:
            category = category[1]
        else:
            category = category[0]

        deps = str(row['deps']).replace('\t',' punct ')
        pos = str(row['xpos']).replace('\t',' . ')
        feats = str(row['feats'].replace('\t',' _ ')).lower().strip()
        lengths = int(row['lengths'])

        counts = dict(Counter(deps.split(' ')))
        for key,value in counts.items():
            #depcounts[key] += value / lengths
            depcounts[key] += value / len(deps.split(' '))
            #depcounts[key] = 1

        poslist = dict(Counter(pos.split(' ')))
        for k, v in poslist.items():
            poscounts[k] += v / len(pos.split(' '))

        feats = feats.split(' ')
        for feat in feats:
            moods = [x.group() for x in re.finditer(r'mood=[a-z]{3}',feat)]
            persons = [x.group() for x in re.finditer(r'person=[0-4]{1}',feat)]
            verbforms = [x.group() for x in re.finditer(r'verbform=[a-z]+',feat)]
            voiceforms = [x.group() for x in re.finditer(r'voice=[a-z]+',feat)]
            polarforms = [x.group() for x in re.finditer(r'polarity=[a-z]{3}',feat)]
            caseforms = [x.group() for x in re.finditer(r'case=[a-z]{3}',feat)]
            tenseforms = [x.group() for x in re.finditer(r'tense=[a-z]+',feat)]


            for mood in moods:
                m = mood.split('=')[1]
                moodcounts[m] = 1
            for person in persons:
                p = int(person.split('=')[1])
                personcounts[p] += 1
            for verb in verbforms:
                v = verb.split('=')[1]
                verbformcounts[v] += 1
            for voice in voiceforms:
                v = voice.split('=')[1]
                voicecounts[v] += 1
            for polar in polarforms:
                p = polar.split('=')[1]
                polarcounts[p] += 1
            for case in caseforms:
                c = case.split('=')[1]
                casecounts[c] += 1
            for tense in tenseforms:
                t = tense.split('=')[1]
                tensecounts[t] += 1

        personcounts = {k:v / len(feats) for k,v in personcounts.items()}
        verbformcounts = {k: v / len(feats) for k, v in verbformcounts.items()}
        voicecounts = {k: v / len(feats) for k, v in voicecounts.items()}
        polarcounts = {k: v / len(feats) for k, v in polarcounts.items()}
        casecounts = {k: v / len(feats) for k, v in casecounts.items()}
        tensecounts = {k: v / len(feats) for k, v in tensecounts.items()}

        sentences = str(row['splits']).split('\t')

        depcounts['subjectivity'] = 0
        for sent in sentences:
            polarity = TextBlob(sent)
            depcounts['subjectivity'] += polarity.sentiment.subjectivity

        depcounts['subjectivity'] /= lengths

        depcounts['positive'] = 0
        depcounts['negative'] = 0
        depcounts['subjectivewords'] = 0

        """
        depcounts['arg0counts'] = 0
        depcounts['arg1counts'] = 0

        for sent in sentences:
            result = predictor.predict(sentence=sent)
            for verb in result['verbs']:
                arg1 = [x.group() for x in re.finditer(r'\[ARG1:[\w\d\s\'\,\?\;\:\-\'\"\.]+\]',verb['description'])]
                for arg in arg1:
                    arg = arg.replace('[','').replace(']','').replace('ARG1:','').strip()
                    if category in ['need','families']:
                        if 'in need' in arg or 'poor families' in arg:
                            depcounts['arg1counts'] = 1
                    elif category in arg:
                        depcounts['arg1counts'] = 1

                arg0 = [x.group() for x in re.finditer(r'\[ARG0:[\w\d\s\'\,\?\;\:\-\'\"\.]+\]', verb['description'])]
                for arg in arg0:
                    arg = arg.replace('[', '').replace(']', '').replace('ARG0:', '').strip()
                    if category in ['need','families']:
                        if 'in need' in arg or 'poor families' in arg:
                            depcounts['arg0counts'] = 1
                    elif category in arg:
                        depcounts['arg0counts'] = 1

        #depcounts['arg0counts'] /= len(feats)
        #depcounts['arg1counts'] /= len(feats)
        """

        depcounts['authoritytokens'] = 0
        sentences = str(row['splits']).replace('\t',' . ')
        for word in sentences.split(' '):
            if word.strip() in sentimentlexicon.keys():
                depcounts['positive'] += sentimentlexicon[word]['pos']
                depcounts['negative'] += sentimentlexicon[word]['neg']
            if word.strip() in subjectivelexicon:
                depcounts['subjectivewords'] += 1

            if word.strip() in ['must','should','ought','obliged','obligated']:
                depcounts['authoritytokens'] += 1


        depcounts['authoritytokens'] /= len(sentences.split(' '))
        depcounts['positive'] /= len(sentences.split(' '))
        depcounts['negative'] /= len(sentences.split(' '))
        depcounts['subjectivewords'] /= len(sentences.split(' '))
        #depcounts['theytokens'] = float(row['theytokens'])

        depsplits = deps.split(' ')
        sents = sentences.split(' ')
        poss = pos.split(' ')

        for i in range(0,len(sents)):
            if sents[i].strip() == category.strip():
                depcats[depsplits[i] + '_category'] += 1
                poscats[poss[i] + '_category'] += 1

        depcats = {k:v / len(sents) for k,v in depcats.items()}
        poscats = {k: v / len(sents) for k, v in poscats.items()}


        #depcounts['presuppadverb'] = 0
        #depcounts['mostsubj'] = 0
        #depcounts['mostobj'] = 0

        """
        for i in range(0,len(sents)):
            if sents[i].strip() in additiveadverbs and deps[i].strip() == 'advmod':
                depcounts['presuppadverb'] += 1


            
            if sents[i].strip() == 'most' and deps[i].strip()  == 'nsubj':
                depcounts['mostsubj'] += 1

            
            if sents[i].strip() == 'most' and deps[i].strip() in ['nsubj:pass','obj']:
                depcounts['mostobj'] += 1
            """


        #depcounts['presuppadverb'] /= len(sents)
        #depcounts['mostsubj'] /= len(sents)
        #depcounts['mostobj'] /= len(sents)


        #label = int(row['label'])
        label = int(row['label'])
        depcounts['label'] = label

        lineid = int(row['lineid'])
        depcounts['lineid'] = lineid
        depcounts['splits'] = str(row['splits'])

        depcounts.update(poscounts)
        depcounts.update(moodcounts)
        #depcounts.update(personcounts)
        depcounts.update(verbformcounts)
        #depcounts.update(voicecounts)
        #depcounts.update(polarcounts)
        #depcounts.update(casecounts)
        depcounts.update(tensecounts)
        #depcounts.update(depcats)
        depcounts.update(poscats)

        counter += 1
        alldata[counter] = depcounts

    return alldata




def build_dataframes_run_models():

    get_devids()
    traindata,testdata = load_preprocessed_data()

    #count_authority_tokens(traindata)
    trainfeatures = build_features(traindata)
    traindatadf = pd.DataFrame.from_dict(data = trainfeatures,orient='index')

    testfeatures = build_features(testdata)
    testdatadf = pd.DataFrame.from_dict(data=testfeatures,orient='index')

    trainlabels = traindatadf['label'].tolist()
    traindatadf = traindatadf.set_index('lineid')
    traindatadf.drop(['splits','label'],axis=1,inplace=True)

    traindatadf.to_csv('train_features.csv')

    testlabels = testdatadf[['lineid','label','splits']]
    testdatadf = testdatadf.set_index('lineid')
    testlabels = testlabels.set_index('lineid')
    testdatadf = testdatadf.loc[devids]
    testlabels = testlabels.loc[devids]
    testdatadf.drop(['label','splits'],axis=1,inplace=True)

    testdatadf.to_csv('test_features.csv')

    clf = svm.SVC(class_weight={1:10,0:1},random_state=43,C=1)
    clf.fit(traindatadf,trainlabels)

    preds = clf.predict(testdatadf)
    predstrain = clf.predict(traindatadf)

    labs = testlabels['label'].tolist()
    testdatadf['preds'] = preds
    testdatadf['splits'] = testlabels['splits'].tolist()
    testdatadf['label'] =  labs

    testdatadf.to_csv('error_svc.csv')


    print(f1_score(labs,preds))
    print(precision_score(labs,preds))
    print(recall_score(labs, preds))

    print(f1_score(trainlabels,predstrain))
    print(precision_score(trainlabels, predstrain))
    print(recall_score(trainlabels, predstrain))




def main():
    build_dataframes_run_models()

if __name__ == "__main__":
    main()