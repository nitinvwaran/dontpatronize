import pandas as pd
import stanza
import re
import numpy as np
import os

from nltk import word_tokenize
from nltk import ParentedTree
from tqdm import tqdm

class PreprocessingUtils():

    def __init__(self,pclfile,categoriesfile,testfile):

        self.nlp = stanza.Pipeline(lang='en',processors='tokenize,pos,constituency')
        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile
        self.devfile = 'data/dev_ids.txt'
        self.devids = []
        self.labeledids = set()
        self.punctuation = r'@#^\*|~`<>\\$'
        self.regextag = r'<.+>'
        self.datafile = 'data/preprocesseddata.tsv'
        self.talkdowndatafile = 'data/talkdownpreprocesseddata.tsv'
        self.constituentphrasecutoff = 5
        self.get_categoriesids()

        # anything starting with http(s):// , ftp(s):// , www and ending with a space as URLs cannot contain spaces
        self.urlregex1 = r'((http(s)?|ftp(s)?|HTTP(S)?|FTP(S)?):\/\/|(www\.|WWW\.))[^\s]*'

        # URL regex to get anything not starting with http(s), ftp(s), www, but get strings that end with common domain names and ccTLDs
        self.urlregex2 = r'[^\s]+(\.gov|\.news|\.pro|\.com|\.net|\.org|\.xxx|\.one|\.ac|\.ad|\.ae|\.af|\.ag|\.ai|\.al|\.am|\.an|\.ao|\.aq|\.ar|\.as|\.at|\.au|\.aw|\.ax|\.az|\.ba|\.bb|\.bd|\.be|\.bf|\.bg|\.bh|\.bi|\.bj|\.bl|\.bm|\.bn|\.bo|\.br|\.bq|\.bs|\.bt|\.bv|\.bw|\.by|\.bz|\.ca|\.cc|\.cd|\.cf|\.cg|\.ch|\.ci|\.ck|\.cl|\.cm|\.cn|\.co|\.cr|\.cs|\.cu|\.cv|\.cw|\.cx|\.cy|\.cz|\.dd|\.de|\.dj|\.dk|\.dm|\.do|\.dz|\.ec|\.ee|\.eg|\.eh|\.er|\.es|\.et|\.eu|\.fi|\.fj|\.fk|\.fm|\.fo|\.fr|\.ga|\.gb|\.gd|\.ge|\.gf|\.gg|\.gh|\.gi|\.gl|\.gm|\.gn|\.gp|\.gq|\.gr|\.gs|\.gt|\.gu|\.gw|\.gy|\.hk|\.hm|\.hn|\.hr|\.ht|\.hu|\.id|\.ie|\.il|\.im|\.in|\.io|\.iq|\.ir|\.is|\.it|\.je|\.jm|\.jo|\.jp|\.ke|\.kg|\.kh|\.ki|\.km|\.kn|\.kp|\.kr|\.kw|\.ky|\.kz|\.la|\.lb|\.lc|\.li|\.lk|\.lr|\.ls|\.lt|\.lu|\.lv|\.ly|\.ma|\.mc|\.md|\.me|\.mf|\.mg|\.mh|\.mk|\.ml|\.mm|\.mn|\.mo|\.mp|\.mq|\.mr|\.ms|\.mt|\.mu|\.mv|\.mw|\.mx|\.my|\.mz|\.na|\.nc|\.ne|\.nf|\.ng|\.ni|\.nl|\.no|\.np|\.nr|\.nu|\.nz|\.om|\.pa|\.pe|\.pf|\.pg|\.ph|\.pk|\.pl|\.pm|\.pn|\.pr|\.ps|\.pt|\.pw|\.py|\.qa|\.re|\.ro|\.rs|\.ru|\.rw|\.sa|\.sb|\.sc|\.sd|\.se|\.sg|\.sh|\.si|\.sj|\.sk|\.sl|\.sm|\.sn|\.so|\.sr|\.ss|\.st|\.su|\.sv|\.sx|\.sy|\.sz|\.tc|\.td|\.tf|\.tg|\.th|\.tj|\.tk|\.tl|\.tm|\.tn|\.to|\.tp|\.tr|\.tt|\.tv|\.tw|\.tz|\.ua|\.ug|\.uk|\.um|\.us|\.uy|\.uz|\.va|\.vc|\.ve|\.vg|\.vi|\.vn|\.vu|\.wf|\.ws|\.ye|\.yt|\.yu|\.za|\.zm|\.zr|\.zw)[\/|\s]'

        # IPV4 regex, will get IPV4 addresses surrounded by whitespaces or the backslash (for URLs)
        # will not get embedded IPV4 addresses not surrounded by above delimiters
        # Extra validation on python code will check range of 0-255
        self.ipv4regex = r'(?<![\.|[0-9a-zA-Z~`!@#$%^&*()_\-+=|\\\]\[\}\{\'\";:?\,><])[\/]?([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})([^\.|^[0-9a-zA-Z~`!@#$%^&*()_\-+=|\\\]\[\}\{\'\";:?\,><]|[\/])'

        # taken from the official docs: https://datatracker.ietf.org/doc/html/rfc5322
        self.emailaddressregex = r'(?:[a-z0-9!#$%&\'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+\/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'

        self.presuppositionadverbs = {'more likely','likely to','typical','yet','also','never','most','just','already','still','even','really','much rather','ever','ever again','such','otherwise','rather than','also','usually'}

    #def presuppositiontriggers(self,phrase):




    def get_train_test_data(self,usetalkdown=False):

        self.refinedlabelsdev = pd.read_csv('refinedlabelsdev.csv')
        talkdowndata = pd.read_csv(self.talkdowndatafile,sep='\t')

        if os.path.isfile('traindata.tsv') and os.path.isfile('devdata.tsv'):
            self.traindata = pd.read_csv('traindata.tsv',sep='\t')
            self.devdata = pd.read_csv('devdata.tsv',sep='\t')

            if usetalkdown == True:
                self.traindata = pd.concat([self.traindata,talkdowndata],axis=0)

            self.traindata.set_index('lineid')
            self.devdata.set_index('lineid')

        else:

            lengths = []
            phraselengths = []

            data = pd.read_csv(self.datafile,sep='\t')
            data = data.sample(frac=1).reset_index(drop=True)

            for _,row in data.iterrows():
                lengths.append(len(str(row['text']).split()))
                phraselengths.append(len(str(row['phrase']).split()))

            self.get_devids()

            mask = data['lineid'].isin(self.devids)
            self.traindata = data.loc[~mask]
            self.devdata = data.loc[mask]

            if usetalkdown == True:
                self.traindata = pd.concat([self.traindata,talkdowndata],axis=0)

            self.traindata.set_index('lineid')
            self.devdata.set_index('lineid')

            self.traindata.to_csv('traindata.tsv',sep='\t',index=False)
            self.devdata.to_csv('devdata.tsv',sep='\t',index=False)


            print('text stats')
            print(np.percentile(lengths,50))
            print(np.percentile(lengths, 90))
            print(np.percentile(lengths, 95))
            print(np.percentile(lengths, 99))
            print(max(lengths))
    
            print('phrase stats')
            print(np.percentile(phraselengths, 50))
            print(np.percentile(phraselengths, 90))
            print(np.percentile(phraselengths, 95))
            print(np.percentile(phraselengths, 99))
            print(max(phraselengths))


    def get_categoriesids(self):

        with open(self.categoriesfile,'r') as cats:
            for line in cats.readlines():
                lineid = int(line.split('\t')[0])

                self.labeledids.add(lineid)



    def get_devids(self):
        with open(self.devfile, 'r') as ds:
            for line in ds.readlines():
                line = int(line.split(',')[0])
                self.devids.append(line)


    def clean_string(self,line):

        line = line.strip()
        line = line.replace('&amp;', ' and ').replace('&apos;', '').replace('&gt;', '').replace('&lt;', '').replace('&quot;', '"')

        line = re.sub(self.regextag, ' ', line)

        line = re.sub(self.urlregex2,' ',line)
        line = re.sub(self.urlregex1, ' ', line)
        line = re.sub(self.emailaddressregex, ' ', line)
        line = re.sub(self.ipv4regex, ' ', line)

        for punct in self.punctuation:
            line = line.replace(punct, '')

        line = re.sub(r'(\! )+', '!', line)
        line = re.sub(r'\!+', ' ! ', line)
        line = re.sub(r'(\? )+', '?', line)
        line = re.sub(r'\?+', ' ? ', line)
        line = re.sub(r'\.\.+', ' ', line)
        line = re.sub(r'"+', '"', line)
        line = line.replace('"', ' ')
        line = re.sub(r'-+', ' ', line)
        line = line.replace('/', ' or ')
        line = line.replace(')', ' ')
        line = line.replace('(', ' ')
        line = line.replace('%',' percent ')
        line = re.sub(r'\.(?! )',' . ',line)
        line = re.sub('([.])', r' \1 ', line)

        line = re.sub(r' +', ' ', line)

        return line.strip()

    def split_string(self,line):

        splits = line.split(' . ')
        splits = [s.split(' ? ') for s in splits]
        splits = [s for inner in splits for s in inner]
        splits = [s.split(' ! ') for s in splits]
        splits = [s for inner in splits for s in inner]
        splits = [s.split(' ; ') for s in splits]
        splits = [s for inner in splits for s in inner]

        splits = [s.strip() for s in splits if s]

        return splits

    def parse_tree(self,tree,constituents):

        if type(tree) != str and tree.label() in ['VP','S','SINV','NP']:
            if len(tree.leaves()) >= self.constituentphrasecutoff:
                if tree.leaves()[0] not in ['is','are','were','am','was','\'m','\'re','\'s','be','will','to','that']:
                    phrase = ' '.join(tree.leaves())
                    constituents.add(phrase.strip())

        if type(tree) != str:
            for child in tree:
                constituents = self.parse_tree(child,constituents)

        return constituents


    def get_const_parses(self,sent):

        doc = self.nlp(sent)
        constituents = set()

        for s in doc.sentences:
            tree = str(s.constituency)
            tree = ParentedTree.fromstring(tree)

            if tree[0].label() == 'NP' or len(tree[0].leaves()) < self.constituentphrasecutoff:
                constituents.add(' '.join(tree.leaves()).strip())
            else:
                constituents = self.parse_tree(tree[0], constituents)

        return constituents

    def preprocess_talkdown(self):


        with open(self.talkdowndatafile,'w') as dataf:

            dataf.write('lineid' + '\t' + 'category' + '\t' + 'text' + '\t' + 'phrase' + '\t' + 'label' + '\n')

            with open('talkdowndata.csv','r') as talk:

                for line in tqdm(talk.readlines()):

                    lineid = int(line.split('\t')[0])
                    category = line.split('\t')[1]
                    text = line.split('\t')[2]
                    phrase = line.split('\t')[3]
                    label = int(line.split('\t')[4])


                    text = self.clean_string(text)
                    phrase = self.clean_string(phrase)

                    try:
                        if len(text) > 0:
                            text = ' '.join(word_tokenize(text))
                        else:
                            continue
                    except Exception:
                        continue



                    try:
                        if len(phrase) > 0:
                            phrase = ' '.join(word_tokenize(phrase))
                        else:
                            continue
                    except Exception:
                        continue


                    try:
                        docpos = self.nlp(text)
                    except Exception:

                        continue

                    newline = []
                    for sent in docpos.sentences:
                        for word in sent.words:
                            if word.upos in ['ADJ', 'ADV', 'INTJ', 'VERB', 'NOUN', 'PROPN', 'PRON',
                                             'ADP']: newline.append(word.text)

                    dataf.write(str(lineid) + '\t' + category.strip() + '\t' + ' '.join(newline) + '\t' + phrase + '\t' + str(label) + '\n')




    def preprocess(self):

        labeledids = set(self.labeledids)

        with open(self.datafile,'w') as dataf:
            dataf.write('lineid' + '\t' + 'category' + '\t' + 'text' + '\t' + 'phrase' + '\t' + 'label' + '\n')

            with open(self.pclfile, 'r') as pclf:

                for line in tqdm(pclf.readlines()):

                    lineid = int(line.split('\t')[0])

                    if lineid not in labeledids:

                        line = line.strip()

                        if line != '':

                            label = int(line.split('\t')[-1])

                            assert label not in [2,3,4]
                            category = line.split('\t')[2]

                            line = line.split('\t')[4]

                            line = self.clean_string(line)
                            splits = self.split_string(line)

                            docpos = self.nlp(line)
                            newline = []
                            for sent in docpos.sentences:
                                for word in sent.words:
                                    if word.upos in ['ADJ','ADV','INTJ','VERB','NOUN','PROPN','PRON','ADP']: newline.append(word.text)

                            consts = set()
                            for split in splits:
                                consts.update(self.get_const_parses(split))

                            if len(consts) > 0:
                                for const in consts:

                                    #startindex = line.find(const)
                                    #assert startindex != -1

                                    if len(const.split()) < self.constituentphrasecutoff: continue

                                    dataf.write(str(lineid) + '\t' + category.strip() + '\t' +  ' '.join(newline) + '\t' + const  + '\t' + str(0) + '\n')
                            else:
                                dataf.write(str(lineid) + '\t' + category.strip() + '\t' +  ' '.join(newline) + '\t' + line + '\t'  + str(0) + '\n')


            with open(self.categoriesfile,'r') as cats:

                for line in tqdm(cats.readlines()):

                    lineid = int(line.split('\t')[0])

                    text = line.split('\t')[2]
                    phrase = line.split('\t')[-3]

                    text = self.clean_string(text)
                    phrase = self.clean_string(phrase)

                    docpos = self.nlp(text)
                    newline = []
                    for sent in docpos.sentences:
                        for word in sent.words:
                            if word.upos in ['ADJ','ADV','INTJ','VERB','NOUN','PROPN','PRON','ADP']: newline.append(word.text)

                    dataf.write(str(lineid) + '\t' + category.strip() + '\t' + ' '.join(newline) + '\t' + phrase + '\t' + str(1) + '\n')




def main():

    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_categories.tsv'
    preprocess = PreprocessingUtils(pclfile,categoriesfile,None)
    #preprocess.preprocess()
    #preprocess.preprocess_talkdown()
    preprocess.get_train_test_data(usetalkdown=True)


if __name__ == "__main__":
    main()