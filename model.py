import torch
import torch.nn as nn

from transformers import RobertaTokenizer, RobertaModel
from preprocessing import PreProcessing


class RobertaWrapper():
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bertmodel = RobertaWrapper()
        self.bertmodel.model.to(self.device)

        self.maxlen = 67

    def forward(self,dataframe):

        """
        :param dataframe: a dataframe with the splits and the lengths and labels
        """
        data = [] # holds flattened sentences

        sentences = dataframe['splits'].tolist()
        lengths = dataframe['lengths'].tolist()
        labels = dataframe['label'].tolist()

        for sent in sentences:
            s = sent.split('\t')
            data.extend(s)

        inp = self.bertmodel.tokenizer(data, max_length=self.maxlen, padding='max_length', truncation=True,add_special_tokens=False,return_tensors='pt')
        inp.to(self.device)
        inp['output_hidden_states'] = True
        attentionmask = inp['attention_mask']

        output = self.bertmodel.model(**inp)
        lasthiddenstate = output['last_hidden_state']

        input_mask_expanded = attentionmask.unsqueeze(-1).expand(lasthiddenstate.size()).float()
        lasthiddenstate[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        maxvectors = torch.max(lasthiddenstate, 1)[0]

        i = 0
        for l in lengths:
            if i == 0:
                squeezedvectors = torch.max(maxvectors[i:i + l,:],0)[0]
                squeezedvectors = torch.unsqueeze(squeezedvectors,0)
            else:
                temp = torch.max(maxvectors[i:i + l, :], 0)[0]
                temp = torch.unsqueeze(temp, 0)
                squeezedvectors = torch.cat((squeezedvectors,temp),0)

            i += l

        assert squeezedvectors.size(dim=0) == len(dataframe)
        print('here')












def main():
    pclfile = 'data/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv'
    categoriesfile = None

    model = Model()
    preprocess = PreProcessing(pclfile,categoriesfile)
    preprocess.preprocess_data()
    sample = preprocess.traindata.sample(n=32,random_state=3)

    model(sample)




if __name__ == "__main__":
    main()




