# System submission for [SemEval2022 Task 4](https://sites.google.com/view/pcl-detection-semeval2022/)

## Introduction / Description

This repo contains my entry for [SemEval2022 Task 4](https://sites.google.com/view/pcl-detection-semeval2022/) contest - Patronising and Condescending Language Detection.

This system placed 26th of 80 in [Task 1](https://competitions.codalab.org/competitions/34344#results) (Top 33%) and 22nd of 50 in [Task 2](https://competitions.codalab.org/competitions/34344#results). Look at the 'Evaluation' phase, ranked by F1 / average F1. My handle is ```thetundraman```

## Main Ideas

The main premise behind the system is that condescension and patronising language is signalled by shorter phrases / constituents, or by single-word lexical items. 

The ideas behind [Wang & Potts 2019](https://arxiv.org/abs/1909.11272) were used, who found the importance of both a condescending phrase and its context necessary in determining condescension. This was extended to this system by considering both a shorter NP / VP sub-constituent and the whole sentence / paragraph it is found in , with the premise that the condescending signal is found in the sub-constituent and guided by the overall sentence / paragraph.

### Three models types were built:
#### 1. CNNs for Sentence Classification following [Yoon Kim 2014](https://arxiv.org/pdf/1408.5882.pdf)
This follows the general architecture from Yoon Kim 2014. 9-grams, 7-grams, 5-grams and 3-grams are considered as encompassing condescending signals. 

#### 2. LSTM / GRU with added Attention Head
This is an LSTM / GRU with an added Attention head on top. Extrapolating from [Wang & Potts 2019](https://arxiv.org/abs/1909.11272), the context and phrase are both passed through separate LSTM / GRU layers, and an Attention head on top attends to information from the context sentence / paragraph, which is concatenated to the hidden state of the phrase signal and passed through a feed-forward layer for final classification. 

This approach turned out to be quite as competitive as the following approach in terms of F1 score. 

#### 3. Pre-trained BERT models 
These are pre-trained models with a Sequence Classification head on top. Following [Wang & Potts 2019](https://arxiv.org/abs/1909.11272), the context and phrase are submitted as sentence 1 and sentence 2, separated by a [SEP] token (for BERT, this might differ for other model types). Pre-trained models from [HuggingFace](https://huggingface.co/) were used.


#### Word embeddings
Pre-trained [HuggingFace](https://huggingface.co/) word embeddings were used for all model types: BERT, DistilBERT, RoBERTa, and XLNET. 

Interestingly, the RoBERTA embeddings consistently outperformed the others across model types. 

### Multi-label training (Task 2)

The architectures are pretty much the same, the main difference is that a Multi-Label head has been added with the BCEWithLogitsLoss in PyTorch, in place of the CrossEntropy loss used for Task 1. 


## OK, enough boring stuff. How do i run this?

A ```requirements.txt``` file has been added to setup a conda environment with. Python 3.8, PyTorch 1.10 with transformers are recommended. 

### Training

All training checkpoints end up in ```data/checkpoints```, predictions on dev data in ```data/errors```, and probabilities on dev data for blending in ```data/proba```.

#### To train pre-trained models with Sequence Classification head:
```
python modules.py --modeltype bert --bertmodeltype <bert,distilbert,roberta,xlnet> --multilabel <0,1>
```

#### To train the RNN-type models:
```
python modules.py --modeltype rnn --rnntype <lstm,gru> --bertmodeltype <rawbert,rawdistilbert,rawroberta,rawxlnet> --multilabel <0,1>
```

#### To train the CNN-type models:
```
python modules.py --modeltype cnn --bertmodeltype <rawbert,rawdistilbert,rawroberta,rawxlnet> --multilabel <0,1>
```

Other parameters:
```
--lr : learning rate
--wd : weight decay for AdamW
--maxlentext : max length of the context
--maxlenphrase : max length of the phrase
--hiddensize : hidden size parameter for RNN
--numlayers : number layers parameter for RNN
--chkpoint : optional checkpoint file. It must match the selected architecture
```

#### Tensorboard!

Tensorboards are created in ```data/tensorboarddir``` for each run. Tensorboards track the dev F1 information across eopchs, along with dev and train loss. 

This folder is re-created upon every run and so previous tensorboards will be lost on re-run. 


### To run inference with a checkpoint file:

The best model name is stored by default after training in ```bestmodel.txt``` and this is used to load the checkpoint.

```
python inference.py --multilabel <0,1>
```

Other parameters:
```
--maxlentext : max length of the context
--maxlenphrase : max length of the phrase
```






