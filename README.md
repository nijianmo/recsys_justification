### recsy justification
This is the code for our EMNLP 19' work
- Justifying recommendations using distantly-labeled reviews and fined-grained aspects, Jianmo Ni, Jiacheng Li, Julian McAuley, Empirical Methods in Natural Language Processing (EMNLP) 2019.

This repo follows the following hierarchy:
```
recsys_justification
|---justitication_classifier
|---reference2seq
|---acmlm
```

### justification classifier 
This is the fine-tuned BERT model that used to train on the labeled justification data. You can simply train the model via `run.sh` and conduct inference over any unlabeled data using `predict.sh`, after you change the data loader correspondingly in the python file. We also provide a pre-trained model here.
	- [bert_config.json](http://deepyeti.ucsd.edu/jianmo/recsys_justification/model/justification_classifier/bert_config.json). 
	- [pytorch_model.bin](http://deepyeti.ucsd.edu/jianmo/recsys_justification/model/justification_classifier/pytorch_model.bin). 

### reference2seq
This is the proposed reference2seq model. It contains files for data processing and model training/evaluation.

### acmlm
This is the proposed aspect-conditional masked language model (acmlm).

### Data
* 2000 labeled data that includes a binary label for each element discourse unit (EDU) in reviews. You can find it under `justification_classifier`.
* Distantly labeled dataset derived from the Yelp and Amazon Clothing dataset. Each line of the json file includes an EDU from a review and the fine-grained aspects convered in it.
    - [Download Yelp](http://deepyeti.ucsd.edu/jianmo/recsys_justification/data/yelp_filter_flat_positive.large.json)
    - [Download Amazon Clothing](http://deepyeti.ucsd.edu/jianmo/recsys_justification/data/cloth_filter_flat_positive.large.json)


### Newly released Amazon product review dataset.
We will soon release a new version of the Amazon product review dataset which increases the reviews in the period from 2014~2018!


### Requirements
- PyTorch=0.4

Please cite our paper if you find the data and code helpful, thanks!
```
@inproceedings{Ni2019RecsysJust
  title={Justifying recommendations using distantly-labeled reviews and fined-grained aspects},
  author={Jianmo Ni and Jiacheng Li and Julian McAuley},
  booktitle={EMNLP},
  year={2019}
}
```

