### recsy justification
This is the code for our EMNLP 19' work
- Justifying recommendations using distantly-labeled reviews and fined-grained aspects, Jianmo Ni, Jiacheng Li, Julian McAuley, Empirical Methods in Natural Language Processing (EMNLP) 2019.

This repo follows the following hierarchy:
```
recsys_justification
|---reference2seq
|---acmlm
```

### reference2seq
This is the proposed reference2seq model.

### acmlm
This is the proposed aspect-conditional masked language model (acmlm).

### Data
- 2000 labeled data that includes a binary label for each element discourse unit (EDU) in reviews.
[Download](deepeti.ucsd.edu/jianmo/recsys_justification/label_data.csv)
- Distantly labeled dataset derived from the Yelp and Amazon Clothing dataset. Each line of the json file includes an EDU from a review and the fine-grained aspects convered in it.
[Download Yelp](deepeti.ucsd.edu/jianmo/recsys_justification/yelp_filter_flat_positive.large.json)
[Download Amazon Clothing](deepeti.ucsd.edu/jianmo/recsys_justification/cloth_filter_flat_positive.large.json)

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

