# HyKnow

This is the code for paper "HyKnow: End-to-End Task-Oriented Dialog Modeling with Hybrid Knowledge Management".

If you use any source code or dataset included in this repo in your work, please cite this paper:

```
@inproceedings{gao2021hyknow,
  title={HyKnow: End-to-End Task-Oriented Dialog Modeling with Hybrid Knowledge Management},
  author={Gao, Silin and Takanobu, Ryuichi and Peng, Wei and Liu, Qun and Huang, Minlie},
  booktitle={ACL-IJCNLP: Findings},
  year={2021}
}
```

## Requirements

- Python 3.6
- PyTorch 1.2.0
- NLTK 3.4.5
- Spacy 2.2.2

We use some NLP tools in NLTK which can be installed through:
```
python -m nltk.downloader stopwords punkt wordnet
```

## Dataset
1. Raw dataset: [modified MultiWOZ 2.1](https://github.com/alexa/alexa-with-dstc9-track1-dataset)

2. Our preprocessed dataset can be downloaded from [this link](https://drive.google.com/file/d/1a-JnNEGkd_2HhQsF1wLQ1BE2D_cJ375B), please unzip the file under the root directory and data is placed in ``data/``.

## Implementations of HyKnow
We build HyKnow in both single-decoder and multi-decoder belief state decoding implementations.

HyKnow with single-decoder belief state decoding implementation: ``HyKnow_Single/``
HyKnow with multi-decoder belief state decoding implementation: ``HyKnow_Multiple/``

## Running Experiments
Before running, place the preprocessed dataset ``data/`` into ``HyKnow_Single/`` or ``HyKnow_Multiple/``.
Go to the experiment root:
```
cd HyKnow_Single
```
or
```
cd HyKnow_Multiple
```

### Training
```
python train.py -mode train -dataset multiwoz -method bssmc -c spv_proportion=100 exp_no=your_exp_name
```

### Testing
```
python train.py -mode test -dataset multiwoz -method bssmc -c eval_load_path=[experimental path]
```

## Best Results
We release the best results obtained by the two implementations of our model. Please unzip the file from [this link](https://drive.google.com/file/d/1a-JnNEGkd_2HhQsF1wLQ1BE2D_cJ375B), and results are placed in ``results/``