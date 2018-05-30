# Dynamic Memory Networks (PyTorch)
PyTorch implementation of the paper, </br>
> *Ask Me Anything: Dynamic Memory Networks for Natural Language Processing* </br>
> Kumar et al., 2016 arxiv

## Requirements
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downlaods)
- Install [cudnn-v5.1](https://developer.nvidia.com/cudnn)
- Install [Pytorch 0.4.0](https://pytorch.org/)
- Python version >= 3.5 is required

## Datasets
- bAQbI Tasks v1.2 data downloaded from [here](https://research.fb.com/downloads/babi/)
- Place files in en-valid-10k under (home)/datasets/babi/en directory.
- Place a pretrained GloVe under (home)/datasets/glove directory.

```bash
# Preprocessing dataset. This will create ./data/babi(tmp).pkl
$ python dataset.py
```

## Run experiments
```bash
# Train and test with default settings
$ python main.py

# Train with different number of hidden units, epochs, and QA sets
$ python main.py --s_rnn_hdim 200 --epoch 20 --set_num 5
```

## Model Overview
![Dynamic Memory Networks](https://yerevann.github.io/public/2016-02-06/dmn-details.png)

## Experimental Results (bAbI)
Task | Accuracy | Task | Accuracy
---- | -------- | ---- | -------
 1 | 100% | 11 | 100%
 2 | 99.51% | 12 | 100%
 3 | 88.28% | 13 | 95.21%
 4 | 100% | 14 | 100%
 5 | 99.51% | 15 | 100%
 6 | 99.51% | 16 | 100%
 7 | 98.93% | 17 | 57.13%
 8 | 95.41% | 18 | 99.41%
 9 | 99.90% | 19 | 82.52%
10 | 99.90% | 20 | 100 %
Mean | 95.76%
