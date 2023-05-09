# GMRec

This is the PyTorch implementation by [@HelloElwin](https://github.com/HelloElwin) for **GMRec** proposed in the paper [*Graph Masked Autoencoder for Sequential Recommendation*](https://arxiv.org/abs/2305.04619) published in SIGIR'23 by [Yaowen Ye](https://helloelwin.github.io/), [Lianghao Xia](https://akaxlh.github.io/), and [Chao Huang](https://sites.google.com/view/chaoh).

<img width="1362" alt="model" src="https://user-images.githubusercontent.com/40925586/236808551-aaf34e77-8e97-4043-8c6b-e83dd5fd943b.png">

### 1. Introduction

GMRec is a simple yet effective graph masked autoencoder that adaptively and dynamically distills global item transitional information for self-supervised augmentation through a novel **adaptive transition path masking** strategy. It naturally addresses the data scarcity and noise perturbation problems in sequential recommendation scenarios and avoids issues in most contrastive learning-based methods.

### 2. Environment

We suggest the following environment for running the model:

```
python==3.8.13
pytorch==1.12.1
numpy==1.18.1
```

### 3. How to run

Please first unzip the desired dataset in the dataset folder, and then run

- Amazon Books: `python main.py --data books`
- Amazon Toys: `python main.py --data toys`
- Retailrocket: `python main.py --data retailrocket`

More explanation of model hyper-parameters can be found [here](./params.py).

### 4. Citing our paper

```
@inproceedings{ye2023graph,
  title={Graph Masked Autoencoder for Sequential Recommendation},
  author={Ye, Yaowen and Xia, Lianghao and Huang, Chao},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'23), July 23-27, 2023, Taipei, Taiwa},
  year={2023}
}
```
