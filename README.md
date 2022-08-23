# OTE-MTL

**OTE-MTL** - **M**ulti-**T**ask **L**earning for **O**pinion **T**riplet **E**xtraction
* Code and preprocessed dataset for [EMNLP 2020](https://2020.emnlp.org/) paper titled "[A Multi-task Learning Framework for Opinion Triplet Extraction](https://arxiv.org/abs/2010.01512)" 
* [Chen Zhang](https://genezc.github.io), [Qiuchi Li](https://qiuchili.github.io), [Dawei Song](http://cs.bit.edu.cn/szdw/jsml/js/sdw/index.htm) and [Benyou Wang](https://wabyking.github.io/old).

## Updates

* Feb. 20th, 2021. As is pointed out in our paper, we have noted that [datav1](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V1-AAAI2020) used in https://arxiv.org/abs/1911.01616 is rather incomplete and have corrected their mistakes. That is, the data used for our experiments is similar to [datav2](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020). However, as is requested by some users and in case of any inconsistencies between our data and datav2, we decide to support the test of our model on datav2. You could just run our model on datav2 with just an additional argument *--v2*.

## Requirements

* Python 3.6
* PyTorch 1.0.0
* numpy 1.15.4

## Usage

* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.
* Train with command, optional arguments could be found in [train.py](/train.py), **--v2** denotes whether test on datav2
```bash
python train.py --model mtl --dataset rest14 [--v2]
```
* Infer with [infer.py](/infer.py)

## Task

An overview of the task opinion triplet extraction (OTE) is given below

![model](/assets/task.png)

 OTE is solving the same task proposed in https://arxiv.org/abs/1911.01616. While our work focuses on extracting (aspect term, opinion term, sentiment) opinion triplets (OTs), they extract (aspect term-sentiment pair, opinion term)s. Owing to the minor difference lying in formulations, two drawbacks in the latter formulation are presented: (i) sentiments are determined without accessing opinion terms, (ii) conflicting opinions expressed towards an aspect cannot be predicted.

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper

```bibtex
@inproceedings{zhang-etal-2020-multi,
    title = "A Multi-task Learning Framework for Opinion Triplet Extraction",
    author = "Zhang, Chen  and
      Li, Qiuchi  and
      Song, Dawei  and
      Wang, Benyou",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.72",
    pages = "819--828",
}
```

## Contact

* For any issues or suggestions about this work, don't hesitate to create an issue or directly contact me via [gene_zhangchen@163.com](mailto:gene_zhangchen@163.com) !
