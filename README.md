# OTE-MTL

**OTE-MTL** - **M**ulti-**T**ask **L**earning for **O**pinion **T**riplet **E**xtraction
* Code and preprocessed dataset for [EMNLP 2020 Findings](https://2020.emnlp.org/) paper titled "[A Multi-task Learning Framework for Opinion Triplet Extraction](https://arxiv.org/abs/2010.01512)" 
* [Chen Zhang](https://genezc.github.io), [Qiuchi Li](https://qiuchili.github.io), [Dawei Song](http://cs.bit.edu.cn/szdw/jsml/js/sdw/index.htm) and [Benyou Wang](https://wabyking.github.io/old).

## Requirements

* Python 3.6
* PyTorch 1.0.0
* numpy 1.15.4

## Usage

* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.
* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python train.py --model_name ote --dataset rest14
```
* Infer with [infer.py](/infer.py)

## Task

The state-of-the-art Aspect-based Sentiment Analysis (ABSA) approaches are mainly based on either detecting aspect terms and their corresponding sentiment polarities, or co-extracting aspect and opinion terms. However, the extraction of aspect-sentiment pairs lacks opinion terms as a reference, while co-extraction of aspect and opinion terms would not lead to meaningful pairs without determining their sentiment dependencies. To address the issue, we present a novel view of ABSA as an opinion triplet extraction task, and propose a multi-task learning framework to jointly extract aspect terms and opinion terms, and simultaneously parses sentiment dependencies between them with a biaffine scorer. At inference phase, the extraction of triplets is facilitated by a triplet decoding method based on the above outputs.

An overview of our proposed task is given below

![model](/assets/task.png)

## Citation

If you use the code in your paper, please kindly star this repo and cite our paper

```bibtex
@inproceedings{Zhang2020, 
    title = "A Multi-task Learning Framework for Opinion Triplet Extraction", 
    author = "Zhang, Chen and Li, Qiuchi and Song, Dawei and Benyou Wang", 
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)", 
    year = "2020", 
    publisher = "Association for Computational Linguistics", 
} 
```

## Contact

* For any issues or suggestions about this work, don't hesitate to create an issue or directly contact me via [gene_zhangchen@163.com](mailto:gene_zhangchen@163.com) !