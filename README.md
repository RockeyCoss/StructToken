# StructToken
The official implementation of the paper "[ StructToken : Rethinking Semantic Segmentation with Structural Prior ](https://arxiv.org/abs/2203.12612)".

## News
(2022/06/27) Add README.md

## Abstract

In this paper, we present structure token (StructToken), a new paradigm for semantic segmentation. From a perspective on semantic segmentation as per-pixel classification, the previous deep learning-based methods learn the per-pixel representation first through an encoder and a decoder head and then classify each pixel representation to a specific category to obtain the semantic masks. Differently, we propose a structure-aware algorithm that takes structural information as prior to predict semantic masks directly without per-pixel classification. Specifically, given an input image, the learnable structure token interacts with the image representations to reason the final semantic masks. Three interaction approaches are explored and the results not only outperform the state-of-the-art methods but also contain more structural information. Experiments are conducted on three widely used datasets including ADE20k, Cityscapes, and COCO-Stuff 10K. We hope that structure token could serve as an alternative for semantic segmentation and inspire future research. 

## Method
![屏幕截图 2022-06-27 180129](https://user-images.githubusercontent.com/41846794/175916972-2a696c52-9cea-48ac-8d4a-81c5742a0316.png)


## Catalog

- [ ] Checkpoints
- [ ] Code
- [x] Initialization

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{lin2022structtoken,
  title={StructToken: Rethinking Semantic Segmentation with Structural Prior},
  author={Lin, Fangjian and Liang, Zhanhao and He, Junjun and Zheng, Miao and Tian, Shengwei and Chen, Kai},
  journal={arXiv preprint arXiv:2203.12612},
  year={2022}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE.md) file.
