# StructToken
The official implementation of the paper "[ StructToken : Rethinking Semantic Segmentation with Structural Prior ](https://arxiv.org/abs/2203.12612)".

***Warning: This repository is still under construction!***

## Abstract

In previous deep-learning-based methods, semantic segmentation has been regarded as a static or dynamic per-pixel classification task, i.e., classify each pixel representation to a specific category. However, these methods only focus on learning better pixel representations or classification kernels while ignoring the structural information of objects, which is critical to human decision-making mechanism. In this paper, we present a new paradigm for semantic segmentation, named structure-aware extraction. Specifically, it generates the segmentation results via the interactions between a set of learnable structure tokens and the image feature, which aims to progressively extract the structural information of each category from the feature. Extensive experiments show that our StructToken outperforms the state-of-the-art on three widely-used benchmarks, including ADE20K, Cityscapes, and COCO-Stuff-10K.

## Method
![图片](https://user-images.githubusercontent.com/41846794/186887467-fd9bbbd4-4660-4f4a-8a28-5232cb2d7a48.png)


## Catalog

- [x] Initialization
- [ ] Code (in process)
- [ ] Checkpoints (in process)

## Data Preparation
Please prepare ADE20K dataset according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Pre-training Sources
Please prepare pretrain checkpoints and put them in `repo_directory/pretrain`  folder according to the [guideline](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter#usage) in MMSegmentation

## Download Checkpoints

You can download our checkpoints using the the download script `tools/download_ckpts.py`. It has the following parameters:

+ --keys: (optional) The keys of the checkpoints that you want to download.
+ --folder: (optional) The directory of the folder that you want to download the checkpoints to. Default to `./checkpoints`.
+ --ckpt-names: (optional) The save names of the checkpoints you want to download. Default to `{key}.pth`.
+ --all: (optional) Download all the checkpoints.

For example, if you want to download the checkpoints with keys `struct-token-cse_vit-b_ade20k` and `struct-token-sse_vit-b_ade20k`, and then save them with save names `cse_vit-b_ade20k.pth` and  `sse_vit-b_ade20k.pth`, you can use:

```bash
python tools/download_ckpts.py --keys struct-token-cse_vit-b_ade20k struct-token-sse_vit-b_ade20k --ckpt-names cse_vit-b_ade20k.pth sse_vit-b_ade20k.pth
```

Then these two checkpoints will be downloaded to `./checkpoints` folder with names `cse_vit-b_ade20k.pth` and  `sse_vit-b_ade20k.pth`.

Another example, if you want to download all checkpoints, then you can use:

```bash
python tools/download_ckpts.py --all
```

Then all checkpoints will be downloaded to `./checkpoints` folder.


## Results and Models

### ADE20K Val

|     Method      | Backbone | Lr Schedule | Crop Size | mIoU(ss) | mIoU(ms) |                            Config                            |         Download Key          |
| :-------------: | :------: | :---------: | :-------: | :------: | :------: | :----------------------------------------------------------: | :---------------------------: |
| StructToken-CSE |  ViT-T   |    160k     |  512x512  |  39.12   |  40.23   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-t_struct-token-cse_ade20k_512x512.py) | struct-token-cse_vit-t_ade20k |
| StructToken-PWE |  ViT-T   |    160k     |  512x512  |  41.87   |  42.99   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-t_struct-token-pwe_ade20k_512x512.py) | struct-token-pwe_vit-t_ade20k |
| StructToken-SSE |  ViT-T   |    160k     |  512x512  |  40.81   |  42.24   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-t_struct-token-sse_ade20k_512x512.py) | struct-token-sse_vit-t_ade20k |
| StructToken-CSE |  ViT-S   |    160k     |  512x512  |  45.86   |  47.44   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-s_struct-token-cse_ade20k_512x512.py) | struct-token-cse_vit-s_ade20k |
| StructToken-PWE |  ViT-S   |    160k     |  512x512  |  47.36   |  48.89   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-s_struct-token-pwe_ade20k_512x512.py) | struct-token-pwe_vit-s_ade20k |
| StructToken-SSE |  ViT-S   |    160k     |  512x512  |  47.11   |  49.07   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-s_struct-token-pwe_ade20k_512x512.py) | struct-token-sse_vit-s_ade20k |
| StructToken-CSE |  ViT-B   |    160k     |  512x512  |  49.51   |  50.87   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-b_struct-token-cse_ade20k_512x512.py) | struct-token-cse_vit-b_ade20k |
| StructToken-PWE |  ViT-B   |    160k     |  512x512  |  50.92   |  51.82   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-b_struct-token-pwe_ade20k_512x512.py) | struct-token-pwe_vit-b_ade20k |
| StructToken-SSE |  ViT-B   |    160k     |  512x512  |  50.72   |  51.85   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-b_struct-token-sse_ade20k_512x512.py) | struct-token-sse_vit-b_ade20k |
| StructToken-PWE |  ViT-L   |    160k     |  640x640  |  52.95   |  54.03   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-l_struct-token-pwe_ade20k_640x640.py) | struct-token-pwe_vit-l_ade20k |
| StructToken-SSE |  ViT-L   |    160k     |  640x640  |  53.04   |  53.95   | [config](https://github.com/RockeyCoss/StructToken/blob/master/struct_token/configs/ade20k/vit-l_struct-token-sse_ade20k_640x640.py) | struct-token-sse_vit-l_ade20k |

## Evaluation
To evaluate a model whose config directory is `path/to/config` and checkpoint directory is `path/to/checkpoint` on a single node with 8 gpus, please run:
```bash
sh tools/dist_test.sh path/to/config path/to/checkpoint 8 --eval mIoU
```

## Training
To train a model whose config directory is `path/to/config` on a single node with 8 gpus, please run:
```bash
sh tools/dist_train.sh path/to/config 8
```

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