<div align=center>
    <h1>Lightning Baseline</h1>
    <strong>Pytorch Lightning Baseline</strong>
    <br>
    <br>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white">
    <img src="https://img.shields.io/badge/lightning-792EE5?style=flat-square&logo=lightning&logoColor=white">
    <img src="https://img.shields.io/badge/weights&biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black">
    <img src="https://img.shields.io/badge/poetry-60A5FA?style=flat-square&logo=poetry&logoColor=white">
</div>

## 모델
**Encoder**

* InternImage [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_InternImage_Exploring_Large-Scale_Vision_Foundation_Models_With_Deformable_Convolutions_CVPR_2023_paper.pdf)]

* Swin Transformer [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)]

**Decoder**

* FPN [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)]

* UPerNet [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.pdf)]

* PAN [[paper](https://arxiv.org/pdf/1805.10180.pdf)]

## 설정
    conda create -n torch python=3.9
    conda activate torch
    poetry install (or update)
## 실행
    poetry run python train.py (or bash run.sh)

## 구조

    .
    ├── README.md
    ├── checkpoints
    ├── configs
    │   ├── augmentation
    │   │   └── base.yaml
    │   ├── base_train.yaml
    │   ├── custom_train.yaml
    │   ├── model
    │   │   ├── hrnet-unet.yaml
    │   │   ├── internimageb-fpn.yaml
    │   │   ├── internimageb-pan.yaml
    │   │   ├── internimageb-upernet.yaml
    │   │   ├── resnet101-deeplabv3+.yaml
    │   │   ├── resnet101-fpn.yaml
    │   │   ├── swinb-fpn.yaml
    │   │   ├── swinb-pan.yaml
    │   │   └── swinb-upernet.yaml
    │   ├── optimizer
    │   │   ├── adam.yaml
    │   │   └── sgd.yaml
    │   ├── scheduler
    │   │   ├── cosinewarmrestarts.yaml
    │   │   ├── reduceonplateau.yaml
    │   │   └── step.yaml
    │   └── sweep
    ├── convert_model.py
    ├── data
    │   ├── data_module.py
    │   └── new_data_module.py
    ├── inference.py
    ├── models
    │   ├── base_module.py
    │   └── components
    │       ├── fcn.py
    │       ├── fpn.py
    │       ├── internimage.py
    │       ├── swintransformer.py
    │       ├── upernet.py
    │       ├── upernet_intern.py
    │       └── upernet_swin.py
    ├── old_train.py
    ├── poetry.lock
    ├── pyproject.toml
    ├── run.sh
    └── train.py

