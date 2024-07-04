# I. Installation

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmengine
mim install mmcv
```

# II. Data Preparation
2 Optional:
## 1. Build dataset from scratch 
### Create all dataset 
- bash create_gt.sh
- python create_data.py
=> all.csv

### Create k-folds
Split train-test like utvm-vtcc team

- python create_kfolds.py


## 2. Download from driver
Download dataset file `0.zip` and `1-15.zip` in [driver](https://drive.google.com/drive/folders/1w-YFVOhUmVZw8c1JGVf7T5hz_RCoDhbs?usp=sharing), and then unzip all the files.

## Data structure
```text
    dataset
    ├── 0
    ├── 1
    ├── 10
    ├── 11
    ├── 12
    ├── 13
    ├── 14
    ├── 15
    ├── 2
    ├── 3
    ├── 4
    ├── 5
    ├── 6
    ├── 7
    ├── 8
    └── 9

```


# III. Train model
Change `data_path` to the path of the dataset.

```
bash run.sh

```

# IV. Validate 
```angular2html
cd infer
```
- Download file 'final_models.zip' in link [driver](https://drive.google.com/drive/folders/1w-YFVOhUmVZw8c1JGVf7T5hz_RCoDhbs) and `unzip` this
- Model folder structure like:
```text
    final_models
    ├── tf_efficientnetv2_l_in21k imgs 256 bs 16
    │   ├── 2023_09_14_09.48
    │   └── 2023_09_14_17.18
    ├── tf_efficientnetv2_m_in21k imgs 256 bs 48
    │   ├── 2023_09_09_16.11
    │   └── 2023_09_09_19.32
    └── tf_efficientnetv2_s_in21k imgs 256 bs 48
        ├── 2023_09_13_17.46
        └── 2023_09_13_20.17

```
- Run code: 
```bash 
python final_inference.py
```

**Reference**: 
- Build docker file: https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature
- Create dataset:  [Divedeeper - Team 83](https://github.com/vtccdivedeeper/2023AICityChallenge-Track3 )
- Extract pose: [mmaction2](https://github.com/open-mmlab/mmaction2)
