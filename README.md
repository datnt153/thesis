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

## 1. Build Dataset from Scratch

### Create the Entire Dataset
- Run the following commands:
  ```bash
  bash create_gt.sh
  python create_data.py
  ```
  This will generate a file named `all.csv`.

### Create K-Folds
To split the data into training and testing sets similar to the **utvm-vtcc team**, use:

```bash
python create_kfolds.py
```

## 2. Download from Drive
Alternatively, you can download the dataset directly. Download `0.zip` and `1-15.zip` from [this Google Drive folder](https://drive.google.com/drive/folders/1w-YFVOhUmVZw8c1JGVf7T5hz_RCoDhbs?usp=sharing) and unzip them.

## Data Structure
The dataset folder should have the following structure:

```text
dataset
├── 0
├── 1
├── 2
├── 3
├── 4
├── 5
├── 6
├── 7
├── 8
├── 9
├── 10
├── 11
├── 12
├── 13
├── 14
└── 15
```

# III. Train the Model
Make sure to update the `data_path` variable to point to your dataset directory. Then run the training script:

```bash
bash run.sh
```

# IV. Validation

1. Navigate to the `infer` directory:
   ```bash
   cd infer
   ```
2. Download the `final_models.zip` file from [this Google Drive folder](https://drive.google.com/drive/folders/1w-YFVOhUmVZw8c1JGVf7T5hz_RCoDhbs) and unzip it.
   
3. The folder structure for the models should look like this:
   
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

4. Run the final inference:
   ```bash
   python final_inference.py
   ```

# Compare Different Methods
You can also experiment with training other algorithms like:
- **SlowFast**
- **C3D**
- **TANet**
- **TSN**

For more details, refer to the `"mmaction2/readme.md"` file.

# References
- [Docker File](https://github.com/tascj/kaggle-hubmap-hacking-the-human-vasculature)
- [Dataset Creation](https://github.com/vtccdivedeeper/2023AICityChallenge-Track3) by **Divedeeper - Team 83**
- [Pose Extraction](https://github.com/open-mmlab/mmaction2) using **mmaction2**