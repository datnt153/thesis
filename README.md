# Cteate enviroment

conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmengine
mim install mmcv


# Create all dataset 
- bash create_gt.sh
- python create_data.py
=> all.csv

# Create k-folds
- python create_kfolds.py

hien tai thi creat theo id cua khong gian mang :(.


Dowload dataset in :
https://drive.google.com/drive/folders/1w-YFVOhUmVZw8c1JGVf7T5hz_RCoDhbs?usp=sharing
