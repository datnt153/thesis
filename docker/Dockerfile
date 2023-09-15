FROM nvcr.io/nvidia/pytorch:23.05-py3

ENV DEBIAN_FRONTEND=noninteractive


RUN python -m pip install -U pip
RUN pip install openmim
RUN pip install mmpretrain==1.0.0rc8
RUN mim install mmcv==2.0.1
RUN mim install mmdet==3.1.0
RUN mim install mmpose==1.1.0
RUN mim install mmaction2
RUN pip install timm 
RUN pip install -U albumentations --no-binary qudida,albumentations
RUN pip install wandb
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install opencv-python==4.8.0.74 