eval "$(conda shell.bash hook)"
conda activate openmmlab

# train with pose
#python train.py --modelname tf_efficientnetv2_m_in21k --img_size 256 --batch_size 48 --use_pose --use_log --use_wandb
#python train.py --modelname tf_efficientnetv2_s_in21k --img_size 256 --batch_size 48 --use_pose --use_log --use_wandb
#python train.py --modelname tf_efficientnetv2_l_in21k --img_size 256 --batch_size 16 --use_pose --use_log --use_wandb

# train with image
#python train.py --modelname tf_efficientnetv2_m_in21k --img_size 256 --batch_size 48 --use_log --use_wandb
#python train.py --modelname tf_efficientnetv2_s_in21k --img_size 256 --batch_size 48 --use_log --use_wandb
#python train.py --modelname tf_efficientnetv2_l_in21k --img_size 256 --batch_size 16 --use_log --use_wandb

# train 3 view image
# python train/three_view/train_3view.py --modelname tf_efficientnetv2_s_in21k --img_size 256 --batch_size 16  --use_log --use_wandb
# python train/three_view/train_3view.py --modelname tf_efficientnetv2_m_in21k --img_size 256 --batch_size 16  --use_log --use_wandb
# python train/three_view/train_3view.py --modelname tf_efficientnetv2_l_in21k --img_size 256 --batch_size 8  --use_log --use_wandb

#python train.py --modelname tf_efficientnetv2_l_in21k --img_size 256 --batch_size 16 --use_pose --use_log --use_wandb

## Train alpla
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.25
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.5
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.75

#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.25 --h_dim 32
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.5 --h_dim 32
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.75 --h_dim 32

#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.25 --h_dim 128
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.5 --h_dim 128
#CUDA_VISIBLE_DEVICES=0 python train/alpha_concat/train_alpha.py --img_size 256 --batch_size 12  --use_log --use_wandb --alpha 0.75 --h_dim 128

## Train concat
#CUDA_VISIBLE_DEVICES=0 python train/concat_alpha/train_concat.py --img_size 256 --batch_size 12 --use_log --use_wandb --h_dim 32
#CUDA_VISIBLE_DEVICES=0 python train/concat_alpha/train_concat.py --img_size 256 --batch_size 12 --use_log --use_wandb --h_dim 64
#CUDA_VISIBLE_DEVICES=0 python train/concat_alpha/train_concat.py --img_size 256 --batch_size 12 --use_log --use_wandb --h_dim 128


## train with vgg backbone

#python train.py --modelname vgg16_bn.tv_in1k --img_size 224 --batch_size 16 --use_pose --use_log --use_wandb
#python train.py --modelname vgg16_bn.tv_in1k --img_size 224 --batch_size 48 --use_log --use_wandb


## Train with resnet backbone
CUDA_VISIBLE_DEVICES=0 python train.py --modelname resnet50.a1_in1k --img_size 256 --batch_size 48 --use_pose --use_log --use_wandb
CUDA_VISIBLE_DEVICES=0 python train.py --modelname resnet50.a1_in1k --img_size 256 --batch_size 48 --use_log --use_wandb
