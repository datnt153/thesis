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
python train_3view.py --modelname tf_efficientnetv2_s_in21k --img_size 256 --batch_size 16  --use_log --use_wandb
python train_3view.py --modelname tf_efficientnetv2_m_in21k --img_size 256 --batch_size 16  --use_log --use_wandb
python train_3view.py --modelname tf_efficientnetv2_l_in21k --img_size 256 --batch_size 8  --use_log --use_wandb
