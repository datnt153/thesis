eval "$(conda shell.bash hook)"
conda activate openmmlab 

# python train.py --modelname tf_efficientnetv2_l_in21k --img_size 256 --batch_size 8 --use_pose --n_epochs 30
# train with pose 
python train.py --modelname tf_efficientnetv2_m_in21k --img_size 256 --batch_size 16 --use_pose
#python train.py  --use_pose 

# train with image 
python train.py --modelname tf_efficientnetv2_m_in21k --img_size 256 --batch_size 16
# python train.py 



