# TSN
# python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_Dashboard.py
# python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Rear_view.py
# python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Righ_side_window.py

# C3D
#python tools/train.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb-Dashboard.py
# python tools/train.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb-Rear_view.py
#python tools/train.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb-Right_side_window.py

#TAM
#python tools/train.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb-Dashboard.py
#python tools/train.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb-Rear_view.py
#python tools/train.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb-Right_side_window.py

# slowfast
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb-Dashboard.py
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb-Rear_view.py
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb-Right_side_window.py
