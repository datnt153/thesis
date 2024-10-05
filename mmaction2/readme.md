# I. Install mmaction2

### 1. Clone and Install mmaction2

```bash
# Clone mmaction2 repository
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2

# Install mmaction2 and its dependencies
pip install -r requirements/build.txt
pip install -v -e .

# Install other necessary dependencies
mim install "mmaction2>=0.24.0"
```

---

# II. Data Preparation

- Merge all frames into 2-second video clips. The resulting videos and their labels should be placed in the `data/video` folder.

---

# III. Training

1. After installing **mmaction2**, copy all the required configuration files to the **mmaction2** folder.

2. Run the following commands to start the training process:

   ```bash
   conda activate mmaction2
   bash run_thesis
   ```

---

# IV. Logs and Results

## Dash view

### Dashboard View

| model    | acc   | config                                                                                                           | log                                                                                                  |
| :------: | :---: | :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
| TSN      | 0.7516 | [config](configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_Dashboard.py) | [log](models/tsn/Dashboard/20240704_235148/20240704_235148.log) |
| C3D      | 0.7895 | [config](configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb-Dashboard.py) | [log](models/c3d/Dashboard/20240709_015303/20240709_015303.log) |
| TAM      | 0.8381 | [config](configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb-Dashboard.py) | [log](models/tam/Dashboard/20240705_090050/20240705_090050.log) |
| SlowFast | 0.8449 | [config](configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb-Dashboard.py) | [log](models/slowfast/Dashboard/20240714_004645/20240714_004645.log) |

## Rear view

| model    | acc   | config                                                                                                           | log                                                                                                  |
| :------: | :---: | :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
| TSN      | 0.7472 | [config](configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Rear_view.py) | [log](models/tsn/Rear_view/20240705_002951/20240705_002951.log) |
| C3D      | 0.7098 | [config](configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb-Rear_view.py) | [log](models/c3d/Rear_view/20240706_155112/20240706_155112.log) |
| TAM      | 0.8177 | [config](configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb-Rear_view.py) | [log](models/tam/Rear_view/20240705_223345/20240705_223345.log) |
| SlowFast | 0.824  | [config](configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb-Rear_view.py) | [log](models/slowfast/Rear_view/20240714_033755/20240714_033755.log) |


## Right side view

### Right Side Window View

| model    | acc   | config                                                                                                           | log                                                                                                  |
| :------: | :---: | :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
| TSN      | 0.6981 | [config](configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Righ_side_window.py) | [log](models/tsn/Right_side_window/20240705_010744/20240705_010744.log) |
| C3D      | 0.6898 | [config](configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb-Right_side_window.py) | [log](models/c3d/Right_side_window/20240705_213840/20240705_213840.log) |
| TAM      | 0.7511 | [config](configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb-Right_side_window.py) | [log](models/tam/Right_side_window/20240706_011235/20240706_011235.log) |
| SlowFast | 0.807  | [config](configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb-Right_side_window.py) | [log](models/slowfast/Right_side_window/20240714_062018/20240714_062018.log) |

Due to the large size of the model, only logs are submitted. For more detailed information, refer to the logs provided in this link: [mmaction2 Logs](https://drive.google.com/file/d/1qltb7iWwjwrTkwuXJ_9BESFSIoK4Y6ni/view?usp=sharing).
