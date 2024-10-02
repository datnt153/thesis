ann_file_test = 'data/video/Rear_view_val.txt'
ann_file_train = 'data/video/Rear_view_train.txt'
ann_file_val = 'data/video/Rear_view_val.txt'
auto_scale_lr = dict(base_batch_size=240, enable=False)
data_root = 'data'
data_root_val = 'data'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        conv_cfg=dict(type='Conv3d'),
        dropout_ratio=0.5,
        init_std=0.005,
        norm_cfg=None,
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',
        style='pytorch',
        type='C3D'),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=4096,
        init_std=0.01,
        num_classes=16,
        spatial_type=None,
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            104,
            117,
            128,
        ],
        std=[
            1,
            1,
            1,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            4,
            8,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/video/Rear_view_val.txt',
        data_prefix=dict(video='data'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=1,
                num_clips=10,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=12,
    dataset=dict(
        ann_file='data/video/Rear_view_train.txt',
        data_prefix=dict(video='data'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(size=112, type='RandomCrop'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=16, frame_interval=1, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(size=112, type='RandomCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=30,
    dataset=dict(
        ann_file='data/video/Rear_view_val.txt',
        data_prefix=dict(video='data'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
view = 'Rear_view'
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'models/c3d/Rear_view'
