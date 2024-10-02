ann_file_test = 'data/video/Dashboard_val.txt'
ann_file_train = 'data/video/Dashboard_train.txt'
ann_file_val = 'data/video/Dashboard_val.txt'
auto_scale_lr = dict(base_batch_size=64, enable=False)
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
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb_20220919-a34346bc.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        depth=50,
        num_segments=8,
        pretrained='torchvision://resnet50',
        tam_cfg=None,
        type='TANet'),
    cls_head=dict(
        average_clips='prob',
        consensus=dict(dim=1, type='AvgConsensus'),
        dropout_ratio=0.5,
        in_channels=2048,
        init_std=0.001,
        num_classes=16,
        spatial_type='avg',
        type='TSMHead'),
    data_preprocessor=dict(
        format_shape='NCHW',
        mean=[
            123.675,
            116.28,
            103.5,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer2D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=20, norm_type=2),
    constructor='TSMOptimWrapperConstructor',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(fc_lr5=True))
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
        ann_file='data/video/Dashboard_val.txt',
        data_prefix=dict(video='data'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True,
                type='DenseSampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=256, type='ThreeCrop'),
            dict(input_format='NCHW', type='FormatShape'),
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
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
        type='DenseSampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=256, type='ThreeCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=30, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=12,
    dataset=dict(
        ann_file='data/video/Dashboard_train.txt',
        data_prefix=dict(video='data'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                type='DenseSampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                input_size=224,
                max_wh_scale_gap=1,
                num_fixed_crops=13,
                random_crop=False,
                scales=(
                    1,
                    0.875,
                    0.75,
                    0.66,
                ),
                type='MultiScaleCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=1, frame_interval=1, num_clips=8, type='DenseSampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        input_size=224,
        max_wh_scale_gap=1,
        num_fixed_crops=13,
        random_crop=False,
        scales=(
            1,
            0.875,
            0.75,
            0.66,
        ),
        type='MultiScaleCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/video/Dashboard_val.txt',
        data_prefix=dict(video='data'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=8,
                test_mode=True,
                type='DenseSampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
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
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
        type='DenseSampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
view = 'Dashboard'
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'models/tam/Dashboard'
