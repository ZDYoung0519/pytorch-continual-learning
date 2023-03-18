data = dict(
    train=dict(
        type='mnist',
        dataset_dir='./data',
        trasforms=[
            # dict(type='Resize', size=32),
            # dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', mean=[0.1307], std=[0.3081])
        ]
    )
)