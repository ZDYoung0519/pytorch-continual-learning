data = dict(
    train=dict(
        type='cifar100',
        dataset_dir='./data',
        transforms=[
            dict(type='Pad', padding=4),
            dict(type='Resize', size=32),
            dict(type='RandomHorizontalFlip', p=0.5),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))
        ]
    ),
    test=dict(
        type='cifar100',
        dataset_dir='./data',
        transforms=[
            # dict(type='Resize', size=32),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023))
        ]
    )
)