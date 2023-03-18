data = dict(
    train=dict(
        type='cifar100',
        dataset_dir='./data',
        transforms=[
            dict(type='Pad', padding=4),
            dict(type='Resize', size=32),
            dict(type='RandomHorizontalFlip', p=0.5),
            dict(type='ToTensor'),
            dict(
                type='Normalize',
                mean=(0.5071, 0.4866, 0.4409),
                std=(0.2009, 0.1984, 0.2023))
        ]),
    test=dict(
        type='cifar100',
        dataset_dir='./data',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='Normalize',
                mean=(0.5071, 0.4866, 0.4409),
                std=(0.2009, 0.1984, 0.2023))
        ]))
num_epochs = 200
batch_size = 64
lr = 0.1
momentum = 0.0
weight_decay = 0.0
lr_decay = 0.1
milestones = [60, 120, 170]
clipgrad = 10000
log_interval = 50
topk = [1, 5]
init_nc = 10
incre_nc = 10
class_order = None
shuffle_classes = True
model = dict(type='resnet18', pretrained=True)
learner = dict(type='ewc', lamb=0.1, importance='fisher')
log_level = 'INFO'
load_from = None
resume_from = None
work_dir = './work_dirs/ewc'
gpu_id = 0