_base_ = ['./datasets/cifar100.py']

# train settings
num_epochs = 200
batch_size = 64
lr = 0.1
momentum = 0.
weight_decay = 0.
lr_decay = 0.1
milestones = [60, 120, 170]
clipgrad = 10000
log_interval = 50
topk = [1, 5]       # use top k accuracy

# incremental settings
init_nc = 10                # initial numbers of classes
incre_nc = 10               # incremental numbers of classes
class_order = None          # specify the class order
shuffle_classes = True      # shuffle class order

# model
model = dict(
    type='resnet18',
    pretrained=True)

# leaner
learner = dict(
    type='ewc',
    lamb=0.1,
    importance='fisher'
)

log_level = 'INFO'
load_from = None
resume_from = None


