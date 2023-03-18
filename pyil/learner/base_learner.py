from tqdm import tqdm
import glob
import os
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import numpy as np
import warnings
from pyil.datasets import IDataset
from pyil.metrics import topk_accuracy
from torch.utils.data import DataLoader


class BaseLeaner(object):
    def __init__(self, cfg, model, datasets, logger,
                 num_epochs=200, lr=0.05, weight_decay=0., momentum=0., lr_decay=0.1, milestones=[60, 120, 170], clipgrad=10000):
        self.cfg = cfg
        self.model = model
        self.datasets = datasets
        self.logger = logger
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.milestones = milestones
        self.clipgrad = clipgrad

        # resume
        resume_from = None
        if self.cfg.resume_from is None and self.cfg.get('auto_resume'):
            resume_from = self.search_latest_checkpoint(self.cfg.work_dir)
        if resume_from is not None:
            self.cfg.resume_from = resume_from
        if self.cfg.resume_from:
            self.resume(self.cfg.resume_from)

        # init incremental tasks
        self.init_tasks()

    def train(self):
        while self.cur_t < self.num_tasks:
            self.before_train()
            self.train_task_t()
            self.after_train()
            self.eval_task_t()
            self.save()
            self.cur_t += 1

    def before_train(self):
        # get optimizer and scheduler
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=self.milestones,
            gamma=self.lr_decay
        )

        # get dataset and dataloader for task t
        self.train_dataloader = self.all_dataloaders[self.cur_t]
        self.test_dataloader = self.all_dataloaders[self.cur_t]
        # expand the last layer
        self.update_model_fc()

    def after_train(self):
        # TODO: update the learner
        pass

    def train_task_t(self):
        self.logger.info('Start training task: {}'.format(self.cur_t))
        self.model.train()
        device = self.cfg.device
        self.model.to(device)
        # train
        for epoch in range(self.cfg.num_epochs):
            train_loss, train_acc, train_count = 0., 0., 0.
            for bnd, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(device).float(), labels.to(device).long()
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                self.optimizer.step()
                with torch.no_grad():
                    pred = outputs.argmax(axis=1)
                    train_loss += loss.item() * len(labels)
                    train_acc += (pred == labels).sum().item()
                    train_count += len(labels)
            self.scheduler.step()

            train_loss /= train_count
            train_acc /= train_count
            if epoch % self.cfg.log_interval == 0:
                self.logger.info('[Epoch: {}/{}], train loss: {:.5f}, train acc: {:.5f}'.format(
                    epoch, self.cfg.num_epochs, train_loss, train_acc))
        return train_loss, train_acc

    @torch.no_grad()
    def eval_task_t(self):
        self.model.eval()
        device = self.cfg.device
        self.model.to(device)

        # a_t_k
        for j, loader in enumerate(self.all_dataloaders):
            if j > self.cur_t:
                break
            y_ture, y_pred = [], []
            for bnd, (images, labels) in enumerate(loader):
                images, labels = images.to(device).float(), labels.to(device).long()
                outputs = self.model(images)
                predictions = torch.topk(outputs, k=max(self.topk), dim=1)
                y_pred.extend(predictions.detach().cpu().numpy())
                y_ture.extend(labels.detach().cpu().numpy())
            y_ture = np.array(y_ture)
            y_pred = np.array(y_pred)

            for topk in self.topk:
                self.metircs[topk]['acc_m'][self.cur_t, j] = topk_accuracy(y_ture, y_pred, topk)

        for topk in self.topk:
            acc_m = self.metircs[topk]['acc_m']
            # acc: average accuracy
            acc = acc_m[self.cur_t, :self.cur_t+1].mean()
            # bwt: backward transfer
            if self.cur_t == 0:
                bwt = np.nan
            else:
                bwt = acc[self.cur_t, : self.cur_t] - np.diagonal(acc)[:self.cur_t]
                bwt = bwt.mean()

            self.metircs[topk]['acc'][self.cur_t] = acc
            self.metircs[topk]['bwt'][self.cur_t] = bwt

        for topk in self.topk:
            self.print_metircs(self.metircs[topk])

    def print_metircs(self, metircs, fmt='{:^10}'):
        s = ''
        s += '\n' +'|' + fmt.format('') + '|'
        # header:
        for i in range(self.num_tasks):
            s += fmt.format('Task {}'.format(i)) + '|'
        s += fmt.format('ACC') + '|'
        s += fmt.format('BWT') + '|'
        s += '\n'
        for i in range(self.num_tasks):
            s += '|' + fmt.format('Session {}'.format(i)) + '|'
            for j in range(self.num_tasks):
                s += fmt.format(metircs['acc_m'][i, j]) + '|'
            s += fmt.format(metircs['acc'][i]) + '|'
            s += fmt.format(metircs['bwt'][i]) + '|'
            s += '\n'
        self.logger.info(s)

    def get_task_data(self, t):
        train_dataset, test_dataset = self.datasets
        offset = 0 if t == 0 else self.init_nc + (t - 1) * self.incre_nc
        task = self.tasks[t]
        train_set = IDataset(train_dataset, tgt_classes=task, offset=offset)
        test_set = IDataset(test_dataset, tgt_classes=task, offset=offset)
        return train_set, test_set

    def init_tasks(self):
        self.logger.info('Initializing tasks for CIL...')
        self.init_nc, self.incre_nc = self.cfg.init_nc, self.cfg.incre_nc
        self.all_classes = np.unique(self.datasets[0].targets)
        self.nc = len(self.all_classes)

        # class order
        class_order = self.cfg.get('class_order', None) or getattr(self, 'class_order', None)
        shuffle_classes = self.cfg.get('shuffle_classes', False)
        if class_order is not None:
            self.class_order = class_order
        else:
            self.class_order = list(range(len(self.all_classes)))
        # shuffle
        if shuffle_classes:
            np.random.shuffle(self.class_order)

        # generate incremental tasks
        tasks = []
        tasks.append(self.class_order[: self.init_nc])
        t = self.init_nc
        while t < self.nc:
            if t + self.incre_nc < self.nc:
                tasks.append(self.class_order[t: t + self.incre_nc])
            else:
                tasks.append(self.class_order[t:])
                break
            t += self.incre_nc

        assert self.nc == len(np.unique(tasks))
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.cur_t = 0

        # get datasets and dataloaders for all tasks
        datasets = []
        dataloaders = []
        for t, task in enumerate(tasks):
            train_set, test_set = self.get_task_data(t)
            train_loader = DataLoader(
                train_set, batch_size=self.cfg.batch_size, shuffle=True)
            test_loader = DataLoader(
                test_set, batch_size=self.cfg.batch_size, shuffle=False)
            datasets.append((train_set, test_set))
            dataloaders.append((train_loader, test_loader))
        self.all_datasets = datasets
        self.all_dataloaders = dataloaders

        # for evaluation
        self.topk = self.cfg.topk
        if isinstance(self.topk, int):
            self.topk = [self.topk]

        self.metircs = {}
        for topk in self.topk:
            self.metircs[topk] = {}
            acc_m = np.zeros([len(tasks), len(tasks)])
            acc = np.zeros([len(tasks)])
            bwt = np.zeros([len(tasks)])
            fwt = np.zeros([len(tasks)])
            self.metircs[topk] = {
                'acc_m': acc_m,
                'acc': acc,
                'bwt': bwt,
                'fwt': fwt
            }

    def search_latest_checkpoint(self, suffix='pth'):
        model_path = os.path.join(self.cfg.work_dir, '/checkpoints/')
        if not os.path.exists(model_path):
            warnings.warn('The path of checkpoints does not exist.')
        checkpoints = glob.glob(os.path.join(model_path, f'*.{suffix}'))

        if len(checkpoints) == 0:
            warnings.warn('There are no checkpoints in the path.')
            return None

        latest = -1
        latest_path = None
        for checkpoint in checkpoints:
            count = int(os.path.basename(checkpoint).split('_')[-1].split('.')[0])
            if count > latest:
                latest = count
                latest_path = checkpoint
        return latest_path

    def resume(self, checkpoint, resume_optimizer=True):
        # model
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            checkpoint = torch.load(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model'])

        # optimizer
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed task %d', self.cur_t)

    def save(self):
        model_state = {
            'model': self.model._state_dict(),
            'optimizer': self.optimizer,
            'tasks': self.tasks
        }
        work_dir = self.cfg.work_dir

    def update_model_fc(self):
        # TODO: debug while using mlp
        use_mpl = isinstance(self.model.fc, nn.Sequential)

        if use_mpl:
            last_layer = self.model.fc[-1]
        else:
            last_layer = self.model.fc

        in_features = last_layer.in_features
        out_features = last_layer.out_features

        if self.cur_t == 0:
            new_layer = nn.Linear(in_features, self.init_nc)
        else:
            pre_weight = last_layer.weight
            new_layer = nn.Linear(in_features, out_features + self.incre_nc)
            new_layer.weight[:out_features, :] = pre_weight

        if use_mpl:
            self.model.fc[-1] = new_layer
        else:
            self.model.fc = new_layer

    def criterion(self, outputs, targets):
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss






