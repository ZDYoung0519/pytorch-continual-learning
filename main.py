import time
import os
import os.path as osp
import numpy as np
import random
import torch
import argparse
from pyil.config import Config, DictAction, replace_cfg_vals
from pyil.builder import *
from pyil.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Incremental Learning')

    parser.add_argument('--config', default='./configs/ewc.py', metavar="FILE", help="path to config file", type=str)
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true',help='resume from the latest checkpoint automatically')
    parser.add_argument('--seed', type=int, default=519, help='random seed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # gpu id, only support single gpu
    if args.gpu_id is not None:
        cfg.gpu_id = args.gpu_id

    # create work_dir
    mkdir(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = torch.device(f'cuda:{cfg.gpu_id}')
    # set random seeds
    seed = args.seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed

    model = build_model(cfg.model)
    datasets = build_dataset(cfg.data)
    # optimizer = build_optimizer(cfg.optimizer)
    # exempler = None

    learner_args = dict(
        cfg=cfg,
        model=model,
        datasets=datasets,
        logger=logger
    )
    learner_args = dict(learner_args, **cfg.learner)
    learner = build_leaner(learner_args)
    learner.train()

if __name__ == "__main__":
    main()
