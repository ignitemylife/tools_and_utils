import argparse
import os
import os.path as osp
import time
import logging
from cprint import cprint

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, set_random_seed, init_dist
from mmcv.utils import get_logger

from .misc import get_timestamp

_DEVICE = 'cuda'
_IS_DIST = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--config', default='config/config_example.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--ckpt', help='the checkpoint file to resume from or to test')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument('--use-cpu', action='store_true', help='whether to use cpu training')
    parser.add_argument('--distributed', action='store_true', help='whether using distributed parallel training')
    parser.add_argument('--master_port', default=12345,  help='used when training distributed')

    parser.add_argument('--add-infos', nargs='+', default=())
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")

    parser.add_argument('--local_rank', type=int, default=0) # if using torchrun, this will be dropped

    return parser.parse_args()

def initialize():
    ######################## load arguments ########################
    # get args
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # initialize
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.ckpt is not None:
        cfg.ckpt = args.ckpt

    ######################## init dist ########################
    if torch.cuda.device_count() > 0 and _IS_DIST:
        init_dist('pytorch')
        rank, world_size = get_dist_info()
        cprint(f'using distribute training, rank: {rank}, world_size: {world_size}')
    else:
        cprint('using dataprallel')
        rank, world_size = 0, 1

    time.sleep(3)

    ##############   initialize save dir and log   ###################
    # init logger before other steps
    timestamp = get_timestamp()
    _learn = cfg.learn
    prefix = '{}_Model_{}_Lr_{:.8f}_Epochs_{:03d}_Opt_{}_Loss_{}_{}'.format(
        os.path.basename(osp.splitext(args.config)[0]),
        cfg.model.name,
        _learn.optimizer.lr,
        _learn.total_epochs,
        _learn.optimizer.name,
        _learn.loss.name,
        '_'.join(args.add_infos)
    )
    if rank == 0:
        cfg.work_dir = osp.join(cfg.work_dir, prefix + timestamp)
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        cfg.log_level = logging.INFO
    else:
        cfg.log_level = logging.WARNING

    log_file = osp.join(cfg.work_dir, f'rank{rank}_{timestamp}.log')
    logger = get_logger('train.log', log_file=log_file, log_level=cfg.log_level)

    ##############   initialize save dir and log   ###################
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    logger.info(f'training device: {_DEVICE}, is_dist: {_IS_DIST}')
    logger.info(f'Config: {cfg.pretty_text}')
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    local_rank = os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count())
    return rank, world_size, local_rank, logger, cfg





