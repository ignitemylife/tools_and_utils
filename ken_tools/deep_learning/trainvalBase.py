import os
import os.path as osp
import torch
import logging
import numpy as np
from pdb import set_trace as st
from torch.utils.tensorboard import SummaryWriter

from mmcv.runner import get_dist_info, master_only
from mmcv.utils import get_logger

from ken_tools.utils.eval_utils import ClsEval
from ken_tools.utils.meters import AverageMeter, ExpMeter, ProgressMeter


class TrainValBase():
    def __init__(self, epoches=-1, work_dir=None, save_interval=1, print_interval=10, cfg=None, logger=None):
        rank, world_size = get_dist_info()

        self.epoches = epoches
        self.work_dir = work_dir

        # tensorboard
        if rank == 0 and work_dir is not None:
            self.tb_dir = osp.join(work_dir, 'tb_dir')
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(self.tb_dir)
        else:
            self.tb_dir = None
            self.tb_writer = None

        self.save_interval = save_interval
        self.print_interval = print_interval
        self.scalar = torch.cuda.amp.GradScaler()

        self.logger = logger if logger is None else get_logger('TrainVal', log_level=logging.INFO if rank==0 else logging.ERROR)

        self.cfg = cfg # reserved


    @master_only
    def log(self, *args, **kwargs):
        info = ''
        for k, v in kwargs.items():
            info += f'{k}:{v} '
        self.logger.info(info)


    @master_only
    def add_to_tb(self, step, **kwargs):
        loss = kwargs.get('loss', -1)
        lr = kwargs.get('lr', -1)
        self.tb_writer.add_scalar('loss', loss, step)
        self.tb_writer.add_scalar('lr', lr, step)


    def init_meters(self):
        avgExample = AverageMeter('example')
        expExample = ExpMeter('example')
        return [avgExample, expExample]


    def train(self, model, loader, criterion, optimizer, scheduler, warmup=None, amp=True, device='cuda', grad_clip=10):
        rank, _ = get_dist_info()

        model.train()
        # meters = self.init_meters()
        for epoch in range(max(1, self.epoches)):
            len_of_loader = len(loader)
            for ind, (data, label) in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=amp):

                    data = data.to(device)
                    label = label.to(device)
                    out = model(data)
                    loss = criterion(out, label)

                self.scalar.scale(loss).backward()

                # grad clip
                if grad_clip > 0:
                    self.scalar.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                # optimizer.step()
                self.scalar.step(optimizer)
                self.scalar.update()
                optimizer.zero_grad(set_to_none=True)
                warmup.step()   # iter update

                if ind % self.print_interval == 0:
                    self.log(epoch=epoch, ind=ind, len_of_loader=len_of_loader, loss=loss.item(), lr=warmup.get_last_lr())

                self.add_to_tb(epoch * len_of_loader + ind, loss=loss.item(), lr=warmup.get_last_lr()[0])

            scheduler.step()  # epoch update
            loader.sampler.set_epoch(epoch)

            if rank == 0 and epoch % self.save_interval == 0: # important, multi processes write one file simultaneously may corupt the file
                torch.save(model.module.state_dict(), osp.join(self.work_dir, f'epoch_{epoch}.pth'))
                self.logger.info(f'saved ckpts of {epoch}')


    def val(self, model, loader, criterion=None, fp16=True, device='cuda', only_rank0=True):
        self.logger.info('begin to validate...')

        if only_rank0:
            rank, _ = get_dist_info()
            if rank != 0:
                return

        model.eval()
        if fp16:
            model.half()

        len_of_loader = len(loader)
        labels = []
        preds = []
        for ind, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                if fp16:
                    data = data.half()
                out = model(data)
                pred = torch.softmax(out, dim=-1)

            self.logger.info(f'Test {ind}/{len_of_loader}')

            labels.append(label.cpu().numpy())
            preds.append(pred.cpu().numpy().astype(np.float32))

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)

        acc = ClsEval.accuracy(labels, preds)
        self.logger.info(f'acc is {acc:.4f}')