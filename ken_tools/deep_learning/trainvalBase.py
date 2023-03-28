import os
import os.path as osp
import torch
import logging

from mmcv.runner import get_dist_info, master_only
from mmcv.utils import get_logger


from pdb import set_trace as st
from torch.utils.tensorboard import SummaryWriter

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
        pass


    def train(self, model, loader, criterion, optimizer, scheduler, warmup=None, amp=True, device='cuda', grad_clip=10):
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

                self.log(epoch=epoch, ind=ind, len_of_loader=len_of_loader, loss=loss.item(), lr=warmup.get_last_lr())
                self.add_to_tb(epoch * len_of_loader + ind, loss=loss.item(), lr=warmup.get_last_lr()[0])

            scheduler.step()  # epoch update
            loader.sampler.set_epoch(epoch)


    def val(self, model, loader, criterion=None, fp16=True, only_rank0=False):
        pass