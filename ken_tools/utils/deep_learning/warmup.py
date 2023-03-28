import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
# m = torch.optim.lr_scheduler._LRScheduler

class WarmupPolicy():
    def __init__(
        self,
        lr_scheduler,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
    ):

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.lr_scheduler = lr_scheduler
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.iter_step = 0

    def get_lr(self):
        warmup_factor = 1
        if self.lr_scheduler.last_epoch == 0 and  self.iter_step < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.iter_step / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            self.iter_step += 1
        return [base_lr * warmup_factor for base_lr in self.lr_scheduler.base_lrs]

    def step(self):
        if self.lr_scheduler.last_epoch == 0 and self.iter_step <= self.warmup_iters:
            values = self.get_lr()

            for param_group, lr in zip(self.lr_scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
            self.lr_scheduler._last_lr = [group['lr'] for group in self.lr_scheduler.optimizer.param_groups]

    def get_last_lr(self):
        return self.lr_scheduler._last_lr