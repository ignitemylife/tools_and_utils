from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, RandomResizedCrop, RandomHorizontalFlip, RandomAdjustSharpness, Compose, \
    Normalize

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

from ken_tools.utils.deep_learning import WarmupPolicy
from ken_tools.deep_learning import initialize, TrainValBase
from ken_tools.utils.deep_learning import torch_distributed_zero_first


if __name__ == "__main__":
    # initialize
    rank, world_size, local_rank, logger, cfg = initialize(is_dist=True)
    print(rank, world_size, local_rank)

    # model
    model = resnet18()
    model = model.to(rank % torch.cuda.device_count())
    model = DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])


    # data
    transforms = Compose([
        ToTensor(),
        RandomAdjustSharpness(sharpness_factor=0.6, p=0.5),
        RandomHorizontalFlip(),
        RandomResizedCrop(size=(224, 224), scale=(0.6, 1)),
        Normalize(128., 128.)
    ])
    with torch_distributed_zero_first(rank):
        dataset = CIFAR10('./data/cifar10', download=True, transform=transforms)
    sampler = DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, num_workers=4, batch_size=128)


    # optim
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = MultiStepLR(optimizer, [6, 9])
    warmup = WarmupPolicy(scheduler, warmup_factor=0.1, warmup_iters=20)


    # train
    trainVal = TrainValBase(epoches=10, work_dir=cfg.work_dir, logger=logger)
    trainVal.train(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup=warmup,
    )

    # val
    trainVal.val(model, loader)