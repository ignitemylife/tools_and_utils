import torch
import torch.distributed as dist
from contextlib import contextmanager
from datetime import datetime
import time

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()


def get_timestamp():
    if not dist.is_available() or not dist.is_initialized():
        return datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

    if dist.get_rank() in [-1, 0]:
        unixstamp = torch.tensor([time.time()], device='cuda')
        dist.broadcast(unixstamp, 0)
    else:
        unixstamp = torch.tensor([0.], device='cuda')
        dist.broadcast(unixstamp, 0)

    timestamp = datetime.fromtimestamp(unixstamp.item()).strftime('%Y%m%d_%H%M%S')
    return timestamp


def evenly_divisible_all_gather(data: torch.Tensor):
    """
    Utility function for distributed data parallel to pad tensor to make it evenly divisible for all_gather.
    Args:
        data: source tensor to pad and execute all_gather in distributed data parallel.

    """
    world_size = dist.get_world_size()
    if world_size <= 1:
        return data

    device = data.device
    # make sure the data is evenly-divisible on multi-GPUs
    length = data.shape[0]
    length = torch.tensor(length, dtype=torch.int32, device=device)
    all_lens = [torch.tensor(0, dtype=torch.int32, device=device) for _ in range(world_size)]
    dist.all_gather(all_lens, length)
    max_len = max(all_lens).item()
    if length < max_len:
        size = [max_len - length.item()] + list(data.shape[1:])
        data = torch.cat([data, data.new_full(size, float("NaN"))], dim=0)
    else:
        size = data.shape

    datum = [torch.zeros(size, dtype=data.dtype, device=device) for _ in range(world_size)]
    # all gather across all processes
    dist.all_gather(datum, data)
    # delete the padding NaN items

    ret = []
    for d, l in zip(datum, all_lens):
        ret.append(d[:l])

    return torch.cat(ret, dim=0)