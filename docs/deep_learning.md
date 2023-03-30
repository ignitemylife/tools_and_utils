this repo can help you train and val your model quickly

# training and val
 ## 1.distributed train and val
    ```
    
    from ken_tools.utils.deep_learning import WarmupPolicy
    from ken_tools.deep_learning import initialize, TrainValBase
    from ken_tools.utils.deep_learning import torch_distributed_zero_first


    if __name__ == "__main__":
        # initialize
        rank, world_size, local_rank, logger, cfg = initialize(is_dist=True)
        print(rank, world_size, local_rank)

        # model
        model = ...
        model = model.to(rank % torch.cuda.device_count())
        model = DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])


        # data
        transforms = ...
        with torch_distributed_zero_first(rank):
            dataset = ...
        sampler = DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, num_workers=4, batch_size=128)


        # optim
        criterion = ...
        optimizer = ...
        scheduler = ...
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
        ```
        
run cmd like this:
> torchrun --nproc_per_node 8 train.py --config ...
    
   
## 2. not using distributed
   ```
    initialize(is_dist=False)
   
    # model
    ...
   
    # data
    ...
   
    # optim
    ...
   
    # train and val
    ...
  ```

run cmd like this:
>python train.py ...
