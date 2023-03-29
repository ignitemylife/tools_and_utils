# this is a tools repository developed by Kenneth
## install
    cd $code_dir
    python setup.py install
## packages
### 1. tools
    [1] deep model train
        >> torchrun --nproc_per_node 4 -m ken_tools.tools.train --config your_config.py
    [2] xgb train
        >> python -m ken_tools.tools.xgb_trainer.xgb_clf -h
    
### 2. deep learning
    - trainValBase 深度模型训练的核心实现
 
### 3. utils
#### 3.1 data
    mxrecord, etc.
#### 3.2 deep_learning
    warmupPolicy, etc.
#### 3.3 visulization 
    tsne, html
    
### 4. web demo
    using flask and tornado to get http server
    
## distributed training 
 - train and val script example
 
    ```
    
    from ken_tools.utils.deep_learning import WarmupPolicy
    from ken_tools.deep_learning import initialize, TrainValBase
    from ken_tools.utils.deep_learning import torch_distributed_zero_first


    if __name__ == "__main__":
        # initialize
        rank, world_size, local_rank, logger, cfg = initialize()
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
      
   - bash cmd
        
    ```
       torchrun --nproc_per_node 8 train.py --config ...
    ```
        


