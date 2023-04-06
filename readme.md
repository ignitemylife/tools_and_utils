# this is a tools repository developed by Kenneth
## install
    cd $code_dir
    python setup.py install
or

    cd $code_dir
    pip install .

**tips**: 

1. help print
   
    
    python -m ken_tools -h    
       
2. **ken** means what?
    
    ken means Kenneth, the author of this repo, contact me with phone num: 13018027867 or wechat
    
## packages
### 1. tools
    [1] deep model train
        >> torchrun --nproc_per_node 4 -m ken_tools.tools.train --config your_config.py
    [2] xgb train
        >> python -m ken_tools.tools.xgb_trainer.xgb_clf -h
    
### 2. deep learning
    - trainValBase 深度模型训练的核心实现
 **more details refer to [doc](./docs/deep_learning.md)**
 
### 3. utils
#### 3.1 data
    mxrecord, etc.
#### 3.2 deep_learning
    warmupPolicy, etc.
#### 3.3 visulization 
    tsne, html
    
### 4. web demo
    using flask and tornado to get http server
    
        


