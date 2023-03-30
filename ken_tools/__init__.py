from ken_tools.utils.visulization.display_imgs import show_imgs, show_img, is_img, display
from ken_tools.deep_learning import initialize, trainvalBase

import sys
_args = sys.argv[1:]
if _args and _args[0].replace('-', '').lower() in ('h', '--help'):
    print(
        'packages: deep_learning, sundries, tools, utils, etc'
        
        '''
         deep_learning
         ├── __init__.py
         └── trainvalBase.py
         sundries
         ├── __init__.py
         ├── clip2gif.py
         ├── collect_file_list.py
         ├── crawl_imgs.py
         ├── multi_youtube_dl.py
         └── split_train_val.py
         tools
         ├── __init__.py
         ├── demo_config
         ├── display.py
         ├── train.py
         └── xgb_trainer
         utils
         ├── __init__.py
         ├── data
         ├── deep_learning
         ├── eval_utils.py
         ├── meters.py
         └── visulization
         '''
    )
