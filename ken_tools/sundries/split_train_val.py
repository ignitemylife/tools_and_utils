import os
import os.path as osp
import argparse
import random
from ken_tools.utils.data import get_record

class SplitTrainVal():
    @staticmethod
    def split_file(filename, val_num, ratio=None, save_dir=None):
        lines = open(filename).readlines()
        total = len(lines)

        ind = SplitTrainVal.__get_ind(val_num, ratio, total)

        if not save_dir:
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = './'

        writer1 = open(os.path.join(save_dir, 'train.txt'), 'w')
        writer2 = open(os.path.join(save_dir, 'val.txt'), 'w')

        for line in lines[:ind]:
            writer1.write(line)
        for line in lines[ind:]:
            writer2.write(line)

    @staticmethod
    def split_json(filename, val_num, ratio=None, save_dir=None):
        import json

        total = json.load(open(filename))
        keys = list(total.keys())
        random.shuffle(keys)

        val_num = SplitTrainVal.__get_ind(val_num, ratio, len(keys))

        val = {key: total[key] for key in keys[:val_num]}
        train = {key: total[key] for key in keys[val_num:]}
        if not save_dir:
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = './'
        json.dump(val, open(os.path.join(save_dir, 'val_temp.json'), 'w'))
        json.dump(train, open(os.path.join(save_dir, 'train_temp.json'), 'w'))
        print('done')


    @staticmethod
    def split_record(filename, val_num, ratio=None, save_dir=None):
        import mxnet as mx # lazy import
        record = get_record(filename, 'r')
        os.makedirs(save_dir, exist_ok=True)

        train_name = osp.join(save_dir, record.split('.')[0] + 'train')
        train_record = get_record(train_name, 'w')

        val_name = osp.join(save_dir, record.split('.')[0] + 'val')
        val_record = get_record(val_name, 'w')

        keys = record.keys
        total = len(keys)

        ind = SplitTrainVal.__get_ind(val_num, ratio, total)
        for key in keys[ind:]:
            data = record.read_idx(key)
            train_record.write_idx(key, data)

        for key in keys[:ind]:
            data = record.read_idx(key)
            val_record.write_idx(key, data)

        record.close()
        train_record.close()
        val_record.close()
        print(f'split and save to {save_dir}')


    @staticmethod
    def __get_ind(val_num, ratio, total):
        if val_num is not None and val_num > 0:
            ind = val_num
        elif ratio is not None:
            assert 0 <= ratio <=1, "please check `ratio`"
            ind = int(total * ratio)
        else:
            raise ValueError

        return ind


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--val-num', default=1000, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    filename = args.filename
    posfix = filename.split('.')[-1]

    if posfix in ('txt', 'csv'):
        SplitTrainVal.split_file(filename, args.val_num, args.save_dir)
    elif posfix == 'json':
        SplitTrainVal.split_json(filename, args.val_num, args.sava_dir)


