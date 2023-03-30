import os
import os.path as osp
import json
from tqdm import tqdm
import mxnet as mx
from collections import Counter, defaultdict
import argparse
import numpy as np
from pdb import set_trace as st
import time
from typing import Union


def encode(record, index, label, data):
    header = mx.recordio.IRHeader(0, label, 0, 0)
    # data = json.dumps([1,2,3])
    if isinstance(data, str):
        data = data.encode()
    s = mx.recordio.pack(header, data)

    if isinstance(index, str):
        index = int(index)
    record.write_idx(index, s)


def decode(record, index):
    if isinstance(index, str):
        index = int(index)
    s = record.read_idx(index)
    header, mes = mx.recordio.unpack(s)
    mes = mes.decode()
    data = json.loads(mes)
    return data, header.label


def get_record_wo_postfix(root, mode='r'):
    root = root.replace('.rec', '').replace('.idx', '')
    idx = root + '.idx'
    rec = root + '.rec'
    if mode == 'w':
        if osp.exists(idx) or osp.exists(rec):
            raise ValueError('overwrite record is not allowed, please manually delete it and re-run')

    os.makedirs(os.path.dirname(idx), exist_ok=True)
    return mx.recordio.MXIndexedRecordIO(idx, rec, mode)


def get_record(meta: Union[tuple, list, str], mode='r'):
    def _check_exist_if_write(idx, rec, mode):
        if mode != 'w':
            return
        if osp.exists(idx) or osp.exists(rec):
            raise ValueError('overwrite record is not allowed, please manually delete it and re-run')

    op = 'read' if mode=='r' else 'write'
    if isinstance(meta, (tuple, list)):
        print(f'{op} record from {meta[0]}')
        try:
            idx = meta[0]
            rec = meta[1]

            _check_exist_if_write(idx, rec, mode)
            record = mx.recordio.MXIndexedRecordIO(meta[0], meta[1], mode)
        except:
            idx = osp.join(meta[0], f'{meta[1]}.idx')
            rec = osp.join(meta[0], f'{meta[1]}.rec')

            _check_exist_if_write(idx, rec, mode)
            record = mx.recordio.MXIndexedRecordIO(idx, rec, mode)
    else:
        print(f'{op} record from {meta}')
        record = get_record_wo_postfix(meta, mode)

    return record


def merge_records(*records, **kwargs):
    dst_root = kwargs.get('dst_root', '.')
    dst_name = kwargs.get('dst_name', 'total.merge')
    os.makedirs(dst_root, exist_ok=True)

    new_record = get_record(osp.join(dst_root, dst_name), 'w')
    for record in records:
        for key in tqdm(record.keys):
            s = record.read_idx(key)
            new_record.write_idx(key, s)

    new_record.close()
    print('merged')