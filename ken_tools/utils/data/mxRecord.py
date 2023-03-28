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


def get_record(root, name, mode='r'):
    os.makedirs(root, exist_ok=True)

    idx = osp.join(root, f'{name}.idx')
    rec = osp.join(root, f'{name}.rec')
    return mx.recordio.MXIndexedRecordIO(idx, rec, mode)