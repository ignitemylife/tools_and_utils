import torch
import os
import argparse
import numpy as np
import json
from pdb import set_trace as st
import xgboost as xgb
import mxnet as mx
from matplotlib import pyplot as plt
from xgboost.callback import TrainingCallback
from tensorboardX import SummaryWriter
torch.random.manual_seed(1234)
np.random.seed(1234)


class myCallback(TrainingCallback):
    def __init__(self, writer):
        super(myCallback, self).__init__()
        self.writer = writer

    def after_iteration(self, model, epoch, evals_log):
        '''Run after each iteration.  Return True when training should stop.'''
        key = list(evals_log.keys())[0]
        for k, v in evals_log.get(key).items():
            self.writer.add_scalar(k, v[-1], epoch)
        return False

def visual_feat(model, savePath):
    if os.path.exists(savePath) == False:
        os.mkdir(savePath)
    for importance_type in ['weight', 'gain', 'cover', 'total_gain']:
        xgb.plot_importance(model, max_num_features=20, importance_type=importance_type)
        plt.title('xgb.plot_importance_type={}'.format(importance_type))
        plt.savefig((os.path.join(savePath, 'plot_importance_type_{}.png'.format(importance_type))))
    return None


def save_model(model, dst_root, model_name='xgb_model.bin'):
    # pickle.dump(model, open(model_name + '.txt', "wb"))
    model.get_booster().dump_model(os.path.join(dst_root, 'dump.raw.txt'))
    model.save_model(os.path.join(dst_root, model_name))


def load_model(model_name='xgb_model.bin', sklearn=True):
    # clf = pickle.load(open(model_name, "rb"))
    # visual_feat(clf) ## visual feature importance
    if sklearn:
        clf = xgb.XGBClassifier()
        booster = xgb.Booster()
        booster.load_model(model_name)
        clf._Booster = booster
    else:
        clf = xgb.Booster(model_file=model_name)
    save_model(clf, model_name='xgb_model.bin')
    return clf

def get_online_user_info(pids):
    idx_file = '/safety/zhangshuai10/data/smoke/ken_files/v1/online_user_info.idx'
    rec_file = '/safety/zhangshuai10/data/smoke/ken_files/v1/online_user_info.rec'
    reader = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
    ret = []
    default = [-1. for _ in range(11+41)]
    for pid in pids:
        try:
            feat = json.loads(mx.recordio.unpack(reader.read_idx(int(pid)))[1].decode())
            ret.append(feat)
        except:
            ret.append(default)
            print('cannot get online user info for pid: {}'.format(pid))
    return np.array(ret)

def filter_pids(feats, labels, pids, choosed_txt, remap_label=False):
    candidate_pids = set([line.strip().split(' ')[0] for line in open(choosed_txt)])
    if remap_label:
        remap = {}
        for line in open(choosed_txt):
            pid, l = line.strip().split(' ')
            remap[pid] = int(l)

    choosesd_feats = []
    choosed_labels = []
    choosed_pids = []
    for ind, pid in enumerate(pids):
        if str(pid) in candidate_pids:
            choosesd_feats.append(feats[ind])
            if remap_label:
                choosed_labels.append(remap[str(pid)])
            else:
                choosed_labels.append(labels[ind])
            choosed_pids.append(pid)

    feats, labels, pids = np.stack(choosesd_feats, axis=0), np.array(choosed_labels), np.array(choosed_pids)
    pos_cnt = labels.sum()
    neg_cnt = (labels==0).sum()
    print(f'choosed {len(choosed_pids)} from {len(pids)} pids, pos: {pos_cnt} neg: {neg_cnt}')

    return feats, labels, pids


def get_data(root):
    files = os.listdir(root)
    feats_file = sorted(list(filter(lambda x: 'feats' in x, files)))
    labels_file = sorted(list(filter(lambda x: 'label' in x, files)))
    pids_file = sorted(list(filter(lambda x: 'pid' in x, files)))
    print(feats_file)
    print(labels_file)
    print(pids_file)

    feats = [np.load(os.path.join(root, f)) for f in feats_file]
    labels = [np.load(os.path.join(root, f)) for f in labels_file]
    pids = [np.load(os.path.join(root, f)) for f in pids_file]

    feats, labels, pids = np.concatenate(feats, axis=0), np.concatenate(labels, axis=0), np.concatenate(pids, axis=0)

    return feats, labels, pids
    # user info
    # user_info = get_online_user_info(pids)
    # return np.concatenate((feats, user_info), axis=-1), labels, pids

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default='/hwsafety/konglingshu/smoke_new_rule/ken_files/v1/xgb_feats/train')
    parser.add_argument("--train_pids", type=str)
    parser.add_argument("--val_dir", type=str, default='/hwsafety/konglingshu/smoke_new_rule/ken_files/v1/xgb_feats/val')
    parser.add_argument("--val_pids", type=str)
    parser.add_argument("--work_dir", type=str, default='./del_tmp')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--out_txt", type=str, default='')
    return parser.parse_args()

def val(val_label, scores, writer):
    length = len(scores)
    pred = scores > 0.5
    acc = sum(pred == val_label[:length]) / length
    tp = np.logical_and(pred == 1, val_label[:length] == 1).sum()
    fp = np.logical_and(pred == 1, val_label[:length] == 0).sum()
    tn = np.logical_and(pred == 0, val_label[:length] == 0).sum()
    fn = np.logical_and(pred == 0, val_label[:length] == 1).sum()
    tpr = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    print('val acc: {:.4f} pr: {} recall: {}'.format(acc, tpr, recall))
    writer.add_pr_curve(f'val', val_label[:length], scores, 0)
    print('add pr curve')

if __name__ == '__main__':
    args = parse_args()
    val_dir = args.val_dir
    train_dir = args.train_dir
    writer = SummaryWriter(args.work_dir)
    classifier = xgb.XGBClassifier(max_depth=6,
                                   learning_rate=0.05,
                                   n_estimators=1200,
                                   objective='binary:logistic',
                                   eval_metric=['map', 'auc'],
                                   verbosity=1,
                                   base_score=0.13,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   alpha=1,
                                   gamma=0.2,
                                   min_child_weight=5,
                                   random_state=1234
                                   )
    if args.ckpt:
        classifier.load_model(args.ckpt)
        print('loaded from {}'.format(args.ckpt))
        
    print('=== val data ===')
    val_data, val_label, val_pid = get_data(val_dir)

    val_data, val_label, val_pid = filter_pids(val_data, val_label, val_pid, args.val_pids, remap_label=True)
    if args.train:
        print('=== train data ===')
        train_data, train_label, train_pid = get_data(train_dir)
        train_data, train_label, train_pid = filter_pids(train_data, train_label, train_pid, args.train_pids)

        eval_set = [(val_data, val_label)]

        classifier.fit(train_data, train_label, early_stopping_rounds=40, eval_set=eval_set, verbose=True,
                       callbacks=[myCallback(writer)])
        save_model(classifier, args.work_dir)
        visual_feat(classifier, args.work_dir)
    else:
        print('validattion only!!')
        

    preds = classifier.predict_proba(val_data)
    val(val_label, preds[:, 1], writer)

    if args.out_txt:
        f = open(args.out_txt, 'w')
        for pid, pred in zip(val_pid.tolist(), preds.tolist()):
            f.write('{},{}\n'.format(pid, pred[-1]))
        f.close()
        print('write preds to:\n{}'.format(args.out_txt))


