import torch
import os
import os.path as osp
import argparse
import numpy as np
import json
from pdb import set_trace as st
import xgboost as xgb
import mxnet as mx
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
torch.random.manual_seed(44)
np.random.seed(44)

def myCallback(writer):
    def callback(env):
        """internal function"""
        for k, v in env.evaluation_result_list:
            pos = k.index('-')
            key = k[:pos]
            metric = k[pos + 1:]
            writer.add_scalar('{}/{}'.format(key, metric), v, env.iteration)

    return callback


def visual_feat(model, savePath):
    if os.path.exists(savePath) == False:
        os.mkdir(savePath)
    for importance_type in ['weight', 'gain', 'cover', 'total_gain']:
        xgb.plot_importance(model, max_num_features=20, importance_type=importance_type)
        plt.title('xgb.plot_importance_type={}'.format(importance_type))
        plt.savefig((os.path.join(savePath, 'plot_importance_type_{}.png'.format(importance_type))))
    return None


def save_model(model, dst_root, model_name='xgb_model.bin'):
    model.save_model(os.path.join(dst_root, model_name))
    # pickle.dump(model, open(model_name + '.bin', "wb"))
    try:
        model.get_booster().dump_model(os.path.join(dst_root, 'dump.raw.txt'))
    except:
        model.dump_model(os.path.join(dst_root, 'dump.raw.txt'))



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
    # save_model(clf, model_name='xgb_model.bin')
    return clf

def filter_pids(feats, labels, pids, choosed_txt, remap_label=False):
    '''
    this function filter pids those not feed into xgb model
    '''
    pass


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default='/hwsafety/konglingshu/smoke_new_rule/ken_files/v1/xgb_feats/train')
    parser.add_argument("--val_dir", type=str, default='/hwsafety/konglingshu/smoke_new_rule/ken_files/v1/xgb_feats/val')
    parser.add_argument("--work_dir", type=str, default='./del_tmp')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--ckpt", type=str, default='')
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
    writer.add_pr_curve(f'pr_curve', val_label[:length], scores, 0)
    print('add pr curve')


if __name__ == '__main__':
    args = parse_args()
    val_dir = args.val_dir
    train_dir = args.train_dir
    params = dict(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1200,
        objective='binary:logistic',
        eval_metric=['map', 'auc'],
        verbosity=1,
        base_score=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        alpha=1,
        gamma=0.2,
        min_child_weight=5,
        random_state=1234
    )
    writer = SummaryWriter(args.work_dir)

    print('=== val data ===')
    val_data, val_label, val_pid = get_data(val_dir)
    # val_data = val_data[:, :1747]
    print(f'val_data.shape = {val_data.shape}')

    if args.train:
        print('=== train data ===')
        train_data, train_label, train_pid = get_data(train_dir)
        # train_data = train_data[:, :1747]
        print(f'train_data.shape = {train_data.shape}')

        eval_set = [(xgb.DMatrix(val_data, val_label), "validation")]
        dtrain = xgb.DMatrix(train_data, train_label)

        classifier = xgb.train(params, dtrain, num_boost_round=1200, evals=eval_set, early_stopping_rounds=40, callbacks=[myCallback(writer)])
        save_model(classifier, args.work_dir)
        visual_feat(classifier, args.work_dir)
    else:
        assert args.ckpt is not None
        print('validattion only!!')
        classifier = xgb.Booster()
        classifier.load_model(args.ckpt)
        print('loaded from {}'.format(args.ckpt))

    preds = classifier.predict(xgb.DMatrix(val_data))
    val(val_label, preds, writer)

    # wirte to val dataset preds to txt
    out_txt = osp.join(args.work_dir, 'pred.txt')
    f = open(out_txt, 'w')
    for pid, pred in zip(val_pid.tolist(), preds.tolist()):
        f.write('{},{}\n'.format(pid, pred))
    f.close()
    print('write preds to:\n{}'.format(out_txt))


