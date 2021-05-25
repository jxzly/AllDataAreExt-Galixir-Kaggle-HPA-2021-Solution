import warnings
warnings.filterwarnings("ignore")

import os
import sys

# libraries
import time
import numpy as np
import pandas as pd
import argparse
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from warmup_scheduler import GradualWarmupScheduler
import albumentations
import timm
from tqdm import tqdm

from model import *
from loss import *

device = torch.device('cuda')

image_size = 512
use_amp = True
data_dir = './input/hpa-single-cell-image-classification/'
image_folder = './input/hpa-512/train/'

p_drop_cell = 0.
batch_size = 32
num_workers = 36
init_lr = 1e-4
num_classes = 19
n_ch = 4

loss_type = 'BCE' # 'BCE' or 'CE'

freeze_epo = 0
warmup_epo = 1
cosine_epo = 14
n_epochs = freeze_epo + warmup_epo + cosine_epo

if use_amp:
    use_torch_amp = torch.__version__ >= '1.6'
    if use_torch_amp:
        import torch.cuda.amp as amp
    else:
        from apex import amp
else:
    use_torch_amp = False

log_dir = './output'
model_dir = './output'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

ext_mean = [30.89923273, 153.09532163,  81.67066827, 230.55380814]
orig_mean = [239.93038613, 246.05603962, 250.16871503, 250.50623682]

df_train_all = pd.read_csv('./input/hpa-512/train_all.csv')
df_train_all['filepath'] = df_train_all.apply(lambda row: os.path.join(image_folder, row.ID + '.png'), axis=1)
print(os.path.exists(df_train_all.loc[0].filepath), df_train_all.loc[0].filepath)
print(os.path.exists(df_train_all.iloc[-1].filepath), df_train_all.iloc[-1].filepath)

class HpaImageDataSet1:
    def __init__(self, df, transform=None):
        self.df = df.reset_index()
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = np.asarray(PIL.Image.open(row.filepath)).copy()
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        for ch in range(4):
            if row.is_ext == 0 or row.is_ext == 2:
                image[:,:,ch] /= orig_mean[ch]
            else:
                image[:,:,ch] /= ext_mean[ch]
        image = image.transpose(2, 0, 1)

        label = np.zeros(num_classes)
        for l in (row.Label.split('|')):
            label[int(l)] = 1.
        return torch.tensor(image).float(), torch.tensor(label).float()

class HpaImageDataSet2:
    def __init__(self, df, image_size=None, crop_size=None, transform=None, cutmix_neg=False, mix_color=False, random_ch=False):
        self.df = df
        self.image_size = image_size
        self.crop_size = crop_size
        self.transform = transform
        self.cutmix_neg = cutmix_neg
        self.mix_color = mix_color
        self.random_ch = random_ch

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mask = None
        image = np.asarray(PIL.Image.open(row.filepath)).copy()
        image = cv2.resize(image,(self.image_size,self.image_size))

        if self.crop_size is not None:
            random_crop_size = int(np.random.uniform(self.crop_size, self.image_size))
            x = int(np.random.uniform(0, self.image_size - random_crop_size))
            y = int(np.random.uniform(0, self.image_size - random_crop_size))
            image = image[x:x + random_crop_size, y:y + random_crop_size,:]
            image = cv2.resize(image,(self.crop_size,self.crop_size))
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = image.astype(np.float32)
        image = np.transpose(image,(2,0,1))

        for ch in range(4):
            if row.is_ext == 0:
                image[ch] = image[ch] / orig_mean[ch]
            else:
                image[ch] = image[ch] / ext_mean[ch]

        add_neg_cell = False
        mix_red = False
        mix_blue = False
        mix_yellow = False
        rand_prob = np.random.rand()
        if self.cutmix_neg and rand_prob < 0.05:
            image[1,...] = image[1,...] * rand_prob * 2
            add_neg_cell = True
        elif self.mix_color and 0.05 < rand_prob < 0.075:
            image[1,...] = image[0,...] * (1-(rand_prob-0.05)*16)
            mix_red = True
        elif self.mix_color and 0.075 < rand_prob < 0.1:
            image[1,...] = image[3,...] * (1-(rand_prob-0.075)*16)
            mix_yellow = True
        elif self.random_ch and 0.1 < rand_prob < 0.15:
            ch_probs = np.random.rand(4)*0.5+0.6
            for ch in range(4):
                image[ch] = image[ch]*ch_probs[ch]

        # image = normed(image)
        labels = np.zeros(num_classes)
        for l in (row.Label.split('|')):
            labels[int(l)] = 1.
        if add_neg_cell:
            labels[:] = 0.0
            labels[18] = 1.0
        elif mix_red:
            labels[:] = 0.0
            labels[10] = 1.0
        elif mix_yellow:
            labels[:] = 0.0
            labels[6] = 1.0
        return [torch.tensor(image, dtype=torch.float),torch.tensor(labels, dtype=torch.float)]

def mAP(pred, target):
    """ Calculate the mean average precision with respect of classes
    Args:
        pred (torch.Tensor | np.ndarray): The model prediction with shape
            (N, C), where C is the number of classes.
        target (torch.Tensor | np.ndarray): The target of each prediction with
            shape (N, C), where C is the number of classes. 1 stands for
            positive examples, 0 stands for negative examples and -1 stands for
            difficult examples.
    Returns:
        float: A single float as mAP value.
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        pred = pred.numpy()
        target = target.numpy()
    elif not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError('pred and target should both be torch.Tensor or'
                        'np.ndarray')

    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'
    num_classes = pred.shape[1]
    ap = np.zeros(num_classes)
    for k in range(num_classes):
        ap[k] = average_precision(pred[:, k], target[:, k])
    mean_ap = ap.mean() * 100.0
    return ap, mean_ap

def average_precision(pred, target):
    """ Calculate the average precision for a single class
    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:
    ..math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n
    Note that no approximation is involved since the curve is piecewise
    constant.
    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).
    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]

    # count true positive examples
    pos_inds = sort_target == 1
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]

    # count not difficult examples
    pn_inds = sort_target != -1
    pn = np.cumsum(pn_inds)

    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap

def global_average_precision_score(y_true, y_pred) -> float:

    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

import torch.optim as optim

class SchedulerBase(object):
    def __init__(self):
        self._is_load_best_weight = True
        self._is_load_best_optim = True
        self._is_freeze_bn=False
        self._is_adjust_lr = True
        self._lr = 0.01
        self._cur_optimizer = None

    def schedule(self, net, epoch, epochs, **kwargs):
        raise Exception('Did not implemented')

    def step(self, net, epoch, epochs):
        optimizer, lr = self.schedule(net, epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list += [param_group['lr']]
        return lr_list

    def is_load_best_weight(self):
        return self._is_load_best_weight

    def is_load_best_optim(self):
        return self._is_load_best_optim

    def is_freeze_bn(self):
        return self._is_freeze_bn

    def reset(self):
        self._is_load_best_weight = True
        self._load_best_optim = True
        self._is_freeze_bn = False

    def is_adjust_lr(self):
        return self._is_adjust_lr

class Adam15(SchedulerBase):
    def __init__(self, params_list=None):
        super().__init__()
        self._lr = 3e-4
        self._cur_optimizer = None
        self.params_list=params_list

    def schedule(self, net, epoch, epochs, **kwargs):
        lr = 30e-5
        if epoch > 3:
            lr = 15e-5
        if epoch > 6:
            lr = 7.5e-5
        if epoch > 9:
            lr = 3e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        return self._cur_optimizer, self._lr

if use_torch_amp:
    scaler = amp.GradScaler()

def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, targets) in bar:

        optimizer.zero_grad()
        data, targets = data.to(device), targets.to(device)

        if use_torch_amp:
            with amp.autocast():
                logits = model(data)
            loss = criterion(logits.float(), targets.float())
            if str(loss.item()) == 'nan': continue
            scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        elif use_amp:
            logits = model(data)
            loss = criterion(logits, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        else:
            logits = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        loss_np = loss.item()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
        bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))

    return np.mean(train_loss)


def valid_epoch(model, loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
#     PROBS = []
#     PREDS = []
    TARGETS = []
    with torch.no_grad():
        for (data, targets) in tqdm(loader):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = criterion(logits, targets)
            val_loss.append(loss.item())

            LOGITS.append(logits.cpu())
            TARGETS.append(targets.cpu())

#             lmax = logits.max(1)
#             probs = lmax.values
#             preds = lmax.indices
#             PROBS.append(probs.detach().cpu())
#             PREDS.append(preds.detach().cpu())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS, 0).sigmoid().numpy()
    TARGETS = torch.cat(TARGETS, 0).numpy()
#     PROBS = torch.cat(PROBS).numpy()
#     PREDS = torch.cat(PREDS).numpy()

    ap, _mAP = mAP(LOGITS, TARGETS)

    if get_output:
        return LOGITS
    else:
        aucs = []
        for cid in range(num_classes):
            try:
                this_auc = roc_auc_score(TARGETS[:, cid], LOGITS[:, cid])
            except:
                this_auc = 0.5
            aucs.append(this_auc)
        acc = ((LOGITS > 0.5).astype(int) == TARGETS).mean()
        return val_loss, acc, aucs, ap, _mAP

def run(fold):

    log_file = os.path.join(log_dir, f'log_{kernel_type}.txt')
    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    train_ = df_train_all.query(f'fold!={fold}').copy()
    valid_1 = df_train_all.query(f'fold=={fold} and is_ext==0').copy()
    valid_2 = df_train_all.query(f'fold=={fold} and is_ext==1').copy()
    global criterion

    model = enetv2(enet_type, num_classes)
    model = model.to(device)

    if not unique_aug:
        criterion = focal_loss
        dataset_train = HpaImageDataSet1(train_, transform=transforms_train1)
        dataset_valid_1 = HpaImageDataSet1(valid_1, transform=None)
        dataset_valid_2 = HpaImageDataSet1(valid_2, transform=None)

        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
        scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    else:
        criterion = FocalSymmetricLovaszHardLogLoss()
        if fold == 0:
            dataset_train = HpaImageDataSet2(train_,768,512,transforms_train2,cutmix_neg=True)
        elif fold == 1:
            dataset_train = HpaImageDataSet2(train_,768,512,transforms_train2,cutmix_neg=True)
        elif fold == 2:
            dataset_train = HpaImageDataSet2(train_,768,512,transforms_train2,cutmix_neg=True,mix_color=True)
        elif fold == 3:
            dataset_train = HpaImageDataSet2(train_,768,512,transforms_train2,cutmix_neg=True,random_ch=True)
        elif fold == 4:
            dataset_train = HpaImageDataSet2(train_,768,512,transforms_train2,cutmix_neg=True,mix_color=True,random_ch=True)

        dataset_valid_1 = HpaImageDataSet2(valid_1, image_size=512)
        dataset_valid_2 = HpaImageDataSet2(valid_2, image_size=512)

        scheduler = Adam15()
        optimizer = scheduler.schedule(model, 0, n_epochs)[0]

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader_1 = torch.utils.data.DataLoader(dataset_valid_1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_loader_2 = torch.utils.data.DataLoader(dataset_valid_2, batch_size=batch_size, shuffle=False, num_workers=num_workers)


#     model.load_state_dict(torch.load(os.path.join(model_dir, f'{kernel_type_load}_best_fold{fold}.pth')), strict=False)
    aucs_max = 0
    best_loss = 100
    best_map = 0
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')


    if use_amp and not use_torch_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        if not unique_aug:
            scheduler.step(epoch-1)
        else:
            scheduler.step(model,epoch-1, n_epochs)
        train_loss = train_epoch(model, train_loader, optimizer)
#         train_loss = -1
        val_loss_1, acc_1, aucs_1, ap1, mAP1 = valid_epoch(model, valid_loader_1)
        val_loss_2, acc_2, aucs_2, ap2, mAP2 = 0,0,0,0,0#valid_epoch(model, valid_loader_2)

        content = time.ctime() + ' ' + 'Fold: ' + str(fold) + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {val_loss_1:.4f}|{val_loss_2:.4f}, mAP: {mAP1:.4f}|{mAP2:.4f}, acc: {acc_1:.4f}|{acc_2:.4f}, aucs: {np.mean(aucs_1):.4f}|{np.mean(aucs_2):.4f}'
#         content += '\n' + ' '.join([f'{x:.4f}' for x in ap1])
#         content += '\n' + ' '.join([f'{x:.4f}' for x in ap2])
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if val_loss_1 < best_loss:
            print('val loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss, val_loss_1))
#             torch.save(model.state_dict(), model_file)
            best_loss = val_loss_1
        if mAP1 > best_map:
            print('val mAP increased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_map, mAP1))
#             torch.save(model.state_dict(), model_file.replace('best', 'bestmAP'))
            best_map = mAP1
    torch.save(model.state_dict(), model_file.replace('best', 'final'))

configs = {
    'rex150_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'rexnet_150',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'resnet50d_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [1],
        'enet_type': 'resnet50d',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'densenet121_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [2],
        'enet_type': 'densenet121',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'b0_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [3],
        'enet_type': 'tf_efficientnet_b0_ns',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'resnet101d_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [4],
        'enet_type': 'resnet101d',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'dpn68b_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'dpn68b',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'densenet169_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [1],
        'enet_type': 'densenet169',
        'input_type': ['512', 'masked'],
        'unique_aug': False
    },
    'densenet121_512_multilabel_8flips_ss22rot45_lr1e4_bs32_lovasz_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0,1,2,3,4],
        'enet_type': 'densenet121',
        'input_type': ['512', 'center_cells'],
        'unique_aug': True
    }
    }

transforms_train1 = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.75),
    albumentations.Resize(image_size, image_size),
    albumentations.Cutout(max_h_size=int(image_size * 0.7), max_w_size=int(image_size * 0.7), num_holes=1, p=0.7),
])

transforms_train2 = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.75),
])

# for kernel_type in configs:
#     print(f'running {kernel_type}')
#     enet_type = configs[kernel_type]['enet_type']
#     folds = configs[kernel_type]['folds']
#     unique_aug = configs[kernel_type]['unique_aug']
#     for fold in folds:
#         run(fold)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-id', type=int, default=0)
    parser.add_argument('--DEBUG', action='store_true')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    kernel_type = list(configs.keys())[args.config_id]
    folds = configs[kernel_type]['folds']
    enet_type = configs[kernel_type]['enet_type']
    unique_aug = configs[kernel_type]['unique_aug']

    if args.DEBUG:
        df_train_all = df_train_all.head(100)

    print(f'running {kernel_type}')

    for fold in folds:
        run(fold)
