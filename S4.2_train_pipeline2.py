
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
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

use_amp = True
data_dir = './input/hpa-single-cell-image-classification'
image_folder = os.path.join(data_dir, f'train_crop_cell_256')

batch_size = 32
cell_per_img = 16
num_workers = 16
init_lr = 1e-4
num_classes = 19
n_ch = 4
lw = [4, 1]
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
cell_list = os.listdir(image_folder)
df_cell = pd.DataFrame({
    'filename': cell_list
})
df_cell['ID'] = df_cell.filename.apply(lambda x: x[: -9])

df_train_all = df_train_all.merge(pd.DataFrame({'ID': df_cell.ID.unique(), 'ok': True}), on='ID', how='left').query('ok == True').reset_index(drop=True)
print(df_train_all)
def make_filepath(row):
    return os.path.join(image_folder, row['filename'])
df_cell['filepath'] = df_cell.apply(lambda x: make_filepath(x), axis=1)
cell_dict = {
    ID: df for ID, df in df_cell.groupby('ID')
}


class CellDataset(Dataset):
    def __init__(self, df, cell_dict, mode='train', transform=None):

        self.df = df.reset_index()
        self.cell_dict = cell_dict
        self.mode = mode
        self.folder = 'test' if mode == 'test' else 'train'
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        df_this = self.cell_dict[row.ID]
        if self.mode != 'test':
            df_this = df_this.sample(cell_per_img) if df_this.shape[0] >= cell_per_img else df_this.sample(cell_per_img, replace=True)

        images = []
        for f in df_this['filepath'].values:
            image = np.asarray(PIL.Image.open(f)).copy()
            image = self.transform(image=image)['image'].astype(np.float32)

            for ch in range(4):
                if row.is_ext == 0:
                    image[:,:,ch] /= orig_mean[ch]
                else:
                    image[:,:,ch] /= ext_mean[ch]
            image = image.transpose(2, 0, 1)
            images.append(image)
        images = np.stack(images, 0)

        if self.mode == 'test':
            return torch.tensor(images).float()
        else:
            label = np.zeros(num_classes)
            for l in (row.Label.split('|')):
                if int(l) >= 0:
                    label[int(l)] = 1.
            return torch.tensor(images).float(), torch.tensor(label).float()



def get_transforms(cell_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.75),
        albumentations.Resize(cell_size, cell_size),
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(cell_size, cell_size),
    ])

    return transforms_train, transforms_val


def criterion(logits, targets):
    return nn.BCEWithLogitsLoss()(logits.view(-1), targets.view(-1))


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


def train_epoch(model, loader, optimizer, scaler=None):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, targets) in bar:

        optimizer.zero_grad()
        data, targets = data.to(device), targets.to(device)
        bs = data.shape[0]
        data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])

        if use_torch_amp:
            with amp.autocast():
                logits = model(data)
                logits = logits.reshape(bs, cell_per_img, -1)
                loss1 = criterion(logits.max(1).values, targets)
                loss2 = criterion(logits.mean(1), targets)
                loss = (loss1 * lw[0] + loss2 * lw[1]) / sum(lw)
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif use_amp:
            logits = model(data)
            logits = logits.reshape(bs, cell_per_img, -1)
            loss1 = criterion(logits.max(1).values, targets)
            loss2 = criterion(logits.mean(1), targets)
            loss = (loss1 * lw[0] + loss2 * lw[1]) / sum(lw)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        else:
            logits = model(data)
            logits = logits.reshape(bs, cell_per_img, -1)
            loss1 = criterion(logits.max(1).values, targets)
            loss2 = criterion(logits.mean(1), targets)
            loss = (loss1 * lw[0] + loss2 * lw[1]) / sum(lw)
            loss.backward()
            optimizer.step()


        loss_np = loss.item()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
        bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))

    return np.mean(train_loss)


def valid_epoch(model, loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    with torch.no_grad():
        for (data, targets) in tqdm(loader):
            data, targets = data.to(device), targets.to(device)

            bs = data.shape[0]
            data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])
            logits = model(data)
            logits = logits.reshape(bs, cell_per_img, -1)

            loss1 = criterion(logits.max(1).values, targets)
            loss2 = criterion(logits.mean(1), targets)
            loss = (loss1 * lw[0] + loss2 * lw[1]) / sum(lw)
            val_loss.append(loss.item())

            LOGITS.append(logits.max(1).values.cpu())
            TARGETS.append(targets.cpu())
#     import ipdb; ipdb.set_trace()
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS, 0).sigmoid().numpy()
    TARGETS = torch.cat(TARGETS, 0).numpy()

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
        return val_loss, acc, aucs


def run(fold):

    log_file = os.path.join(log_dir, f'log_{kernel_type}.txt')
    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    train_ = df_train_all.query(f'fold!={fold}').copy()
    valid_ = df_train_all.query(f'fold=={fold}').copy()

    transforms_train, transforms_val = get_transforms(cell_size)
    dataset_train = CellDataset(train_, cell_dict, 'train', transform=transforms_train)
    dataset_valid = CellDataset(valid_, cell_dict, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = enetv2(enet_type, num_classes)
    model = model.to(device)
    aucs_max = 0
    best_loss = 100
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
#     if DP: model = nn.DataParallel(model)
    if use_amp and not use_torch_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    scaler = amp.GradScaler() if use_torch_amp else None

    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch-1)

        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        val_loss, acc, aucs = valid_epoch(model, valid_loader)

        content = time.ctime() + ' ' + 'Fold: ' + str(fold) + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {val_loss:.4f}, acc: {acc:.4f}, aucs: {np.mean(aucs):.4f}.'
        content += '\n' + ' '.join([f'{x:.4f}' for x in aucs])
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if val_loss < best_loss:
            print('val loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss, val_loss))
            torch.save(model.state_dict(), model_file)
            best_loss = val_loss

    torch.save(model.state_dict(), model_file.replace('best', 'final'))


configs = {
    'b0_3d128_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [2],
        'enet_type': 'tf_efficientnet_b0_ns',
        'input_type': ['cells_128'],
    },
    'resnet50d_3d128_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'resnet50d',
        'input_type': ['cells_128'],
    },
    'mixnet_m_3d128_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'mixnet_m',
        'input_type': ['cells_128'],
    },
    'densenet121_3d128_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [3],
        'enet_type': 'densenet121',
        'input_type': ['cells_128'],
    },
    ### 2.5d 256
    'b0_3d256_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'tf_efficientnet_b0_ns',
        'input_type': ['cells_256'],
    },
    'b1_3d256_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [3],
        'enet_type': 'tf_efficientnet_b1_ns',
        'input_type': ['cells_256'],
    },
    'densenet121_3d256_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [2],
        'enet_type': 'densenet121',
        'input_type': ['cells_256'],
    },
    'dpn68b_3d256_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [4],
        'enet_type': 'dpn68b',
        'input_type': ['cells_256'],
    },
    'mixnet_m_3d256_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [2],
        'enet_type': 'mixnet_m',
        'input_type': ['cells_256'],
    },
    'resnet50d_3d256_multilabel_8flips_ss22rot45_lr1e4_bs32cell16_bce_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [1],
        'enet_type': 'resnet50d',
        'input_type': ['cells_256'],
    },
}

# for kernel_type in configs:
#     print(f'running {kernel_type}')
#     folds = configs[kernel_type]['folds']
#     enet_type = configs[kernel_type]['enet_type']
#     cell_size = int(configs[kernel_type]['input_type'][0].split('_')[-1])
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
    cell_size = int(configs[kernel_type]['input_type'][0].split('_')[-1])

    if args.DEBUG:
        df_train_all = df_train_all.head(100)

    print(f'running {kernel_type}')

    for fold in folds:
        run(fold)
