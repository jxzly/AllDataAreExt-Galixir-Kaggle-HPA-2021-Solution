import warnings
warnings.filterwarnings("ignore")

import os
import sys
import typing as t

# libraries
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools import _mask as coco_mask
import zlib
import base64

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.cuda.amp as amp

import albumentations
import timm
from tqdm import tqdm

from model import *

data_dir = './input/hpa-single-cell-image-classification/'
mask_dir = './input/hpa-512/test_cell_mask/'
model_dir = './output/'
device = torch.device('cuda')

TTA = {
    'masked': 3,
    'cells_128': 8,
    'cells_256': 6,
    'center_cells': 3,
}
n_ch = 4
num_classes = 19
image_size = 512

orig_mean = [239.93038613, 246.05603962, 250.16871503, 250.50623682]

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode('ascii')

def binary_mask_to_ascii(mask, mask_val=1):
    """Converts a binary mask into OID challenge encoding ascii text."""
    mask = np.where(mask==mask_val, 1, 0).astype(np.bool)

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()

def read_img(image_id, color, train_or_test='test', image_size=None):
    filename = f'{data_dir}/{train_or_test}/{image_id}_{color}.png'
    img = cv2.imread(filename, 0)
    return img

class HPADatasetTest(Dataset):
    def __init__(self, image_ids, mode='test'):
        self.image_ids = image_ids
        self.mode = mode

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        #try:
        image_id = self.image_ids[index]
        red = read_img(image_id, "red", self.mode, 0)
        green = read_img(image_id, "green", self.mode, 0)
        blue = read_img(image_id, "blue", self.mode, 0)
        yellow = read_img(image_id, "yellow", self.mode, 0)
        image = np.stack([blue, green, red, yellow], axis=-1)

        image_512 = cv2.resize(image, (512, 512)).transpose(2,0,1).astype(np.float32)
        image_768 = cv2.resize(image, (768, 768)).transpose(2,0,1).astype(np.float32)
        cell_mask = np.load(f'{mask_dir}/{image_id}.npz')['arr_0']
        ### for debug
#             cell_mask = np.zeros(cell_mask.shape).astype(int) if random.random() < 0.5 else cell_mask
        ###
        cell_mask = cv2.resize(cell_mask, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)

        encs = ''
        masked_images = []
        cells_128 = []
        cells_256 = []
        center_cells = []
        for cell_id in range(1, np.max(cell_mask)+1):
            bmask = (cell_mask == cell_id).astype(np.uint8)
            enc = encode_binary_mask(bmask==1)
            x, y, w, h = cv2.boundingRect(bmask)

            max_l = max(w, h)
            cx = x + w // 2
            cy = y + h // 2
            x1 = max(0, cx - max_l // 2)
            x1 = min(x1, image.shape[1] - max_l)
            y1 = max(0, cy - max_l // 2)
            y1 = min(y1, image.shape[0] - max_l)
            tmp = image.copy()
            tmp[bmask==0] = 0

            cropped_cell_orig = tmp[y1:y1+max_l, x1:x1+max_l]
            cropped_cell_128 = cv2.resize(cropped_cell_orig, (128, 128))
            cells_128.append(cropped_cell_128)
            cropped_cell_256 = cv2.resize(cropped_cell_orig, (256, 256))
            cells_256.append(cropped_cell_256)
            masked = cv2.resize(tmp, (image_size, image_size))
            masked_images.append(masked)

            ### daishu part
            cropped_cell = cv2.resize(tmp[y:y+h, x:x+w],
                                        (int(w / image.shape[1] * 768),
                                         int(h / image.shape[0] * 768))
                                     )
            final_size = 512
            new_h, new_w, _ = cropped_cell.shape
            new_h = final_size if cropped_cell.shape[0] > final_size else new_h
            new_w = final_size if cropped_cell.shape[1] > final_size else new_w
            cropped_cell = cv2.resize(cropped_cell, (new_w, new_h))

            center_cell = np.zeros((final_size, final_size, 4))
            center = final_size // 2
            h_start = max(0,center-cropped_cell.shape[0]//2)
            h_end = min(final_size,h_start+cropped_cell.shape[0])
            w_start = max(0,center-cropped_cell.shape[1]//2)
            w_end = min(final_size,w_start+cropped_cell.shape[1])

            center_cell[h_start:h_end,w_start:w_end,:] = cropped_cell
            center_cells.append(center_cell)
            ###

            if encs == '':
                encs += enc
            else:
                encs = encs + ' ' + enc

        if len(masked_images) > 0:
            masked_images = np.stack(masked_images).transpose(0, 3, 1, 2).astype(np.float32)
            cells_128 = np.stack(cells_128).transpose(0, 3, 1, 2).astype(np.float32)
            cells_256 = np.stack(cells_256).transpose(0, 3, 1, 2).astype(np.float32)
            center_cells = np.stack(center_cells).transpose(0, 3, 1, 2).astype(np.float32)
        else:
            masked_images = np.zeros((4, 4, image_size, image_size))
            cells_128 = np.zeros((4, 4, 128, 128))
            cells_256 = np.zeros((4, 4, 256, 256))

        for ch in range(4):
            image_512[ch] /= orig_mean[ch]
            image_768[ch] /= orig_mean[ch]
            masked_images[:, ch] /= orig_mean[ch]
            cells_128[:, ch] /= orig_mean[ch]
            cells_256[:, ch] /= orig_mean[ch]
            center_cells[:, ch] /= orig_mean[ch]

        '''except:
            image_id = ''
            encs = ''
            image_512 = np.zeros((4, 512, 512))
            image_768 = np.zeros((4, 768, 768))
            masked_images = np.zeros((5, 4, image_size, image_size))
            cells_128 = np.zeros((5, 4, 128, 128))
            cells_256 = np.zeros((5, 4, 256, 256))
            center_cells = np.zeros((5, 4, 512, 512))'''

        return image_id, encs, {
            '512': torch.tensor(image_512),
            '768': torch.tensor(image_768),
            'masked': torch.tensor(masked_images),
            'cells_128': torch.tensor(cells_128),
            'cells_256': torch.tensor(cells_256),
            'center_cells': torch.tensor(center_cells)
        }

df_sub = test = pd.read_csv(data_dir+'sample_submission.csv')

dataset = HPADatasetTest(df_sub.ID.values, mode='test')
dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

kernel_types = {
    'rex150_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'rexnet_150',
        'input_type': ['512', 'masked'],
    },
    'resnet50d_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [1],
        'enet_type': 'resnet50d',
        'input_type': ['512', 'masked'],
    },
    'densenet121_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [2],
        'enet_type': 'densenet121',
        'input_type': ['512', 'masked'],
    },
    'b0_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [3],
        'enet_type': 'tf_efficientnet_b0_ns',
        'input_type': ['512', 'masked'],
    },
    'resnet101d_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [4],
        'enet_type': 'resnet101d',
        'input_type': ['512', 'masked'],
    },
    'dpn68b_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0],
        'enet_type': 'dpn68b',
        'input_type': ['512', 'masked'],
    },
    'densenet169_512_multilabel_8flips_ss22rot45_co7_lr1e4_bs32_focal_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [1],
        'enet_type': 'densenet169',
        'input_type': ['512', 'masked'],
    },
    'densenet121_512_multilabel_8flips_ss22rot45_lr1e4_bs32_lovasz_ext_15epo': {
        'model_class': 'enetv2',
        'folds': [0,1,2,3,4],
        'enet_type': 'densenet121',
        'input_type': ['512', 'center_cells'],
    },
    ### 2.5d 128
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

def load_state_dict(model, model_file):
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

models = []
input_types = []
for key in kernel_types.keys():
    for fold in kernel_types[key]['folds']:
        model = eval(kernel_types[key]['model_class'])(
            kernel_types[key]['enet_type'],num_classes,pretrained=False
        )
        model = model.to(device)
        model_file = f'{key}_final_fold{fold}.pth'
        print(f'loading {model_file} ...')
        model = load_state_dict(model, model_file)
        if model is not None:
            models.append(model)
            input_types.append(kernel_types[key]['input_type'])

n_models = len(models)
print('done!')
print('model count:', n_models)

def get_trans(img, I, mode='bgry'):
    if mode == 'rgby':
        img = img[:, [2,1,0,3]]
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)

print('TTA', TTA)


IDs = []
encs = []
PRED_FINAL = []
little_bs = 16
with torch.no_grad():
    for ID, enc, images in tqdm(dataloader):
        try:
            if len(enc[0]) > 0:
                with amp.autocast():
                    for k in images.keys():
                        images[k] = images[k].cuda()
                        if images[k].ndim == 5:
                            images[k] = images[k].squeeze(0)

                    preds = {
                        'cells': [],
                    }

                    # single cell
                    for m, inp_types in zip(models, input_types):
                        for t in inp_types:
                            if t in ['masked', 'cells_128', 'cells_256','center_cells']:
                                for I in np.random.choice(range(8), TTA[t], replace=False):
                                    this_pred = torch.cat([
                                        m(get_trans(images[t][b:b+little_bs], I)).sigmoid() \
                                            for b in range(0, images[t].shape[0], little_bs)
                                    ])
                                    preds['cells'].append(this_pred)

                    for k in preds.keys():
                        if len(preds[k]) > 0:
                            preds[k] = torch.stack(preds[k], 0).mean(0)
                        else:
                            preds[k] = 0

                    pred_final = preds['cells']

                    PRED_FINAL.append(pred_final.cpu())
                    IDs += [ID[0]] * images['cells_128'].shape[0]
                    encs += enc[0].split(' ')
        except:
            print('error')
            pass

PRED_FINAL = torch.cat(PRED_FINAL).float()
print(PRED_FINAL)
print(PRED_FINAL.shape, PRED_FINAL.max(), PRED_FINAL.min(), PRED_FINAL.mean())
PredictionString = []
for i in tqdm(range(PRED_FINAL.shape[0])):
    enc = encs[i]
    prob = PRED_FINAL[i]
    sub_string = []
    for cid, p in enumerate(prob):
        sub_string.append(' '.join([str(cid), f'{p:.5f}', enc]))
    sub_string = ' '.join(sub_string)
    PredictionString.append(sub_string)
df_pred = pd.DataFrame({
    'ID': IDs,
    'PredictionString': PredictionString
})
df_pred = df_pred.groupby(['ID'])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()
df_sub = df_sub[['ID', 'ImageWidth', 'ImageHeight']].merge(df_pred, on='ID', how="left")
df_sub.fillna('', inplace=True)
df_sub.to_csv('./output/submission.csv', index=False)
