import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def collate_fn(batch):
    return list(batch)

def read_img(image_id, color, train_or_test='train', image_size=None):
    for fmt in ['png', 'jpg']:
        filename = f'{data_dir}/{train_or_test}/{image_id}_{color}.{fmt}'
        if os.path.exists(filename):
            img = cv2.imread(filename, 0)
            if img is None:
                print(filename)
            else:
                return img
    raise

class CropCellDataset(Dataset):
    def __init__(self, image_ids, mode='train'):
        self.image_ids = image_ids
        self.mode = mode
        self.folder = mode if mode != 'ext' else 'train_external'
        self.fmt = 'png' if mode != 'ext' else 'jpg'
        self.mask_dir = f'{mask_dir}/{self.mode}_cell_mask' if mode != 'ext' else f'{mask_dir}/train_cell_mask'
        if self.mode in ['train']:
            self.out_dir = f'{data_dir}/train_crop_cell_{cell_size}'
        else:
            self.out_dir = f'{data_dir}/test_crop_cell_{cell_size}'
        os.makedirs(self.out_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        if not os.path.exists(f'{self.mask_dir}/{image_id}.npz'):
            return
        red = read_img(image_id, "red", self.folder, None)
        green = read_img(image_id, "green", self.folder, None)
        blue = read_img(image_id, "blue", self.folder, None)
        yellow = read_img(image_id, "yellow", self.folder, None)
        image = np.stack([red, green, blue, yellow], axis=-1)
        cell_mask = np.load(f'{self.mask_dir}/{image_id}.npz')['arr_0']
        cell_mask = cv2.resize(cell_mask, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
        for cell_id in range(1, np.max(cell_mask)+1):
            bmask = cell_mask == cell_id
            bmask = np.array(bmask, dtype=np.uint8)
            x, y, w, h = cv2.boundingRect(bmask)
            max_l = max(w, h)
            cx = x + w // 2
            cy = y + h // 2
            x1 = max(0, cx - max_l // 2)
            x1 = min(x1, image.shape[1]-max_l)
            y1 = max(0, cy - max_l // 2)
            y1 = min(y1, image.shape[0]-max_l)
            tmp = image.copy()
            tmp[bmask==0] = 0
            cropped_cell = tmp[y1:y1+max_l, x1:x1+max_l]
            if max_l > cell_size:
                cropped_cell = cv2.resize(cropped_cell, (cell_size, cell_size))
            cv2.imwrite(f'{self.out_dir}/{image_id}_{cell_id:04d}.png', cropped_cell)


data_dir = './input/hpa-single-cell-image-classification'
mask_dir = './input/hpa-512'
cell_size = 256

df_train = pd.read_csv(f'{data_dir}/train.csv')
dataset = CropCellDataset(df_train.ID.values, mode='train')
dataloader = DataLoader(dataset, batch_size=64, num_workers=64, collate_fn=collate_fn)
for _ in tqdm(dataloader):
    pass

df_ext = pd.read_csv(os.path.join(data_dir, f'train_external.csv'))
dataset_ext = CropCellDataset(df_ext.ID.values, mode='ext')
loader_ext = DataLoader(dataset_ext, batch_size=64, num_workers=64, collate_fn=collate_fn)
for _ in tqdm(loader_ext):
    pass
