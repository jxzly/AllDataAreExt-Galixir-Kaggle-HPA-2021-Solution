import os
import cv2,PIL
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial

from torch.utils.data import DataLoader,Dataset

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from hpasegmentator import *

data_dir = './input/hpa-single-cell-image-classification/'
out_size = 512
os.makedirs('./input/hpa-512/train/', exist_ok=True)
os.makedirs('./input/hpa-512/test/', exist_ok=True)
os.makedirs('./input/hpa-512/train_cell_mask/', exist_ok=True)
os.makedirs('./input/hpa-512/test_cell_mask/', exist_ok=True)

train = pd.read_csv(data_dir+'train.csv')
test = pd.read_csv(data_dir+'sample_submission.csv')
train_external = pd.read_csv(data_dir+'train_external.csv')
train['is_ext'] = 0
test['is_ext'] = 0
train_external['is_ext'] = 1
train_all = train.append(train_external).reset_index(drop=True)
def resize_combine_img(idx,df,output_folder,train_or_test):
    row = df.iloc[idx]
    images = []
    for color in ['red', 'green', 'blue', 'yellow']:
        if train_or_test == 'train':
            if row.is_ext == 0:
                filepath = os.path.join(data_dir, 'train', row.ID + f'_{color}.png')
            else:
                filepath = os.path.join(data_dir, 'train_external', row.ID + f'_{color}.png')
                if not os.path.exists(filepath):
                    filepath = filepath.replace('.png','.jpg')
        else:
            filepath = os.path.join(data_dir, 'test', row.ID + f'_{color}.png')
        if os.path.exists(filepath):
            images.append(cv2.imread(filepath, 0))
        else:
            print(filepath)
            return
    image = np.stack(images, -1)
    image_size = image.shape[0]
    # print(row.ID,image)
    image = cv2.resize(image, (out_size, out_size))
    # Save image to local
    out_path = os.path.join(output_folder, row.ID + '.png')
    cv2.imwrite(out_path, image)
    return image_size

with multiprocessing.Pool(multiprocessing.cpu_count()*2) as p:
    train_image_sizes = list(tqdm(p.imap(partial(resize_combine_img,df=train_all,output_folder='./input/hpa-512/train/',train_or_test='train'), list(range(train_all.shape[0]))), total=train_all.shape[0]))
    test_image_sizes = list(tqdm(p.imap(partial(resize_combine_img,df=test,output_folder='./input/hpa-512/test/',train_or_test='test'), list(range(test.shape[0]))), total=test.shape[0]))
train_all['n_label'] = train_all['Label'].apply(lambda x:len(x.split('|')))
train_all['fold'] = 0
train_all['image_size'] = train_image_sizes
test['image_size'] = test_image_sizes

labels = np.zeros(train_all.shape[0]*19).reshape(train_all.shape[0],19)
for i in train_all.index:
    for l in train_all.loc[i,'Label'].split('|'):
            labels[i,int(l)] = 1
skf = MultilabelStratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
for fold, (trn_index, val_index) in enumerate(skf.split(train_all,labels)):
    train_all.loc[val_index,'fold'] = fold
train_all.to_csv('./input/hpa-512/train_all.csv',index=False)

seg_size = 512

class HPADatasetSeg(Dataset):
    def __init__(self, df, root='./input/hpa-512/',train_or_test='train'):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.train_or_test = train_or_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.loc[index]
        image = filename = f'{self.root}/{self.train_or_test}/{row.ID}.png'
        image = np.asarray(PIL.Image.open(filename))
        r = image[:,:,2]
        y = image[:,:,3]
        b = image[:,:,0]
        target_shape = row.image_size
        gray_image = b
        rgb_image = np.stack((r, y, b), axis=2)
        if b.shape[0] != seg_size:
            gray_image = cv2.resize(gray_image, (seg_size, seg_size))
            rgb_image = cv2.resize(rgb_image, (seg_size, seg_size))

        return gray_image, rgb_image, target_shape, row.ID

def collate_fn(batch):
    gray = []
    rgb_image = []
    target_shape = []
    IDs = []
    for data_point in batch:
        gray.append(data_point[0])
        rgb_image.append(data_point[1])
        target_shape.append(data_point[2])
        IDs.append(data_point[3])
    return gray, rgb_image, target_shape, IDs


def segment(df,train_or_test):
    dataset_seg = HPADatasetSeg(df,train_or_test=train_or_test)
    loader_seg = DataLoader(dataset_seg, batch_size=seg_bs, num_workers=8, collate_fn=collate_fn)
    mask_dir = f'{train_or_test}_cell_mask'
    for gray, rgb, target_shapes, IDs in tqdm(loader_seg):
        nuc_segmentations = cellsegmentor.pred_nuclei(gray)
        cell_segmentations = cellsegmentor.pred_cells(rgb, precombined=True)
        for data_id, target_shape, nuc_seg, cell_seg in zip(IDs, target_shapes, nuc_segmentations, cell_segmentations):
            nuc, cell = label_cell(nuc_seg, cell_seg)
            np.savez_compressed(f'./input/hpa-512/{mask_dir}/{data_id}', cell.astype(np.uint8))

cellsegmentor = CellSegmentator()
segment(train_all,'train')
segment(test,'test')
