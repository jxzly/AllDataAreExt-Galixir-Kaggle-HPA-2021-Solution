#  https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg

import os
import cv2
import requests
import gzip
import io
import imageio
import multiprocessing
import pandas as pd
from tqdm import tqdm

data_dir = './input/hpa-single-cell-image-classification/' #https://www.kaggle.com/c/hpa-single-cell-image-classification/data

save_dir = os.path.join(data_dir, 'train_external')
os.makedirs(save_dir, exist_ok=True)

all_locations = dict({
    "Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18
})


def add_label_idx(df, all_locations):
    '''Function to convert label name to index
    '''
    df["Label_idx"] = None
    for i, row in df.iterrows():
        labels = row.Label.split(',')
        idx = []
        for l in labels:
            if l in all_locations.keys():
                idx.append(str(all_locations[l]))
        if len(idx)>0:
            df.loc[i,"Label_idx"] = "|".join(idx)

        print(df.loc[i,"Label"], df.loc[i,"Label_idx"])
    return df

public_hpa_df = pd.read_csv(f'{data_dir}/kaggle_2021.tsv') #https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg/data?select=kaggle_2021.tsv
public_hpa_df['ID'] = public_hpa_df['Image'].apply(lambda x: '_'.join(x.split('/')[-2:]))
# Remove all images overlapping with Training set
public_hpa_df = public_hpa_df[public_hpa_df.in_trainset == False]

# Remove all images with only labels that are not in this competition
public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]

colors = ['red', 'green', 'blue', 'yellow']
celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30', 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)]
print(len(public_hpa_df), len(public_hpa_df_17))
print(public_hpa_df.tail())

def save_jpg(idx):
    row = public_hpa_df_17.iloc[idx]
    for color in colors:
        try:
            img_url = f'{row.Image}_{color}.jpg'
            response = requests.get(img_url)
            output_file = f'{save_dir}/{row.ID}_{color}.jpg'
            with open(output_file, "wb") as fp:
                fp.write(response.content)
        except:
            print(f'failed to download: {img_url}\n')

with multiprocessing.Pool(multiprocessing.cpu_count()*2) as p:
    _ = list(tqdm(p.imap(save_jpg, list(range(public_hpa_df_17.shape[0]))), total=public_hpa_df_17.shape[0]))


def try_read_jpg_img(idx):
    row = public_hpa_df_17.iloc[idx]
    ok = True
    for color in colors:
        output_file = f'{save_dir}/{row.ID}_{color}.jpg'
        img = cv2.imread(output_file)
        if img is None:
            ok = False
    return ok

def try_read_png_img(idx):
    row = df_c11.iloc[idx]
    ok = True
    for color in colors:
        output_file = f'{save_dir}/{row.ID}_{color}.png'
        img = cv2.imread(output_file)
        print(img)
        if img is None:
            ok = False
    return ok

with multiprocessing.Pool(multiprocessing.cpu_count()*2) as p:
    res = list(tqdm(p.imap(try_read_jpg_img, list(range(public_hpa_df_17.shape[0]))), total=public_hpa_df_17.shape[0]))

public_hpa_df_17['ok'] = res

df_c11 = public_hpa_df_17[public_hpa_df_17.Label_idx.apply(lambda x: '11' in x.split('|'))].reset_index(drop=True)

df_hard11 = df_c11.loc[~df_c11['ok']]

def save_tif(idx):
    row = df_hard11.iloc[idx]
    for color in colors:
        try:
            img_url = f'{row.Image}_{color}.tif.gz'
            response = requests.get(img_url)
            f = io.BytesIO(response.content)
            tf = gzip.open(f).read()
            img = imageio.imread(tf, 'tiff')
            output_file = f'{save_dir}/{row.ID}_{color}.png'
            imageio.imwrite(output_file, img)
        except:
            print(f'failed to download: {img_url}\n')

for i in tqdm(range(df_hard11.shape[0])):
    save_tif(i)

with multiprocessing.Pool(multiprocessing.cpu_count()*2) as p:
    res = list(tqdm(p.imap(try_read_png_img, list(range(df_c11.shape[0]))), total=df_c11.shape[0]))

df_c11['ok'] = res

train_external = public_hpa_df_17.loc[public_hpa_df_17['ok'],['ID','Label_idx']].append(df_c11.loc[df_c11['ok'],['ID','Label_idx']]).reset_index(drop=True)
train_external.columns = ['ID','Label']
train_external.to_csv(f'{data_dir}/train_external.csv',index=False)
