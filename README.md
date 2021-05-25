# AllDataAreExt-Galixir-Kaggle-HPA-2021-Solution

##  Teammates

  - Qishen Ha is Master of Engineering from the University of Tokyo. Machine Learning Engineer at LINE Corp. Kaggle Competition Grandmaster. Z by HP & NVIDIA Global Data Science Ambassador.

  - Bo is currently a Senior Deep Learning Data Scientist at NVIDIA. Kaggle Competition Grandmaster.

  - Gary is currently a Algorithm Engineer at ReadSense based in the China. Kaggle Competition Grandmaster. Z by HP & NVIDIA Global Data Science Ambassador.

  - daishu is a Researcher at Galixir. Kaggle Competition Grandmaster.

## Methods

![Overview of Methods](https://github.com/jxzly/AllDataAreExt-Galixir-Kaggle-HPA-2021-Solution/blob/main/fig/methods.jpg)

### Image-to-cell augmentation module

We used two methods to train-test this pipeline.

The first one is to train with 512 images, and the test input is also 512. We loop n times for each image (n is the number of cells in the image), leaving only one cell in each time and masking out the other cells to get single cell predictions.

The second one is trained with 768 random crop 512, and then tested almost the same way as the first one, but not only mask out the other cells, but we also put the position of the cells left in the center of the image.

The training process incorporates two augmentation methods designed for this task, in addition to augment by random rotation, flipping, cropping, cutout and adjusting brightness. One is to multiply the data of the green channel (Protein) by a random number in the range of `[0.0,0.1]` with small probability while setting the label to negative to improve the model's ability to recognize negative samples. The other is to set the green channel to red (Microtubules) or yellow (Endoplasmicreticulum) with a small probability and multiply it by a random number in the range of `[0.6,1.0]`, and change the label to the Microtubules or Endoplasmicreticulum.

### pseudo-3D cell augmentation module

We pre-crop all the cells of each image and save them locally. Then during training, for each image we randomly select 16 cells. We then set `bs=32`, so for each batch we have `32x16=512` cells in total.

We resize each cell to `128x128`, so the returned data shape from the dataloader is `(32, 16, 4, 128, 128) `. Next we reshape it into `(512, 4, 128, 128)` and then use a very common CNN to forward it, the output shape is `(512, 19)`.

In the prediction phase we use the predicted average of different augmented images of a cell as the predicted value for each cell. But during the training process, we rereshape this `(512, 19)` prediction back into `(32, 16, 19)` . Then the loss is calculated for each cell with image-level GT label.

### Featurziation with deep neural network

We use multipe CNN variants to train, such as EfficientNet, ResNet, DenseNet.

### Classification

We average the different model predictions from different methods.

## Tree-Structured Directory

  **├── `input`**

  **│   ├──`hpa-512`**: 512-image and 512-cell mask

  **│   │   ├── `test`**

  **│   │   ├── `test_cell_mask`**

  **│   │   ├── `train`**

  **│   │   └── `train_cell_mask`**

  **│   ├── `hpa-seg`** : official segmentation models

  **│   └── `hpa-single-cell-image-classification`** : official data and kaggle_2021.tsv

  **├── `output` ** :  logs, models and submission

## Code

  - **`S1_external_data_download.py`**: download external train data

  - **`S2_data_process.py`**: generate 512-image and 512-cell mask

  - **`S3_train_pipeline1.py`**: train 2d image level models

  - **`S4.1_crop_cells.py`**: crop training cells for 3d models

  - **`S4.2_train_pipeline2`**.py: train 3d cell level models

  - **`S5_predict.py`**: generate submission.csv
