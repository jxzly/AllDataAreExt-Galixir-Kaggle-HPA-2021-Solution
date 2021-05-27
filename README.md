# AllDataAreExt-Galixir-Kaggle-HPA-2021-Solution

##  Team Members

  - Qishen Ha is Master of Engineering from the University of Tokyo. Machine Learning Engineer at LINE Corp. Kaggle Competition Grandmaster. Z by HP & NVIDIA Global Data Science Ambassador.

  - Bo Liu is currently a Senior Deep Learning Data Scientist at NVIDIA based in the U.S. and a Kaggle Competition Grandmaster.

  - Fuxu Liu is currently a Algorithm Engineer at ReadSense based in the China. Kaggle Competition Grandmaster. Z by HP & NVIDIA Global Data Science Ambassador.

  - Daishu is currently a Senior Research Scientist at Galixir. Kaggle Competition Grandmaster.

## Methods

![Overview of Methods](https://github.com/jxzly/AllDataAreExt-Galixir-Kaggle-HPA-2021-Solution/blob/main/fig/methods.jpg)

### Image-to-cell augmentation module

We used two methods to train and make predictions in our pipeline.

Firstly, we use 512 x 512 image size to train and test. For predicting, we loop n times for each image (n is the number of cells in the image), leaving only one cell in each time and masking out the other cells to get single cell predictions.

The second method is trained with 768 x 786 images with random crop to 512 x 512 then tested almost the same way as our first approach. Specifically, we not only mask out the other cells but reposition of the cells in the left to the center of the image as well.

The two methods share the same training process, in which we incorporate two augmentation approach specifically designed for this task, in addition to regular augmentation methods such as random rotation, flipping, cropping, cutout and brightness adjusting.
The first augmentation approach is, with a small probability, multiplying the data of the green channel (protein) by a random number in the range of [0.0,0.1] while setting the label to negative to improve the model's ability to recognize negative samples.
The other augmentation approach is, with a small probability, setting the green channel to red (Microtubules) or yellow (Endoplasmicreticulum), multiplying it by a random number in the range of [0.6,1.0] and changing the label to the Microtubules or Endoplasmicreticulum.

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

  **├── `output`** :  logs, models and submission

## Code

  - **`S1_external_data_download.py`**: download external train data

  - **`S2_data_process.py`**: generate 512-image and 512-cell mask

  - **`S3_train_pipeline1.py`**: train image-to-cell augmentation module

  - **`S4.1_crop_cells.py`**: crop training cells for pseudo-3D cell augmentation module

  - **`S4.2_train_pipeline2.py`**: train pseudo-3D cell augmentation module

  - **`S5_predict.py`**: generate submission.csv
