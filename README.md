# Building Detection from Aerial Images.

U-Net based building detection. Creates semantic segmentation map and cuts into polygons. Based on [the winning solution of Spacenet Building Detection](https://github.com/SpaceNetChallenge/BuildingDetectors_Round2/tree/master/1-XD_XD).

We merge datasets of 3 cities(Vegas, Paris and Shanghai) to build 1 model which will generalize well on other datasets. We also don't use any ensembling or OpenStreetMap layers as in the competition. We were able to get validation accuracy of 0.545 on the validation set using 8-bit RGB images only.  

Our code uses a modified version of [pytorch implementation of U-Net](https://github.com/milesial/Pytorch-UNet).

## Getting Started

Create a Docker image using the Dockerfile in the repository. Alternatively take a look into it for the prerequisites.

## Downloading the Data

We use Spacenet dataset for this project. To download run the following commands:
```
aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs . --recursive
```
We do not use Khartoum city annotations due to lower quality of annotations.

Unzip all the tar.gz files, and merge all the images into one folder. We used RGB images, so we merged all the RGB-PanSharpen folders into 1. You also need to merge all the summaryData csv files into 1.

## Running Preprocessing

Our code read the images from a given directory, splits them into training and validation sets, cuts them into slices of 256x256 pixels and saves into hdf5 files. Also ground truth masks and sliced accordingly. In order to run this preprocessing use the following command:
```
python3 preprocessor.py preprocess-spacenet-rgb path_to_dataset
```

If you are using some other dataset(8-channel 16 bit images), different from spacenet use the following command:

```
python3 preprocessor.py preprocess-multichannel-training-data images_folder_path ground_truth_csv_path output_directory_path
```

## Running Training
To run the training use following command:
```
python3 classifier.py train datapath working_dir
```
where datapath is path to the initial data and working_dir is the directory with preprocessed hdf5 files.

## Running Inference
Running inference on an 8-bit RGB image:
```
python3 classifier.py predict-8-bit-rgb image_path model_weights_path output_json_path
Sample call:
python3 classifier.py predict-8-bit-rgb sample_images/15OCT22183656-S2AS_R5C4-056155973040_01_P001_8_bit.jpg /data/working/ALL_IN_ONE/model_weights/model_2/weights_for_epoch_29.h5 annotation.json
```
Running inference on multiple 8-bit RGB images in the same folder:
```
python3 classifier.py predict-8-bit-rgb-folder data_folder_path, model_weights_path, output_folder_path
Sample call:
ython3 classifier.py predict-8-bit-rgb-folder sample_images /data/working/ALL_IN_ONE/model_weights/model_2/weights_for_epoch_29.h5 annotations_folder
```

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details

