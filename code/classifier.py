from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from pathlib import Path
import glob
import json
import os
from collections import OrderedDict

import rasterio.features
import rasterio

import shapely.wkt
import shapely.ops
import shapely.geometry
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt

import skimage.transform
import skimage.morphology
from skimage.draw import polygon
import skimage.draw
import skimage.transform

import pandas as pd
import numpy as np
import h5py
import tqdm
import click

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from base_classifier import BaseClassifier
from unet_model import UNet
from train_unet import train_net
import eval
from eval import predict_generator
from hdf5_dataset import HDF5Dataset
from hdf5_wrappers import matrix_to_hdf5
import imagelist_dataset
from path_manager import PathManager
from utils import *
import sa_vector_to_spacenet_csv

logger = getLogger("MainLogger")

class UNet_classifier(BaseClassifier):
    def __init__(
            self, datapath, model_name, original_size_x, original_size_y, input_size,
            slice_count_x=1, slice_count_y=1, is_8_channel=True,
            gpu_number=None, working_dir="/data/working", is_8_bit=True):
        BaseClassifier.__init__(
            self, datapath, model_name, original_size_x, original_size_y, input_size,
            slice_count_x, slice_count_y, is_8_channel,
            gpu_number, working_dir, is_8_bit=is_8_bit)
        # Set up the logger.
        logging_handler = FileHandler("{}.log".format(model_name))
        logging_handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(logging_handler)

    def load_mean_image(self):
        """ Loads mean image.
        """
        mean_path = self.path_mgr.get_mean_image_file_path()
        # For RGB 8-bit images we do not substract the mean.
        if mean_path is None:
            return None
        return self.load_mean_image_from_path(mean_path)

    def load_mean_image_from_path(self, mean_path):
        with h5py.File(mean_path, 'r', libver='latest', swmr=True) as f:
            im_mean = np.array(f.get('mean'))
        return im_mean

    def remove_interiors(self, line):
        if "), (" in line:
            line_prefix = line.split('), (')[0]
            line_terminate = line.split('))",')[-1]
            line = (
                line_prefix +
                '))",' +
                line_terminate
            )
        return line

    def get_unet(self, use_distributed_data_parallel=True):
        """ Creates a new network and returns. If machine has multiple GPUs, uses them. 
        """
        net = UNet(n_channels=self.channel_count, n_classes=1, bilinear=True,
                running_on_gpu=(self.gpu_number is not None))
        net.to(self.device)
        if not use_distributed_data_parallel:
            return net
        net = nn.parallel.DistributedDataParallel(net, device_ids=[self.gpu_number])
        return net

    def load_model(self, epoch=None):
        """ Loads U-Net model for given epoch, or best epoch based on performance on Validation set.
        Args:
            epoch (int) - Number of epoch model of which to load.
        """
        # Load model weights
        # Predict and Save prediction result
        if epoch is None:
            model_weights_path = self.path_mgr.get_last_weights_path()
        else:
            model_weights_path = self.path_mgr.get_weights_path_for_epoch(epoch)
        return self.load_model_from_path(model_weights_path)

    def load_model_from_path(self, model_weights_path, use_distributed_data_parallel=True):
        net = self.get_unet(use_distributed_data_parallel)
        net.to(device=self.device)
        logger.info("Loading model weights from {}".format(model_weights_path))
        if use_distributed_data_parallel:
            net.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        else:
            # When loading to CPU, we need to change layer names, remove the "module." from the start.
            state_dict = torch.load(model_weights_path, map_location=self.device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
        return net

    def build_dataset_from_images(self, image_file_paths):
        """ Pytorch dataset for given [large]images.
        """
        X_mean = self.load_mean_image()
        dataset = imagelist_dataset.ImageListDataset(
            image_file_paths, mean_image=X_mean,
            bandstats_file_path=self.path_mgr.get_bandstats_file_path(),
            original_size_x=self.original_size_x, original_size_y=self.original_size_y,
            input_size=self.input_size, slice_count_x=self.slice_count_x,
            slice_count_y=self.slice_count_y, is_8_channel=self.is_8_channel)
        return dataset

    def predict(self, image_file_paths, model_weights_path, batch_size=8):
        """ Runs the model on the given set of images, batch_size images at a time.
        Args:
            batch_size (int) - Number of images to run on per batch. If image is sliced, will run on all the slices for this many images.
        """
        net = self.load_model_from_path(model_weights_path, use_distributed_data_parallel=False) 
        dataset = self.build_dataset_from_images(image_file_paths)
        predict_gen = eval.predict_generator_test(
            net, dataset, batch_size * self.slice_count, self.device)
        
        # Predict and Save prediction result
        for idx, y_pred in enumerate(predict_gen):
            current_image_paths = image_file_paths[idx*batch_size:idx*batch_size+batch_size]
            yield (current_image_paths, y_pred)
        del net

    def mask_to_file(self, pred_values, image_id):
        import imageio
        # Write the mask, using this for Inria dataset challenge.
        create_dir_if_doesnt_exist("masks_inria_new")
        mask_path = os.path.join("masks_inria_new", os.path.basename(image_id))
        logger.info("Saving image mask to {}".format(mask_path))
        mask = ((pred_values > 0.5).astype(np.uint8)) * 255
        # Remove very small blobs from the mask to reduce file size.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
        mask2 = np.zeros((output.shape), dtype=np.uint8)
        for i in range(1, nb_components):
            if stats[i, -1] >= 60: # Remove small noise of <60 px.
                mask2[output == i] = 255
        imageio.imwrite(mask_path, mask2)

    def run_predictions_on_images(
            self, image_file_paths, output_file_path, model_weights_path, 
            batch_size=8, min_th = 60, enable_tqdm=False):
        with open(output_file_path, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            y_pred_generator = self.predict(
                image_file_paths, model_weights_path, batch_size=batch_size)
            if enable_tqdm:
                y_pred_generator = tqdm.tqdm(y_pred_generator,
                                             total=len(image_file_paths) / batch_size)
            for (image_ids, y_pred) in y_pred_generator:
                for (idx, image_id) in enumerate(image_ids):
                    pred_values = self.get_prediction_mask_for_image(
                        idx, image_id, y_pred)
                    # Uncomment next line if you want to store the semgentation masks.
                    # self.mask_to_file(pred_values, image_id)
                    self.write_resulting_csv_prediction(
                        f, image_id, pred_values, min_polygon_area_th=min_th)

    def get_prediction_mask_for_image(self, idx, image_id, y_pred):
        """ Returns prediction matrix for given test image.
        Args:
            idx - sequential number of the image in this particular batch.
            image_id - name of the image, unused but intentionally left here.
            y_pred - Predictions for some list of images.
        """
        if self.slice_count == 1:
            return y_pred[idx][0]
        pred_values = np.zeros((self.original_size_x, self.original_size_y))
        # If an image is sliced, the slices will overlap in some areas,
        # so we must take an average prediction for those pixels.
        pred_count = np.zeros((self.original_size_x, self.original_size_y))
        for pos_i in range(self.slice_count_x):
            for pos_j in range(self.slice_count_y):
                slice_idx = idx * self.slice_count_x * self.slice_count_y + \
                    pos_i * self.slice_count_x + pos_j
                x0 = self.stride_size_x * pos_i
                y0 = self.stride_size_y * pos_j
                pred_values[x0:x0+self.input_size, y0:y0+self.input_size] += (
                    y_pred[slice_idx][0]
                )
                pred_count[x0:x0+self.input_size, y0:y0+self.input_size] += 1
        pred_values = pred_values / pred_count
        return pred_values

    def write_resulting_csv_prediction(
            self, csv_file, image_id, pred_values_mask, min_polygon_area_th):
        """ Takes a mask of predictions, creates polygons for it and writes to csv file.
        Args:
            csv_file - Output csv file, not the path, the opened file itself.
            pred_values_mask - A mask matrix.
            min_polygon_area_th - Minimal size of a polygon to write, used to skip noise.
        """
        for line in self.get_resulting_csv_prediction(
                image_id, pred_values_mask, min_polygon_area_th):
            csv_file.write(line)

    def get_resulting_csv_prediction(
            self, image_id, pred_values_mask, min_polygon_area_th):
        df_poly = self.mask_to_poly(pred_values_mask, min_polygon_area_th=min_polygon_area_th)
        if len(df_poly) > 0:
            for i, row in df_poly.iterrows():
                line = "{},{},\"{}\",{:.6f}\n".format(
                    image_id,
                    row.bid,
                    row.wkt,
                    1.0) # Confidence percent
                    #row.area_ratio)
                line = self.remove_interiors(line)
                yield line
        else:
            # Martun: looks like we want to have 1 line even for images with no buildings
            # in the output csv file.
            line = "{},{},{},0\n".format(
                image_id,
                -1,
                "POLYGON EMPTY")
            yield line

    def get_validation_dataset(self):
        X_mean = self.load_mean_image()
        slice_ids=image_ids_to_slice_ids(self.load_image_id_list(split=DataSplit.Validation),
                                         self.slice_count)

        # Martun: Validation runs on a single GPU, and is very slow compared to training speed.
        # let's use up to 600 slices for validation for now.
        # slice_ids = slice_ids[:300]
        validation_dataset = HDF5Dataset(
            image_file_path=self.path_mgr.get_image_store_path(DataSplit.Validation),
            mask_file_path=self.path_mgr.get_mask_store_path(is_training=False),
            slice_ids=slice_ids,
            mean_image=X_mean)
        return validation_dataset

    def predict_on_validation_set_generator(
            self, epoch=None, batch_size=8, save_pred=True, enable_tqdm=False, world_size=1, rank=0):
        """ Runs predictions on the validaction set, using saved weights from the given epoch.
            For sliced images does NOT take care of the slices, the caller must merge them himself.
        Args:
            epoch (int) - Number of epoch weights of which to load. If None, will load the last one. 
            batch_size (int) - size of a batch.
            save_pred (bool) - If True, saves predictions, otherwise just yeilds them.
        Yields:
            (image_ids, y_pred) where image_ids is a list of image ids, and y_pred is prediction masks for [slices] for given images.
        """
        net = self.load_model(epoch=epoch)
        validation_dataset = self.get_validation_dataset()
        logger.info("validation dataset size is {} slices".format(str(len(validation_dataset))))
        predict_gen = predict_generator(
            net, validation_dataset, batch_size * self.slice_count, self.device, world_size, rank)

        all_image_ids = self.load_image_id_list(split=DataSplit.Validation)
        # Predict and Save prediction result
        for idx, y_pred in enumerate(predict_gen):
            image_ids = all_image_ids[idx*batch_size:idx*batch_size+batch_size] 
            # Save prediction result
            if save_pred:
                self.save_predictions(image_ids, y_pred, split=DataSplit.Validation)
            yield (image_ids, y_pred)

    def save_predictions(self, image_ids, y_pred, split):
        """ Saves prediction mask results.
        Args:
            image_ids - Ids of images predictions of which are provided. Not Slice ids.
        """ 
        # Copy y_pred to make sure we don't write in parallel.
        # Need to copy y_pred before writing, otherwise somehow hdf5 crashes.
        y_pred_copy = np.copy(y_pred)

        file_out = self.path_mgr.get_predictions_output_path(split)
        image_or_slice_ids = image_ids_to_slice_ids(image_ids, self.slice_count)
        with h5py.File(file_out, 'w') as f:
            for (slice_id, pred) in zip(image_or_slice_ids, y_pred_copy):
                matrix_to_hdf5(f, pred, slice_id)

    def predict_on_validation_set(
            self, epoch=None, min_th=30, 
            enable_tqdm=False,
            store_prediction_in_file=False,
            world_size=1, rank=0):
        """ Runs prediction on validation set, or loads them from the file.
            Creates polygons csv file with final predictions. 
        Args:
            epoch (int) - Epoch weights of which will be used.
            slices_number (int) - Number of slices for the image, 1 if no slices, 9 if 9 slices.
            min_th (int) - Minimal area of a polygon to consider. Used to remove too small objects.
            enable_tqdm (bool) - Show progress bar or not.
            store_prediction_in_file (bool) - Used only if load_predictions_from_file==False. If set to True, the newly created predictions will be saved to the file. 
        Returns: Nothing.
        """
        # Run Prediction phase on validation set.
        logger.info("Prediction phase")
        batch_size=6
        # Increase batch size if we have multiple GPUs.
        #if torch.cuda.device_count() > 1:
        #    batch_size *= torch.cuda.device_count()
        y_pred_generator = self.predict_on_validation_set_generator(
            epoch=epoch,
            batch_size=batch_size,
            save_pred=False,
            enable_tqdm=enable_tqdm,
            world_size=world_size,
            rank=rank)
    
        # Postprocessing phase
        logger.info("Postprocessing phase, converting masks to polygons and saving.")

        fn_out = self.path_mgr.get_predicted_polygons_csv_file_path(split=DataSplit.Validation)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            for (image_ids, y_pred) in y_pred_generator:
                for (idx, image_id) in enumerate(image_ids):
                    prediction_mask = self.get_prediction_mask_for_image(
                        idx, image_id, y_pred)
                    # Now we have many processes, which will write to the file in parallel.
                    self.write_resulting_csv_prediction(
                        f, image_id, prediction_mask, min_polygon_area_th=min_th)
   

    def mask_to_poly(self, mask, min_polygon_area_th=30, resize_to_original_size=True):
        """ Convert from mask to polygons on (self.original_size_x, self.original_size_y) image.
            If mask has smaller dimensions, resize to the output dimensions self.original_size_x=650.
        """
        if mask.shape != (self.original_size_x, self.original_size_y) and resize_to_original_size:
            mask = skimage.transform.resize(mask, (self.original_size_x, self.original_size_y))
        # Now threshold on 0.5.
        mask = (mask > 0.5).astype(np.uint8)
        shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
        poly_list = []
        mp = shapely.ops.cascaded_union(
            shapely.geometry.MultiPolygon([
                shapely.geometry.shape(shape)
                for shape, value in shapes
            ]))
        max_points = 20
        mp = [shapely.geometry.Polygon(reduce_polygon_points(np.array(polygon.exterior.coords).astype(np.int32), max_points)) for polygon in mp]
        if isinstance(mp, shapely.geometry.Polygon):
            df = pd.DataFrame({
                'area_size': [mp.area],
                'poly': [mp],
            })
        else:
            df = pd.DataFrame({
                'area_size': [p.area for p in mp],
                'poly': [p for p in mp],
            })
    
        df = df[df.area_size > min_polygon_area_th].sort_values(
            by='area_size', ascending=False)
        df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
            x, rounding_precision=0))
        df.loc[:, 'bid'] = list(range(1, len(df) + 1))
        df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
        return df

    def train(self, batch_size=32, epoch_count=60, world_size=1, rank=0, load_epoch=None):
        """ This is the main function to train the classifier.
        """
        logger.info(">> Starting training on GPU number {}.".format(rank))
    
        # Increase batch size if we have multiple GPUs.
        #if torch.cuda.device_count() > 1:
        #    batch_size *= torch.cuda.device_count()
        if load_epoch is not None:
            net = self.load_model(epoch=load_epoch)
        else:
            net = self.get_unet()
        X_mean = self.load_mean_image()
        # Shuffle training images here, this allows for caching.
        training_slice_ids = image_ids_to_slice_ids(
            self.load_image_id_list(split=DataSplit.Train), self.slice_count)
        # Uncomment next line is good for debugging/testing. 
        # training_slice_ids = training_slice_ids[:300]

        if rank == 0:
            logger.info("Training on data from {}".format(
                self.path_mgr.get_image_store_path(DataSplit.Train)))
            logger.info("Training on ground truth masks from {}".format(
                self.path_mgr.get_mask_store_path(is_training=True)))
        training_dataset = HDF5Dataset(
            image_file_path=self.path_mgr.get_image_store_path(DataSplit.Train),
            mask_file_path=self.path_mgr.get_mask_store_path(is_training=True),
            slice_ids=training_slice_ids,
            mean_image=X_mean)

        validation_dataset = self.get_validation_dataset()
        if rank == 0:
            logger.info("Validating on {}".format(
                self.path_mgr.get_image_store_path(DataSplit.Validation)))
            logger.info("validation dataset size is {} slices".format(str(len(validation_dataset))))

        train_net(
            net,
            training_dataset,
            validation_dataset,
            optimizer=optim.SGD(net.parameters(), lr=0.01, momentum=0.9, 
                nesterov=True, weight_decay=0),
            device=self.device,
            epochs=epoch_count,
            batch_size=batch_size,
            save_after_each_epoch=True,
            model_file_path_format=self.path_mgr.get_weights_path_for_epoch_format(),
            world_size=world_size,
            rank=rank)
                  
        # (martun) Saves the last model, fully trained one. Probably will be the copy of 
        # last model saved by the 'ModelCheckpoint' callback.
        torch.save(net.state_dict(), self.path_mgr.get_last_weights_path())
    
        # Save evaluation history
        logger.info(">> Training Done")


@click.group()
def cli():
    pass
    
    
def train_body(gpu_number, args):
    logger.addHandler(handler)
    rank = args['node_rank'] * args['gpus'] + gpu_number
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=args['world_size'],
    	rank=rank                                               
    )
    # Next line is important, otherwise processes can not sync.
    torch.manual_seed(0)

    classifier = UNet_classifier(
            args['datapath'], model_name="model_2",
            original_size_x=650, original_size_y=650, input_size=256,
            slice_count_x=3, slice_count_y=3, is_8_channel=False,
            gpu_number=gpu_number, working_dir=args['working_dir'])
    # Set load_epoch to some number, if we want to continue the training.
    classifier.train(batch_size=32, epoch_count=60, world_size=args['world_size'],
            rank=rank, load_epoch=None)


@cli.command()
@click.argument('datapath', type=str)
@click.argument('working_dir', type=str)
def train(datapath, working_dir):
    args = {'gpus': torch.cuda.device_count(), 'nodes': 1}
    args['node_rank'] = 0 # We have just 1 node for now.
    args['world_size'] = args['gpus'] * args['nodes']
    args['datapath'] = datapath
    args['working_dir'] = working_dir
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train_body, nprocs=args['gpus'], args=(args,))
       

@cli.command()
@click.argument('images_folder_path', type=str)
@click.argument('model_weights_path', type=str)
@click.argument('mean_image_path', type=str)
@click.argument('bandstats_file_path', type=str)
@click.argument('output_file_path', type=str)
def predict_multiple(images_folder_path, model_weights_path, mean_image_path, bandstats_file_path,
            output_file_path):
    """ Sample call:
        python3 classifier.py predict sample_images /data/working/ALL_IN_ONE/model_weights/model_2/weights_for_epoch_29.h5 /data/working/ALL_IN_ONE/preprocessed/mean_image_store_3_channels_9_slices.h5 /data/working/ALL_IN_ONE/preprocessed/bandstats_3_channels.csv prediction_results.csv 8192 8192
    """
    image_paths = glob.glob(images_folder_path + "/*.tif")
    image_0 = load_raster_image(image_paths[0])
    original_size_x = image_0.shape[1]
    original_size_y = image_0.shape[2]
 
    # We want the slices to overlap by at least 64 pixels.
    slice_count_x = original_size_x // (256 - 64)
    slice_count_y = original_size_y // (256 - 64)
    classifier = UNet_classifier(
        None, model_name="model_2",
        original_size_x=original_size_x, original_size_y=original_size_y, input_size=256,
        slice_count_x=slice_count_x, slice_count_y=slice_count_y, is_8_channel=True,
        gpu_number=0)
    classifier.path_mgr.bandstats_file = bandstats_file_path
    classifier.path_mgr.mean_image_path = mean_image_path
    classifier.run_predictions_on_images(
        image_paths, output_file_path, model_weights_path,
        batch_size=8, min_th = 60, enable_tqdm=True)


def predict_body(image_path, model_weights_path, mean_image_path, bandstats_file_path, output_json_path, gpu_number=None):
    """ Sample call:
        python3 classifier.py predict sample_images/15OCT22183656-S2AS_R5C4-056155973040_01_P001_8_bit.jpg /data/working/ALL_IN_ONE/model_weights/model_2/weights_for_epoch_29.h5 /data/working/ALL_IN_ONE/preprocessed/mean_image_store_3_channels_9_slices.h5 bandstats_3_channels_8_bits.csv annotation.json
    """
    image = load_raster_image(image_path)
    original_size_x = image.shape[1]
    original_size_y = image.shape[2]
    
    logger.info("Running on image of size {}x{}".format(original_size_x, original_size_y))
    # We want the slices to overlap by at least 64 pixels.
    slice_count_x = original_size_x // (256 - 64)
    slice_count_y = original_size_y // (256 - 64)
    classifier = UNet_classifier(
        None, model_name="model_2",
        original_size_x=original_size_x, original_size_y=original_size_y, input_size=256,
        slice_count_x=slice_count_x, slice_count_y=slice_count_y, is_8_channel=False,
        gpu_number=gpu_number) 
    classifier.path_mgr.bandstats_file = bandstats_file_path
    classifier.path_mgr.mean_image_path = mean_image_path
    output_csv_path = output_json_path + ".csv"
    classifier.run_predictions_on_images(
        [image_path], output_csv_path, model_weights_path,
        batch_size=1, min_th = 60, enable_tqdm=True)
    sa_vector_to_spacenet_csv.spacenet_csv_to_sa_vector_file(output_csv_path, output_json_path)


@cli.command()
@click.argument('image_path', type=str)
@click.argument('model_weights_path', type=str)
@click.argument('mean_image_path', type=str)
@click.argument('bandstats_file_path', type=str)
@click.argument('output_json_path', type=str)
def predict(image_path, model_weights_path, mean_image_path, bandstats_file_path, output_json_path):
    predict_body(image_path, model_weights_path, mean_image_path,
            bandstats_file_path, output_json_path)


@cli.command()
@click.argument('image_path', type=str)
@click.argument('model_weights_path', type=str)
@click.argument('output_json_path', type=str)
def predict_8_bit_rgb(image_path, model_weights_path, output_json_path):
    """ Sample call:
        python3 classifier.py predict-8-bit-rgb sample_images/15OCT22183656-S2AS_R5C4-056155973040_01_P001_8_bit.jpg /data/working/ALL_IN_ONE/model_weights/model_2/weights_for_epoch_29.h5 annotation.json
    """
    gpu_number = 0 if torch.cuda.is_available() else None
    predict_body(image_path, model_weights_path, None, None, output_json_path, gpu_number)


@cli.command()
@click.argument('data_path', type=str)
@click.argument('model_weights_path', type=str)
@click.argument('output_folder_path', type=str)
def predict_8_bit_rgb_folder(data_path, model_weights_path, output_folder_path):
    """ Sample call:
        python3 classifier.py predict-8-bit-rgb-folder sample_images /data/working/ALL_IN_ONE/model_weights/model_2/weights_for_epoch_29.h5 annotations
    """
    image_paths = glob.glob(data_path + "/*.tif")
    create_dir_if_doesnt_exist(output_folder_path)
    gpu_number = 0 if torch.cuda.is_available() else None
    for image in image_paths:
        output_json_path = os.path.join(output_folder_path, "".join([os.path.basename(image), "___objects.json"]))
        predict_body(image, model_weights_path, None, None, output_json_path, gpu_number)



if __name__ == '__main__':
    cli()


