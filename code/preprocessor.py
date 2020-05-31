# -*- coding: utf-8 -*-
"""
Image preprocessing module
"""
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from pathlib import Path
import click
import tqdm
import h5py
import pandas as pd
import numpy as np
import skimage
import skimage.draw
import shapely.wkt

from base_classifier import BaseClassifier
from hdf5_wrappers import matrix_to_hdf5
from hdf5_wrappers import load_images_from_hdf5_file
from utils import *


logger = getLogger("MainLogger")

"Reads images in tiff format and creates a bunch of hdf5 files."
class Preprocessor(BaseClassifier):
    def __init__(
            self, datapath, model_name='preprocessor', original_size_x=650, original_size_y=650,
            input_size=256, slice_count_x=1, slice_count_y=1, is_8_channel=True,
            working_dir="/data/working", is_8_bit=True):
        BaseClassifier.__init__(
            self, datapath, model_name, original_size_x, original_size_y, input_size,
            slice_count_x, slice_count_y, is_8_channel,
            gpu_number=None, working_dir=working_dir, is_8_bit=is_8_bit)
        # Split 70-30 train/validation.
        self.training_data_percentage = 0.7

        # Set up logging to file.
        logging_handler = FileHandler("{}.log".format(model_name))
        logging_handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(logging_handler)

    def load_bandstats(self):
        # For 8 bit images no need to change.
        if self.is_8_bit:
            return {k: dict(max=255, min=0) for k in range(self.channel_count)}
            
        fn_stats = self.path_mgr.get_bandstats_file_path()
        df_stats = pd.read_csv(fn_stats, index_col='prefix')
        r = df_stats.loc[self.prefix]

        stats_dict = {}
        for chan_i in range(self.channel_count):
            stats_dict[chan_i] = dict(
                min=r['chan{}_min'.format(chan_i)],
                max=r['chan{}_max'.format(chan_i)])
        return stats_dict

    def load_ground_truth_csv_data(self):
        fn = self.path_mgr.get_training_summary_path()
        df = pd.read_csv(fn)
        return df

    def load_ground_truth_masks_generator(self, split, batch_size=32, image_or_slices_list=None):
        if not image_or_slices_list:
            image_or_slices_list = image_ids_to_slice_ids(
                self.load_image_id_list(split=split), self.slice_count)
        total_sz = len(image_or_slices_list)
        n_batch = int(total_sz // batch_size + 1)
        for i_batch in range(n_batch):
            image_ids_to_load = image_id_list[i_batch*batch_size:i_batch*batch_size+batch_size] 
            yield self.load_ground_truth_masks(split, image_ids_to_load)

    def load_ground_truth_masks(self, split, image_or_slices_list=None):
        y_val = []
        fn_mask = self.path_mgr.get_mask_store_path(is_training=(split==DataSplit.Train))
        # logger.info("Loading ground truth masks from file {}.".format(fn_mask))
        if not image_or_slices_list:
            image_or_slices_list = image_ids_to_slice_ids(
                self.load_image_id_list(split=split), self.slice_count)
        with h5py.File(fn_mask, 'r', libver='latest', swmr=True) as f:
            for image_id in image_or_slices_list:
                mask = np.array(f.get(image_id))
                mask = (mask > 0.5).astype(np.uint8)
                #logger.info("Loaded a mask for image {} with {} filled pixels".format(
                #    image_id, str(np.sum(mask))))
                y_val.append(mask)
        y_val = np.array(y_val)
        y_val = y_val.reshape((-1, 1, self.input_size, self.input_size))

        return y_val

    def get_image_count(self, split):
        """ Returns total number of images or slices for given set.
        """
        return len(self.load_image_id_list(split))

    def get_slice_count(self, split):
        """ Returns total number of images or slices for given set.
        """
        image_ids = self.load_image_id_list(split)
        image_or_slices_list = image_ids_to_slice_ids(image_ids, self.slice_count)
        return len(image_or_slices_list)

    def compute_band_cut_threshold(self):
        # For 8 bit images no need to change.
        if self.is_8_bit:
            return {k: dict(max=255, min=0) for k in range(self.channel_count)}
        band_values = {k: [] for k in range(self.channel_count)}
        band_cut_th = {k: dict(max=0, min=0) for k in range(self.channel_count)}
    
        image_id_list = self.load_image_id_list(split=DataSplit.Train)
        for image_id in tqdm.tqdm(image_id_list[:200]):
            image_fn = self.path_mgr.get_train_image_path_from_imageid(image_id)
            values = load_raster_image(image_fn)
            for i_chan in range(self.channel_count):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)
    
        image_id_list = self.load_image_id_list(split=DataSplit.Validation)
        # Martun: Here skipping most of images, 200 is enough.
        for image_id in image_id_list[:200]:
            image_fn = self.path_mgr.get_train_image_path_from_imageid(image_id)
            values = load_raster_image(image_fn)
            for i_chan in range(self.channel_count):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)
    
        logger.info("Calc percentile point ...")
        for i_chan in range(self.channel_count):
            band_values[i_chan] = np.concatenate(
                band_values[i_chan]).ravel()
            band_cut_th[i_chan]['max'] = np.percentile(
                band_values[i_chan], 98)
            band_cut_th[i_chan]['min'] = np.percentile(
                band_values[i_chan], 2)
        return band_cut_th
 
    def calc_band_cut_thresholds(self):
        band_cut_th = self.compute_band_cut_threshold()
        rows = []
        row = dict(prefix=self.prefix)
        row['prefix'] = self.prefix
        for chan_i in band_cut_th.keys():
            row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
            row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
        rows.append(row)
        return rows
    
    def save_cut_threshold(self, rows):
       pd.DataFrame(rows).to_csv(self.path_mgr.get_bandstats_file_path(), index=False)

    def normalize_image(self, image):
        bandstats = self.load_bandstats() 
        for chan_i in range(self.channel_count):
            min_val = bandstats[chan_i]['min']
            max_val = bandstats[chan_i]['max']
            image[chan_i] = np.clip(image[chan_i], min_val, max_val)
            image[chan_i] = (image[chan_i] - min_val) / (max_val - min_val)

        image = np.swapaxes(image, 0, 2) # -> (h, w, ch)
        image = np.swapaxes(image, 0, 1) # -> (w, h, ch)
        return image

    def load_image_ids_from_summary_file(self):
        df = self.load_ground_truth_csv_data()
        df_agg = df.groupby('ImageId').agg('first')
        return df_agg.index.tolist()
   
    def split_image_ids_train_validation(self):
        image_id_list = self.load_image_ids_from_summary_file()
        np.random.shuffle(image_id_list)
        training_image_count = int(len(image_id_list) * self.training_data_percentage)
        validation_image_count = len(image_id_list) - training_image_count
    
        base_dir = Path(self.path_mgr.get_image_list_csv_path(split=DataSplit.Train)).parent
        if not base_dir.exists():
            base_dir.mkdir(parents=True)
    
        pd.DataFrame({'ImageId': image_id_list[:training_image_count]}).to_csv(
            self.path_mgr.get_image_list_csv_path(split=DataSplit.Train),
            index=False)
        pd.DataFrame({'ImageId': image_id_list[training_image_count:]}).to_csv(
            self.path_mgr.get_image_list_csv_path(split=DataSplit.Validation),
            index=False)
    
    def image_mask_from_summary(self, df, image_id):
        """ Creates and returns an image mask for ground truth.
        Args:
            df (string) - CSV data file path containing the polygons.
            image_id (string) - name of the image
        """
        im_mask = self.image_mask_from_polygons(self.get_image_polygons(df, image_id))
        im_mask = (im_mask > 0.5).astype(np.uint8)
        return im_mask

    def image_mask_from_polygons(self, polygons):
        """ Draws an image mask based on polygons in original size.
        """
        im_mask = np.zeros((self.original_size_x, self.original_size_y))
        for idx, row in polygons:
            shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
            if shape_obj.exterior is not None:
                coords = list(shape_obj.exterior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                # Looks like we have some empty polygons.
                if not x:
                    continue 
                yy, xx = skimage.draw.polygon(y, x, (self.original_size_x, self.original_size_y))
                im_mask[yy, xx] = 1

                interiors = shape_obj.interiors
                for interior in interiors:
                    coords = list(interior.coords)
                    x = [round(float(pp[0])) for pp in coords]
                    y = [round(float(pp[1])) for pp in coords]
                    yy, xx = skimage.draw.polygon(y, x, (self.original_size_x, self.original_size_y))
                    im_mask[yy, xx] = 0
        return im_mask

    # Loads image polygons from a summary csv data file 'df'.
    def get_image_polygons(self, df, image_id):
        if len(df[df.ImageId == image_id]) == 0:
            raise RuntimeError("ImageId not found on summaryData: {}".format(
                image_id))
        return df[df.ImageId == image_id].iterrows()

    def get_image_slice_gt_mask(self, df, image_id):
        im_mask = self.image_mask_from_summary(df, image_id)
        if self.slice_count == 1:
            yield image_id, im_mask
        else:
            yield from self.slice_image(
                im_mask, image_id=image_id)
    
    def prep_image_mask(self, split):
        fn_mask = self.path_mgr.get_mask_store_path(split==DataSplit.Train)
        if Path(fn_mask).exists():
            logger.info("Generate MASK {} ... skip".format(split.value))
            return
        logger.info("Generate MASK {} for {}".format(split.value, self.prefix))
  
        df_summary = self.load_ground_truth_csv_data()
        logger.info("Preparing image container: {}".format(fn_mask))
        with h5py.File(fn_mask, 'w', libver='latest') as f:
            image_id_list = self.load_image_id_list(split)
            for image_id in tqdm.tqdm(image_id_list):
                for slice_id, im_mask in self.get_image_slice_gt_mask(df_summary, image_id):
                    matrix_to_hdf5(f, im_mask, slice_id)

    def slice_image(self, image, image_id):
        for pos_i in range(self.slice_count_x):
            for pos_j in range(self.slice_count_y):
                slice_pos = pos_i * self.slice_count_x + pos_j
                x0 = self.stride_size_x * pos_i
                y0 = self.stride_size_y * pos_j
                image_part = image[x0:x0+self.input_size, y0:y0+self.input_size]
                yield '{}_{}'.format(image_id, slice_pos), image_part
 
    def read_and_slice_image(self, image_id, split):
        image_path = self.path_mgr.get_image_path_from_imageid(image_id, split)
        yield from self.read_and_slice_image_from_path(image_path, image_id)

    def read_and_slice_image_from_path(self, image_path, image_id):
        image = load_raster_image(image_path)
        image = self.normalize_image(image)

        # Sometimes we don't slice.
        if self.slice_count == 1:
            yield image_id, image
        else:
            yield from self.slice_image(image, image_id=image_id)

    def prepare_image_store(self, split):
        fn_store = self.path_mgr.get_image_store_path(split)
        if Path(fn_store).exists():
            logger.info("Generating IMAGE STORE {} at path {} ... skip".format(
                split.name, fn_store))
            return
        logger.info("Generating IMAGE STORE {} for {}".format(split.name, self.prefix))

        logger.info("Image store file: {}".format(fn_store))
        with h5py.File(fn_store, 'w') as f:
            image_ids = self.load_image_id_list(split)
            for image_id in tqdm.tqdm(image_ids):
                for image_slice_id, im in self.read_and_slice_image(image_id, split):
                    matrix_to_hdf5(f, im, image_slice_id)
  
    def preprocess_data(self):
        """ train.sh """
        logger.info("Preproc for training on {}".format(self.prefix))
    
        # Imagelist
        if (Path(self.path_mgr.get_image_list_csv_path(split=DataSplit.Train)).exists() and 
            Path(self.path_mgr.get_image_list_csv_path(split=DataSplit.Validation)).exists()):
            logger.info("Generate IMAGELIST csv ... skip")
        else:
            logger.info("Generate IMAGELIST csv")
            self.split_image_ids_train_validation()
   
        # Band stats (MUL)
        bandstats_path = self.path_mgr.get_bandstats_file_path()
        if Path(bandstats_path).exists():
            logger.info("Generate band stats csv ... skip")
        else:
            logger.info("Generate band stats csv")
            thresholds = self.calc_band_cut_thresholds()
            self.save_cut_threshold(thresholds)
        self.prepare_masks_and_image_stores()
    
    def prepare_masks_and_image_stores(self):
        # Mask (Target output)
        self.prep_image_mask(DataSplit.Train)
        self.prep_image_mask(DataSplit.Validation)
    
        # Image HDF5 store (MUL)
        self.prepare_image_store(DataSplit.Train)
        self.prepare_image_store(DataSplit.Validation)
    
        # Image Mean (MUL)
        self.prepare_mean_image()
    
        # DONE!
        logger.info("Preproc for training on {} ... done".format(self.prefix))

    def load_data_generator(self, split, batch_size=8,
                            include_groud_truth=True, substract_mean=True):
        """ Loads images data from hdf file(s), substracts mean values.
        Args:
            split (DataSplit) - One of Train, Validation or Test.
            batch_size (int) - Number of images to load at a time. If image is sliced, will load all slices for the given number of images. Loading part of slices will create problems on prediction.
            include_groud_truth (bool) - If true, will include ground truths, otherwise will return null for Y.
        Returns:
            Tuple: (image_ids_to_load, X, Y), Y can be None for testset, image ids are NOT slice ids, but whole image ids.
        """
        fn_im = self.path_mgr.get_image_store_path(split)
        image_id_list = self.load_image_id_list(split=split)
        np.random.shuffle(image_id_list)
        if substract_mean:
            X_mean = self.load_mean_image()

        # No Ground truth masks for Test set.
        if split == DataSplit.Test:
            include_groud_truth = False
        total_sz = len(image_id_list)
        n_batch = int(total_sz // batch_size + 1)
        for i_batch in range(n_batch):
            image_ids_to_load = image_id_list[i_batch*batch_size:i_batch*batch_size+batch_size] 
            if len(image_ids_to_load) == 0:
                continue
            if (self.slice_count != 1):
                slice_ids = image_ids_to_slice_ids(image_ids_to_load, self.slice_count)
            else:
                slice_ids = image_ids_to_load
            X = load_images_from_hdf5_file(fn_im, slice_ids)
            if substract_mean:
                X -= X_mean
            # No Ground truth data for test set.
            Y = None
            if include_groud_truth:
                Y = self.load_ground_truth_masks(split, image_ids_to_load)
            yield (image_ids_to_load, X, Y)
 
    def prepare_mean_image(self):
        """ Computes mean image for 3 or 8 channel images. 
        """
        output_file_path = self.path_mgr.get_mean_image_file_path()
        if output_file_path is None:
            # Mean computation not required.
            return
        if Path(output_file_path).exists():
            logger.info("Generation of {} ... skipped".format(output_file_path))
            return

        logger.info("Generating {}".format(output_file_path))
        X_mean = np.zeros((self.channel_count, self.input_size, self.input_size), dtype=float)
        count = 0

        batch_size = 8
        logger.info("Reading Training data.")
        generator = self.load_data_generator(
            split=DataSplit.Train, batch_size=batch_size,
            include_groud_truth=False, substract_mean=False)
        batch_count = self.get_image_count(split=DataSplit.Train) // batch_size + 1
        for (image_ids, X_train, _) in tqdm.tqdm(generator, total=batch_count):
            count += X_train.shape[0]
            X_mean += np.sum(X_train, axis=0)
        
        logger.info("Reading Validation data.")
        generator = self.load_data_generator(
            split=DataSplit.Validation, batch_size=batch_size,
            include_groud_truth=False, substract_mean=False)
        batch_count = self.get_image_count(split=DataSplit.Validation) // batch_size + 1
        for (image_ids, X_val, _) in tqdm.tqdm(generator, total=batch_count):
            count += X_val.shape[0]
            X_mean += np.sum(X_val, axis=0)
        X_mean /= count

        logger.info("Saving mean image image to {}".format(output_file_path))
        with h5py.File(output_file_path, 'w') as f:
            matrix_to_hdf5(f, X_mean, "mean")

# >>> ------ Command line commands follow --------------------------------

@click.group()
def cli():
    pass


@cli.command()
@click.argument('datapath', type=str)
def preprocess_spacenet(datapath):
    preprocessor = Preprocessor(
        datapath, model_name='preprocessor', original_size_x=650, original_size_y=650,
        input_size=256, slice_count_x=3, slice_count_y=3, is_8_channel=False)
    preprocessor.preprocess_data()
    

@cli.command()
@click.argument('images_folder_path', type=str)
@click.argument('ground_truth_csv_path', type=str)
@click.argument('output_directory_path', type=str)
def preprocess_multichannel_training_data(
        images_folder_path, ground_truth_csv_path, output_directory_path):
    """ Runs preprocessor on any folder with 8-channel images and ground truth csv file.
        First column of the csv file must contain the image file names without '.tif'.
        Sample call: python3 preprocessor.py preproc-multichannel-training-data /data/train/AOI_2_Vegas_Train/MUL-PanSharpen/ GroundTruth.csv /data/working_dir_large_images
    """
    preprocessor = Preprocessor(
        datapath=None, model_name='preprocessor', original_size_x=650, original_size_y=650,
        input_size=256, slice_count_x=3, slice_count_y=3, is_8_channel=True,
        working_dir=output_directory_path)
    # Martun: next 2 lines are a bit dirty hacks, but it was faster this way.
    # Tell path manager where the 8-channel images are located, and that 
    # image ID will be the image name. Also tell the ground truth csv path.
    preprocessor.path_mgr.mul_training_image_path_format = images_folder_path + "/{image_id:s}.tif"
    preprocessor.path_mgr.ground_truth_csv_file_path = ground_truth_csv_path
    preprocessor.preprocess_data()


if __name__ == '__main__':
    cli()

