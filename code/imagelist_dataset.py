import numpy as np
import torch
from torch.utils import data
import warnings
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
import preprocessor

# Logger
logger = getLogger("MainLogger")


class ImageListDataset(data.Dataset):
    """Represents a HDF5 dataset. Loads images from compressed HDF5 file.
    
    Input params:
    """
    def __init__(
            self, image_file_paths, mean_image=None, bandstats_file_path=None, 
            original_size_x=650, original_size_y=650,
            input_size=256, slice_count_x=1, slice_count_y=1, is_8_channel=True):
        super().__init__()
        # logger.info("Creating image list dataset from {} images".format(str(len(image_file_paths))))
        self.preprocessor = preprocessor.Preprocessor(
            datapath=None, original_size_x=original_size_x, original_size_y=original_size_y,
            input_size=input_size, slice_count_x=slice_count_x, slice_count_y=slice_count_y,
            is_8_channel=is_8_channel)
        # Change location of bandstats file, it will not figure out on its own.
        self.preprocessor.path_mgr.bandstats_file = bandstats_file_path
        self.image_file_paths = image_file_paths
        self.slice_count = slice_count_x * slice_count_y
        self.current_image_path = ""
        self.is_8_channel = is_8_channel
        self.preloaded_slices = {}

        # TODO(martun): later change mean substraction as a transformation.
        self.mean_image = mean_image
            
    def __getitem__(self, index):
        if index not in self.preloaded_slices:
            # Load slices from next image.
            self.preloaded_slices = {}
            # get data
            current_image_id = index // self.slice_count
            current_image_path = self.image_file_paths[current_image_id]
            image_gen = self.preprocessor.read_and_slice_image_from_path(
                current_image_path, current_image_path)
            for idx, (slice_id, image) in enumerate(image_gen, start=0):
                self.preloaded_slices[current_image_id * self.slice_count + idx] = image
            #logger.info("current_image_path = {}".format(current_image_path))
        im = self.preloaded_slices[index]
        im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
        im = np.swapaxes(im, 1, 2)  # -> (w, h, ch)
        # For 8-bit RGB images mean image will be None.
        if self.mean_image is not None:
            im -= self.mean_image
        return {'image': torch.from_numpy(im)}

    def __len__(self):
        return len(self.image_file_paths) * self.slice_count

