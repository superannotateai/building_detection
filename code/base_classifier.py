from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
import torch
from path_manager import PathManager
import pandas as pd
import utils

logger = getLogger("MainLogger")

# Class which will be base for both classifier and preprocessor. Sets up all necessary paths and 
# other common settings.
class BaseClassifier:
    def __init__(
            self, datapath, model_name, original_size_x, original_size_y, input_size,
            slice_count_x=1, slice_count_y=1, is_8_channel=True,
            gpu_number=None, working_dir="/data/working", is_8_bit=True):
        """ 
        Args:
            is_8_channel (bool) - True for 8 channel, false for RGB images.
        """
        # Set up what data is used by current classifier.
        self.slice_count_x = slice_count_x
        self.slice_count_y = slice_count_y
        self.slice_count = slice_count_x * slice_count_y

        self.is_8_channel = is_8_channel
        self.is_8_bit = is_8_bit
        self.channel_count = 8 if is_8_channel else 3

        self.original_size_x = original_size_x
        self.original_size_y = original_size_y

        self.input_size = input_size
        self.path_mgr = PathManager(
            datapath, self.slice_count, is_8_channel,
            model_name, create_dirs=(gpu_number==0) or (gpu_number is None),
            working_dir=working_dir)

        self.datapath = datapath
        self.prefix = utils.datapath_to_prefix(datapath)

        self.gpu_number = gpu_number
        if gpu_number is None:
            self.device = torch.device('cpu')
            logger.info("Running on CPU")
        else:
            torch.cuda.set_device(gpu_number)
            self.device = torch.device('cuda') # Get current Cuda device
            logger.info("Running on GPU #{}".format(gpu_number))

        if self.slice_count != 1:
            self.stride_size_x = int((self.original_size_x - self.input_size) / 
                    (self.slice_count_x - 1))
            self.stride_size_y = int((self.original_size_y - self.input_size) / 
                    (self.slice_count_y - 1))

    def load_image_id_list(self, split):
        """ Loads list of image ids from csv file, either for training or validation.
        Args:
            split (DataSplit): One of Train, Validation or Test.
        """
        fn_list = self.path_mgr.get_image_list_csv_path(split)
        df = pd.read_csv(fn_list, index_col='ImageId')
        return [image_id for image_id in df.index]

