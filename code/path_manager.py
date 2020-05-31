from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from pathlib import Path
import utils
from utils import *

logger = getLogger("MainLogger")

class PathManager:
    def __init__(self, datapath, slice_count, is_8_channel,
                 model_name, create_dirs=True,
                 working_dir="/data/working"):
        self.datapath = datapath
        self.slice_count = slice_count
        self.is_8_channel = is_8_channel
        self.model_name = model_name
        self.working_dir = working_dir
        self.prefix = utils.datapath_to_prefix(datapath)

        if datapath is not None:
            self.ground_truth_csv_file_path = str(Path("{datapath:s}/") /
                Path("summaryData/{prefix:s}_Train_Building_Solutions.csv")).format(
                    datapath=self.datapath, prefix=self.prefix)
            self.mul_training_image_path_format = str(
                Path(self.datapath) /
                Path("MUL-PanSharpen-8-bit/MUL-PanSharpen_{image_id:s}_8_bit.jpg"))
            # Input files
            self.rgb_training_image_path_format = str(
                Path(self.datapath) /
                Path("RGB-PanSharpen-8-bit/RGB-PanSharpen_{image_id:s}_8_bit.jpg"))
            self.FMT_TEST_RGB_IMAGE_PATH = str(
                Path("{datapath:s}/") /
                Path("RGB-PanSharpen-8-bit/RGB-PanSharpen_{image_id:s}_8_bit.jpg"))
            self.FMT_TEST_MSPEC_IMAGE_PATH = str(
                Path("{datapath:s}/") /
                Path("MUL-PanSharpen-8-bit/MUL-PanSharpen_{image_id:s}_8_bit.jpg"))

        self.PREPROC_DIR = "{}/{}/preprocessed".format(self.working_dir, self.prefix)
        self.INFERENCE_DIR = "{}/{}/inference/{}".format(self.working_dir, self.prefix, model_name)
        self.MODEL_DIR = "{}/{}/model_weights/{}".format(self.working_dir, self.prefix, model_name)

        # Create working dir if not exists. If there are multiple processes running,
        # only one of them will create the directories.
        if create_dirs:
            utils.create_dir_if_doesnt_exist(self.PREPROC_DIR)
            utils.create_dir_if_doesnt_exist(self.INFERENCE_DIR)
            utils.create_dir_if_doesnt_exist(self.MODEL_DIR)
        self.bandstats_file = (self.PREPROC_DIR + "/bandstats_{channels}_channels.csv").format(
            channels="8" if self.is_8_channel else "3")
        self.mean_image_path = (self.PREPROC_DIR + "/mean_image_store_{channels}_channels_{slice_count}_slices.h5").format(
            channels="8" if self.is_8_channel else "3",
            slice_count=self.slice_count)

           
    def get_final_merged_solution_file_path(self):
        return "/data/output/{}.csv".format(self.model_name)

    def get_evaluation_history_csv_file_path(self):
        return self.INFERENCE_DIR + "/evaluation_history.csv"

    def get_evaluation_history_with_threshold_csv_path(self):
        return self.INFERENCE_DIR + "/evaluation_history_with_thresholds.csv"

    def get_last_weights_path(self):
        return self.MODEL_DIR + '/weights_for_best_epoch.h5'

    def get_weights_path_for_epoch(self, epoch):
        return self.get_weights_path_for_epoch_format().format(epoch=epoch)

    def get_weights_path_for_epoch_format(self):
        return self.MODEL_DIR + '/weights_for_epoch_{epoch:02d}.h5'

    def get_predicted_polygons_csv_file_path(self, split):
        return (self.INFERENCE_DIR + "/predicted_polygons_{split}.csv").format(
            split=split.name)

    def get_ground_truth_csv_file_path(self, split):
        return (self.PREPROC_DIR + "/ground_truth_polygons_{split}.csv").format(
            split=split.name)

    def get_training_summary_path(self):
        return self.ground_truth_csv_file_path

    def get_test_image_path_from_imageid(self, image_id):
        if self.is_8_channel:
            return self.FMT_TEST_MSPEC_IMAGE_PATH.format(
                datapath=self.datapath, image_id=image_id)
        else:
            return self.FMT_TEST_RGB_IMAGE_PATH.format(
                datapath=self.datapath, image_id=image_id)

    def get_train_image_path_from_imageid(self, image_id):
        if self.is_8_channel:
            return self.mul_training_image_path_format.format(image_id=image_id)
        else:
            return self.rgb_training_image_path_format.format(image_id=image_id)

    def get_image_path_from_imageid(self, image_id, split):
        if split in [DataSplit.Train, DataSplit.Validation]:
            return self.get_train_image_path_from_imageid(image_id)
        else:
            return self.get_test_image_path_from_imageid(image_id)

    def get_predictions_output_path(self, split):
        return (self.INFERENCE_DIR + "/result_{channels}_channels_{slice_count}_slices_{split}.h5").format(
            channels="8" if self.is_8_channel else "3",
            slice_count=self.slice_count,
            split=split.name)

    def get_mean_image_file_path(self):
        if not self.is_8_channel:
            return None
        return self.mean_image_path

    def get_image_list_csv_path(self, split):
        return (self.PREPROC_DIR + "/image_list_{split}.csv").format(
            split=split.name)

    def get_image_store_path(self, split):
        return (self.PREPROC_DIR + "/image_store_{channel_count}_channels_{slice_count}_slices_{split}.h5").format(
            channel_count=8 if self.is_8_channel else 3,
            slice_count=self.slice_count,
            split=split.name)

    def get_mask_store_path(self, is_training=True):
        """ Returns path to hdf5 file which stores ground truth annotation masks.
        """
        return (self.PREPROC_DIR + "/masks_{split}_{slice_count}_slices_{channels}_channels.h5").format(
            channels="8" if self.is_8_channel else "3",
            slice_count=self.slice_count,
            split="training" if is_training else "validation")

    def get_bandstats_file_path(self):
        return self.bandstats_file

