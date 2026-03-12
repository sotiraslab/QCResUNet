import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from torch.utils.data import Dataset
from nnunet.evaluation.region_based_evaluation import create_region_from_mask
from batchgenerators.dataloading.data_loader import DataLoader, SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def get_brats_regions():
    regions = [
        (3,), # "enhancing tumor"
        (2,), # "ncr/net"
        (1,), # "ed"
    ]

    return regions

def get_regional_mask(mask, regions):
    masks = []
    for r in regions:
        masks.append(create_region_from_mask(mask, r))
    
    return np.stack(masks).transpose(1, 0, 2, 3, 4)

def resampling(dsc, min_number, bins=np.linspace(0, 1, 11)):
    bins_low = bins[:-1]
    bins_upper = bins[1:]

    selected_idx = []
    for i in range(len(bins)-1):
        if i == len(bins)-2:
            idx = np.where(np.logical_and(dsc >= bins_low[i], dsc <= bins_upper[i]))[0]
        else:
            idx = np.where(np.logical_and(dsc >= bins_low[i], dsc < bins_upper[i]))[0]
        if len(idx) > min_number:
            selected_idx.extend(np.random.choice(idx, min_number))
        else:
            selected_idx.extend(idx)

    selected_idx = np.array(selected_idx)
    np.random.shuffle(selected_idx)

    return selected_idx


class RegDataLoaderV1(DataLoader):
    def __init__(self, dataroot, data_dir, mode, fold, patch_size, batch_size, 
                    concate_seg=False, onehot=False, 
                    infinite=True, number_of_threads_in_multithreaded=None):
        self.onehot = onehot
        self.data_dir = data_dir
        self.mode = mode
        self.concate_seg = concate_seg

        self.query_dir = join(data_dir, mode, f"fold_{fold}")

        data = pd.read_csv(dataroot)
        self.data_all = data.values

        if mode == 'tr':
            self.dsc = self.data_all[:, -1]

            plt.figure()
            n, bins, patches = plt.hist(self.dsc)
            self.num_min = int(np.min(n))
            self.bins = bins
            plt.close('all')

            resampled_idx = resampling(self.dsc, self.num_min, self.bins)
            data = self.data_all[resampled_idx, ...]
        else:
            data = self.data_all

        super().__init__(data, batch_size, number_of_threads_in_multithreaded, infinite=infinite)
        self.patch_size = patch_size
        self.list_of_keys = list(range(len(self._data)))
        self.indices = list(range(len(self._data)))
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        data_shape = (self.batch_size, 4, *self.patch_size)
        seg_shape = (self.batch_size, 1, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        # selected_idx = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        selected_idx = self.get_indices()
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        seg_gt = np.zeros(self.seg_shape, dtype=np.float32)
        target = np.zeros((self.batch_size, 1), dtype=np.float32)

        subject_ids = []
        ids = []
        for j, i in enumerate(selected_idx):
            subject_id = self._data[i, 0]
            subject_ids.append(subject_id)
            seg_id = self._data[i, 3]
            ids.append(seg_id)
            seg_path = join(self.query_dir, f"{seg_id}.npy")
            raw_data_file = self._data[i, 2]

            # print(raw_data_file)

            if os.path.isfile(raw_data_file[:-4] + ".npy"):
                case_all_data = np.load(raw_data_file[:-4] + ".npy", "r+")
            else:
                case_all_data = np.load(raw_data_file)['data']

            input_modal = case_all_data[:4, ...]
            # seg_gt = case_all_data[-1, ...][np.newaxis, ...]
            seg_pred = np.load(seg_path)[np.newaxis, ...]

            data[j, ...] = input_modal
            seg[j, ...] = seg_pred
            seg_gt[j, ...] = (case_all_data[-1, ...][np.newaxis, ...]).astype(np.uint8)
            target[j] = self._data[i, -1]
    
        if self.onehot:
            seg = get_regional_mask(seg.squeeze(), get_brats_regions())
            seg_gt = get_regional_mask(seg_gt.squeeze(), get_brats_regions())

        if not self.concate_seg:
            return {'data': data, 'seg':seg, 'seg_gt': seg_gt, 'target':target, 'subject_ids':subject_ids, 'ids':ids}
        else:
            seg_concat = np.concatenate([seg, seg_gt], axis=1)
            return {'data': data, 'seg':seg_concat, 'target':target, 'subject_ids':subject_ids, 'ids':ids}

    def resample(self):
        resampled_idx = resampling(self.dsc, self.num_min, self.bins)
        self._data = self.data_all[resampled_idx]
