import random
import numpy as np
import h5py
import scipy.ndimage
import glob


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.train_data_dir = cfg.train_data_dir
        self.valid_data_dir = cfg.valid_data_dir
        self.test_data_dir = cfg.test_data_dir
        self.batch_size = cfg.batch_size
        self.num_tr = cfg.num_tr
        self.height, self.width, self.depth, self.channel = cfg.height, cfg.width, cfg.depth, cfg.channel
        project_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation'
        self.train_file = project_path + '/data_preparation/our_data/6_2d/train_2d.h5'
        self.valid_file = project_path + '/data_preparation/our_data/6_2d/test_2d.h5'
        self.test_file = project_path + '/data_preparation/our_data/6_2d/test_2d.h5'
        self.num_tr = self.count_num_samples(mode='train')

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            img_idx = np.sort(np.random.choice(self.num_tr, replace=False, size=self.batch_size))
            h5f = h5py.File(self.train_file, 'r')
            x = h5f['x_norm'][img_idx]
            y = h5f['y'][img_idx]
            h5f.close()
        elif mode == 'valid':
            h5f = h5py.File(self.valid_file, 'r')
            x = h5f['x_norm'][start:end]
            y = h5f['y'][start:end]
            h5f.close()
        else:
            h5f = h5py.File(self.test_file, 'r')
            x = h5f['x_norm'][start:end]
            y = h5f['y'][start:end]
            h5f.close()
        return x, y

    def count_num_samples(self, mode='valid'):
        if mode == 'train':
            h5f = h5py.File(self.train_file, 'r')
            num_ = h5f['y'][:].shape[0]
            h5f.close()
        elif mode == 'valid':
            h5f = h5py.File(self.valid_file, 'r')
            num_ = h5f['y'][:].shape[0]
        else:
            h5f = h5py.File(self.test_file, 'r')
            num_ = h5f['y'][:].shape[0]
        return num_
