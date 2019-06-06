import torch
import random
import numpy as np
from skimage import io


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data_files, label_files, cache=False, augmentation=True):
        super(DataGenerator, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        self.data_files = data_files
        self.label_files = label_files

        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return len(self.data_files)

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        # random_idx = random.randint(0, len(self.data_files) - 1)
        random_idx = i

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            img = io.imread(self.data_files[random_idx])
            data = 1 / 255 * np.asarray(img.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = io.imread(self.label_files[random_idx], as_gray=True)

            label = np.float32(label)

            if self.cache:
                self.label_cache_[random_idx] = label

        # Data augmentation
        if self.augmentation:
            data, label = self.data_augmentation(data, label)

        # Return the torch.Tensor values
        return (torch.from_numpy(data), torch.from_numpy(label))