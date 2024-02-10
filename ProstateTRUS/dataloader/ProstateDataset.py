import torch
import os
import numpy as np
from torch.utils.data import Dataset


class PD1C(Dataset):
    def __init__(self, case_list, data_root_path, image_id='', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.transform = transform
        self.image_id = image_id

    def __getitem__(self, index):
        data_path = os.path.join(self.data_root_path, self.case_list[index] + '.npz')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        volume = data['volume{}'.format(self.image_id)]
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume = volume[0:1, :, :, :].transpose(0, 3, 2, 1)
        # volume = volume[0:1, :, :, :]
        name = self.case_list[index]
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': case_id, 'volume': volume, 'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PD3C(Dataset):
    def __init__(self, case_list, data_root_path, image_id='', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.transform = transform
        self.image_id = image_id

    def __getitem__(self, index):
        data_path = os.path.join(self.data_root_path, self.case_list[index] + '.npz')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        if self.image_id == 0:
            volume = data['volume']
        else:
            volume = data['volume{}'.format(self.image_id)]
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume = volume[:, :, :, :].transpose(0, 3, 2, 1)
        # volume = volume[0:1, :, :, :]
        name = self.case_list[index]
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': case_id, 'volume': volume, 'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PD4C(Dataset):
    def __init__(self, case_list, data_root_path, image_id=False, transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.transform = transform

    def __getitem__(self, index):
        data1_path = os.path.join(self.data_root_path, self.case_list[index] + '.npz')
        case_id = self.case_list[index]
        data = np.load(data1_path, allow_pickle=True)
        volume1 = data['volume1']
        volume2 = data['volume2']
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume1 = volume1[0:1, :, :, :].transpose(0, 3, 2, 1)
        volume2 = volume2[:, :, :, :].transpose(0, 3, 2, 1)
        volume = np.concatenate((volume1, volume2), axis=0)
        # volume = volume[0:1, :, :, :]
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': case_id, 'volume': volume, 'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PD6C(Dataset):
    def __init__(self, case_list, data_root_path, image_id=False, transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.transform = transform

    def __getitem__(self, index):
        data1_path = os.path.join(self.data_root_path, self.case_list[index] + '.npz')
        case_id = self.case_list[index]
        data = np.load(data1_path, allow_pickle=True)
        volume1 = data['volume1']
        volume2 = data['volume2']
        benign_malignant = data['label']
        # c z y x -> c x y z
        volume1 = volume1[:, :, :, :].transpose(0, 3, 2, 1)
        volume2 = volume2[:, :, :, :].transpose(0, 3, 2, 1)
        volume = np.concatenate((volume1, volume2), axis=0)
        # volume = volume[0:1, :, :, :]
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': case_id, 'volume': volume, 'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class PDCAM(Dataset):
    def __init__(self, case_list, data_root_path, image_id='', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.transform = transform
        self.image_id = image_id

    def __getitem__(self, index):
        data_path = os.path.join(self.data_root_path, self.case_list[index] + '.npz')
        case_id = self.case_list[index]
        data = np.load(data_path, allow_pickle=True)
        volume = data['volume{}'.format(self.image_id)]
        benign_malignant = data['label']
        # c z y x -> x y z
        volume = volume[:, :, :, :].transpose(0, 3, 2, 1)
        # volume = volume[0:1, :, :, :]
        name = self.case_list[index]
        cspca = data['CsPCa'] if benign_malignant != 0 else 0
        sample = {'name': case_id, 'volume': volume, 'benign_malignant': benign_malignant,
                  'cspca': cspca}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.case_list)


class Normalization(object):
    def __init__(self, volume_key='volume'):
        self.volume_key = volume_key

    def __call__(self, sample):
        image_array = sample[self.volume_key]
        arr = image_array.reshape(-1)
        arr_mean = np.mean(arr)
        arr_var = np.var(arr)
        image_array = (image_array - arr_mean) / (arr_var + 1e-6)
        sample[self.volume_key] = image_array
        return sample


class ToTensor(object):
    def __init__(self, volume_key='volume'):
        self.volume_key = volume_key

    def __call__(self, sample):
        volume = sample[self.volume_key]
        # volume = np.expand_dims(volume, axis=1)
        sample[self.volume_key] = torch.from_numpy(volume.copy())
        return sample


class SparseZSlice(object):
    def __init__(self, sample_interval=2, volume_key='volume'):
        self.volume_key = volume_key
        self.sample_interval = sample_interval

    def __call__(self, sample):
        volume = sample[self.volume_key]
        z_slice = volume.shape[-1]
        if z_slice % self.sample_interval != 0:
            z_num = z_slice - (z_slice % self.sample_interval)
            volume = volume[:, : z_num, ...]
        prob = np.random.random()
        if self.sample_interval == 3:
            if prob < 1/3:
                volume = volume[..., ::self.sample_interval]
            elif 1/3 < prob < 2/3:
                volume = volume[..., 1::self.sample_interval]
            else:
                volume = volume[..., 2::self.sample_interval]
        elif self.sample_interval == 2:
            if prob < 0.5:
                volume = volume[..., ::self.sample_interval]
            else:
                volume = volume[..., 1::self.sample_interval]
        elif self.sample_interval == 4:
            if prob < 0.25:
                volume = volume[..., ::self.sample_interval]
            elif 0.25 < prob < 0.5:
                volume = volume[..., 1::self.sample_interval]
            elif 0.5 < prob < 0.75:
                volume = volume[..., 2::self.sample_interval]
            else:
                volume = volume[..., 3::self.sample_interval]
        else:
            return -1
        sample[self.volume_key] = volume

        return sample


class SparseZSliceGauss(object):
    def __init__(self, z_num=176, volume_key='volume'):
        self.volume_key = volume_key
        self.z_num = z_num

    def __call__(self, sample):
        volume = sample[self.volume_key]
        z_slice = volume.shape[-1]
        gaussian_indices = sorted(np.random.choice(np.arange(z_slice), size=self.z_num, replace=False))
        sample[self.volume_key] = volume[..., gaussian_indices]

        return sample


class Crop(object):
    def __init__(self, crop_size, volume_key='volume'):
        self.volume_key = volume_key
        self.crop_size = crop_size

    def forward(self, sample):
        volume = sample[self.volume_key]
        sample[self.volume_key] = volume[:, :, 30: -20, :]
        return sample




class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def next(self):
        sample = self.sample
        self.preload()
        return sample

    def preload(self):
        try:
            self.sample = next(self.loader)
        except StopIteration:
            self.sample = None
            return

