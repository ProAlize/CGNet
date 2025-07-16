import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
import random

class Gas_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=512, input_w=640 ,transform=[], seed = 3409):
        super(Gas_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night


        image_path = os.path.join(data_dir,"images")
        self.data = sorted([name for name in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, name))])
        random.seed(seed)
        random.shuffle(self.data)

        self.image_path = image_path
        self.label_path = os.path.join(data_dir,"labels")
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.data)
    
    def read_thermal_image(self, name):
        bg_path = os.path.join(self.image_path, name, "bg.png")
        gas_path = os.path.join(self.image_path, name, "gas.png")
        bg = np.asarray(PIL.Image.open(bg_path).convert('L'))
        bg = np.expand_dims(bg, axis=-1)
        gas = np.asarray(PIL.Image.open(gas_path).convert('L'))
        gas = np.expand_dims(gas, axis=-1)
        return (bg,gas)

    def read_label(self, name):
        gt_path = os.path.join(self.label_path, name, "gt.png")
        label = np.asarray(PIL.Image.open(gt_path).resize((self.input_w, self.input_h)))
        label = np.where(label > 128, 1, 0).astype(np.uint8)
        return label


    def __getitem__(self, index):
        name  = self.data[index]
        bg,gas = self.read_thermal_image(name)
        label = self.read_label(name)
        for func in self.transform:
            bg, gas, label = func(bg, gas, label)
        bg = bg.copy()
        gas = gas.copy()
        label = label.copy()
        bg = np.asarray(PIL.Image.fromarray(bg[:,:,-1]).resize((self.input_w, self.input_h)), dtype=np.float32)
        bg = np.expand_dims(bg, axis=-1).transpose((2,0,1))/255
        bg = torch.tensor(bg)
        gas = np.asarray(PIL.Image.fromarray(gas[:,:,-1]).resize((self.input_w, self.input_h)), dtype=np.float32)
        gas = np.expand_dims(gas, axis=-1).transpose((2,0,1))/255
        gas = torch.tensor(gas)
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)
        label = torch.tensor(label)
        return bg, gas, label, name

    def __len__(self):
        return self.n_data
