import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
import random

def find_gt_file(label_path, name):
    # 尝试 gt.png 和 gt3.png
    for filename in [ "gt3.png","gt.png",]:
        gt_path = os.path.join(label_path, name, filename)
        if os.path.exists(gt_path):
            return gt_path
    return None  # 如果两个文件都不存在

def find_image_file(base_path, name, img_type):
    # 尝试 jpg 和 png 扩展名
    for ext in ['.jpg', '.png']:
        img_path = os.path.join(base_path, name, f"{img_type}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None  # 如果两个扩展名的文件都不存在

class Gas_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=256, input_w=320 ,transform=[], seed = 3409):
        super(Gas_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        # with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
        #     blacklist = open(os.path.join(data_dir, 'blacklist.txt')).readlines()
        #     self.data =[]
        #     for name in f.readlines():
        #         if name not in blacklist:
        #             self.data.append(name.strip())
        #         else:
        #             print(name.strip())
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
        bg_path = find_image_file(self.image_path, name, "bg")
        gas_path = find_image_file(self.image_path, name, "gas")
        bg = np.asarray(PIL.Image.open(bg_path).convert('L'))
        bg = np.expand_dims(bg, axis=-1)
        gas = np.asarray(PIL.Image.open(gas_path).convert('L'))
        gas = np.expand_dims(gas, axis=-1)
        return (bg,gas)

    def read_label(self, name):
        gt_path = find_gt_file(self.label_path, name)
        label = np.asarray(PIL.Image.open(gt_path).resize((self.input_w, self.input_h)))
        label = np.where(label > 128, 1, 0).astype(np.uint8)
        # label = np.expand_dims(label, axis=-1)
        return label


    def __getitem__(self, index):
        name  = self.data[index]
        bg,gas = self.read_thermal_image(name)
        # image = self.read_image(name, 'ablation') # 消融实验 无可见光
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
        # label = np.expand_dims(label, axis=-1)
        label = torch.tensor(label)
        # label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)
        return bg, gas, label, name

    def __len__(self):
        return self.n_data
