import os
import argparse
import time
import datetime
import sys
import shutil
import stat
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.util import compute_results, show
from sklearn.metrics import confusion_matrix
from model.model import GasSegNet
import PIL
import cv2

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--weight_name', '-w', type=str, default='ResNet152_12_6_model1_2')
parser.add_argument('--file_name', '-f', type=str, default='44_latest.pth')
parser.add_argument('--dataset_split', '-d', type=str,
                    default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=512)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='dataset/test/images')
parser.add_argument('--save_dir', '-sr', type=str, default='./save_test')
args = parser.parse_args()
#############################################################################################
def get_palette():
    unlabelled = [0,0,0]
    gas        = [255,0,255]
    palette    = np.array([unlabelled,gas])
    return palette

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    os.makedirs("save_test",exist_ok=True)
    model_dir = os.path.join('./checkpoints/', args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')

    conf_total = np.zeros((args.n_class, args.n_class))
    if args.weight_name.startswith("ResNet50"):
        num_resnet_layers = 50
    elif args.weight_name.startswith("ResNet152"):
        num_resnet_layers = 152
    else:
        sys.exit('no such type model.')
    model = GasSegNet(args.n_class,num_resnet_layers)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(
        model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()

    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    for name, param in pretrained_weight.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param)
    print('done!')

    ave_time_cost = 0.0

    model.eval()

    for root, dirs, files in os.walk(args.data_dir):
        for dir in dirs:
            bg_path = os.path.join(args.data_dir,dir,"bg.png")
            gas_path = os.path.join(args.data_dir,dir,"gas.png")

    # bg_path = "dataset\\test\\images\\00821\\bg.png"
    # gas_path = "dataset\\test\\images\\00821\\gas.png"

    # bg_path = "dataset\\test\\images\\01225\\bg.png"
    # gas_path = "dataset\\test\\images\\01225\\gas.png"

            bg = np.asarray(PIL.Image.open(bg_path).convert('L'), dtype=np.float32)
            gas = np.asarray(PIL.Image.open(gas_path).convert('L'), dtype=np.float32)
            origin = np.asarray(PIL.Image.open(gas_path).convert('L').resize((640, 512)), dtype=np.uint8)

            bg = np.asarray(PIL.Image.fromarray(bg).resize((640, 512)), dtype=np.float32)
            bg = np.expand_dims(bg, axis=-1).transpose((2,0,1))/255

            gas = np.asarray(PIL.Image.fromarray(gas).resize((640, 512)), dtype=np.float32)
            gas = np.expand_dims(gas, axis=-1).transpose((2,0,1))/255
            bg = torch.tensor(bg).unsqueeze(0)
            gas = torch.tensor(gas).unsqueeze(0)

            bg = Variable(bg).cuda(args.gpu)
            gas = Variable(gas).cuda(args.gpu)
            logit, logits, _, _, _ = model(bg, gas)
            logit_mix = logit + logits
            predictions = logit_mix.argmax(1)
            palette = get_palette()
            pred = predictions[0].cpu().numpy()
            img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
                    img[pred == cid] = palette[cid]

            thermal = cv2.cvtColor(origin,cv2.COLOR_GRAY2BGR)
            # 创建掩膜（灰度值不为0的部分作为mask）
            # mask = cv2.inRange(img, 1, 255)  # 非黑色部分的掩膜

            # 将掩膜扩展为三通道（与图片形状匹配）
            # mask_bgr = cv2.merge([mask, mask, mask])

            # 定义透明度 alpha（范围：0-1）
            alpha = 0.5

            # 初始化输出图像，直接复制目标图片
            output = thermal.copy()

            # 在 mask 区域内进行逐像素透明度混合
            # output[img > 0] = (thermal[img > 0] // 2 + img[img > 0] // 2).astype(np.uint8)
            output[img > 0] = cv2.addWeighted(thermal,0.5,img,0.5,0)[img > 0]

            # img = cv2.addWeighted(thermal,0.5,img,0.5,0)
            cv2.imwrite(f"save_test/{dir}_pred.png",output)