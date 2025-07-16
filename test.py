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
from util.Gas_dataset import Gas_dataset
from util.util import compute_results,compute_results2, visualize,visualize_gt
from sklearn.metrics import confusion_matrix
from model.model import GasSegNet
from tqdm import tqdm

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--weight_name', '-wn', type=str, default='ResNet152')
parser.add_argument('--weight_dir', '-wd', type=str, default='checkpoints\\ResNet152')

parser.add_argument('--file_name', '-f', type=str, default='latest.pth')
parser.add_argument('--dataset_split', '-d', type=str,
                    default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=512)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='dataset\\test')
args = parser.parse_args()
#############################################################################################

def testing2(model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))

    with torch.no_grad():
        for it, (bg, gas, labels, names) in enumerate(test_loader):
            bg = Variable(bg).cuda(args.gpu)
            gas = Variable(gas).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logit, logits,_,_,_ = model(bg, gas)
            logit_mix = (logit + logits)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logit_mix.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[
                                    0, 1])
            conf_total += conf
    precision, Acc, IoU, F1, F2, _ = compute_results2(conf_total)
    return precision, Acc, IoU, F1, F2

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    weight_dir = os.path.join('./checkpoints/', args.weight_dir)

    batch_size = 1  # do not change this parameter!
    test_dataset = Gas_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    if args.weight_name == "ResNet50":
        num_resnet_layers = 50
    elif args.weight_name == "ResNet152":
        num_resnet_layers = 152
    else:
        sys.exit('no such type model.')


    model_file = os.path.join(weight_dir, args.file_name)
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
    precision, Acc, IoU, F1, F2 = testing2(model, test_loader)   
    print(Acc[1], IoU[1], F1[1])