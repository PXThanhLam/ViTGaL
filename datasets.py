import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
import albumentations.pytorch
import torch
import cv2

class RetrievalDataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, args, mode='train' ):
        assert os.path.exists(args.data_path), "Data path '{}' not found".format(args.data_path)
        self._split = args.train_split if mode == 'train' else args.val_split
        print("Constructing dataset from {}...".format(self._split))
        self._data_path = args.data_path
        self._construct_imdb()
        self.transform = build_data_transform(args, mode)

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        self._imdb, self._class_ids = [], []
        with open(os.path.join(self._data_path, self._split), "r") as fin:
            for line in fin:
                im_dir, cont_id = line.strip().split(" ")
                im_path = os.path.join(self._data_path, im_dir)
                self._imdb.append({"im_path": im_path, "class": int(cont_id)})
                self._class_ids.append(int(cont_id))
        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(set(self._class_ids))))


    def __getitem__(self, index):
        # Load the image
        try:
            img = cv2.imread(self._imdb[index]["im_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print('error: ', self._imdb[index]["im_path"])
        # Prepare the image for training / testing
        img = self.transform(image=img)['image']
        # Retrieve the label
        label = self._imdb[index]["class"]
        return img, label

    def __len__(self):
        return len(self._imdb)


def build_data_transform(args, mode='train'):
    if mode == 'train':
        transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter,
                              saturation=args.color_jitter,hue=args.color_jitter),
                A.ImageCompression(quality_lower=args.compress_lower, quality_upper=args.compress_upper),
                A.ShiftScaleRotate(shift_limit=args.shift_limit, scale_limit=args.scale_limit,                                                       rotate_limit=args.rotate_limit, border_mode=0, 
                                     p=args.shift_roate_scale_p),
                A.RandomResizedCrop(args.input_size, args.input_size,scale = (args.crop_scale,1),p=args.crop_p),
                A.Resize(args.input_size, args.input_size),
                A.Cutout(max_h_size=int(args.input_size * args.cutout_size), max_w_size=int(args.input_size *                                                         args.cutout_size), num_holes=1, p=args.cutout_p),
                A.Normalize(),
                A.pytorch.transforms.ToTensorV2()
                  ])

    elif mode == 'val' or mode == 'test':
        transform = A.Compose([
                    A.Resize(args.input_size, args.input_size),
                    A.Normalize(),
                    A.pytorch.transforms.ToTensorV2()
                ])
    return transform

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('XCiT Retrieval training and evaluation script', add_help=False)
    parser.add_argument('--input-size', default= 512, type=int, help='images input size')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--compress_lower', type=float, default=99, metavar='PCT',
                        help='Image compress lower (default: 99)')
    parser.add_argument('--compress_upper', type=float, default=100, metavar='PCT',
                        help='Image compress upper (default: 100)')
    parser.add_argument('--shift-limit', type=float, default=0.2, metavar='PCT',
                        help='shifting augment (default: 0.2)')
    parser.add_argument('--scale-limit', type=float, default=0.3, metavar='PCT',
                        help='scale augment (default: 0.2)')
    parser.add_argument('--rotate-limit', type=float, default=10, metavar='PCT',
                        help='rotate augment (default: 10)')
    parser.add_argument('--shift-roate-scale-p', type=float, default=0.7, metavar='PCT',
                        help='shift-roate-scale augment prop (default: 0.7)')
    parser.add_argument('--cutout_size', type=float, default=0.25, metavar='PCT',
                        help='max cutout relative size (default: 0.3)')
    parser.add_argument('--cutout_p', type=float, default=0.5, metavar='PCT',
                        help='cutout apply prob (default: 0.5)')
    parser.add_argument('--crop_scale', type=float, default=0.6, metavar='PCT',
                        help='min crop scale)')
    parser.add_argument('--crop_p', type=float, default=0.5, metavar='PCT',
                        help='apply crop prob')

    args = parser.parse_args()
    transform  =   A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter, 
                              saturation=args.color_jitter,   
                              hue=args.color_jitter),
                A.ImageCompression(quality_lower=args.compress_lower, quality_upper=args.compress_upper),
                A.ShiftScaleRotate(shift_limit=args.shift_limit, scale_limit=args.scale_limit,                                                                 rotate_limit=args.rotate_limit, border_mode=0, p=args.shift_roate_scale_p),
                A.RandomResizedCrop(args.input_size, args.input_size,scale = (args.crop_scale,1),p=args.crop_p),
                A.Resize(args.input_size, args.input_size),
                A.Cutout(max_h_size=int(args.input_size * args.cutout_size), max_w_size=int(args.input_size *                                                         args.cutout_size), num_holes=1, p=args.cutout_p)
                  ])
    im_path = '../gglandmark-v2-clean/train/a/c/7/ac7d5f77c34a3a91.jpg'
    for idx,im_path in enumerate([im_path] * 9):
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if idx !=0:
            img = transform(image=img)['image']
        else:
            img = cv2.resize(img,(512,512)) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape)
        cv2.imwrite('transform_ ' + str(idx) + '.png',img)
    
    
    
          



