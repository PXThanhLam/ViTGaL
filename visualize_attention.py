import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import cv2
import utils
from timm.models import create_model
import albumentations.pytorch
import albumentations as A

import xcit_retrieval


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--model_backbone', default='xcit_retrievalv2_small_12_p16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--patch_size', default=16, type=int, metavar='MODEL',
                        help='')
    parser.add_argument('--weight_path', 
                        default='../model_checkpoint/xcit_small_24_p16_retrieval_reduction_ae/checkpoint.pth',
                        type=str, help='')
    parser.add_argument('--up_factor', default=1, type=int, metavar='MODEL',
                        help='')
    parser.add_argument('--nb_classes', default=1000, type=int, metavar='MODEL',
                        help='')
    parser.add_argument('--pretrained', default='../model_checkpoint/xcit_small_12_p16_retrieval_5e-5-1e-7_19_tune_aug/checkpoint_model_epoch_37.pth', type=str,                                  metavar='MODEL',help='Name of model to train')
    parser.add_argument("--image_path", default='AttentionVisualizeInput', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(512, 512), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--keep_aspec_ratio", default=False, type=utils.bool_flag, help="Keep aspec_ratio.")
    parser.add_argument('--output_dir', default='AttentionVisualizeOuput', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    #################
    model_backbone = create_model(
        args.model_backbone,
        pretrained=False,
        num_classes=0,
    )
    model = create_model(
        'xcit_retrievalv2_reduction_ae',
        pretrained=False,
        num_classes=0,
        retrieval_back_bone = model_backbone,
        drop_block_rate=None,
    )

    device = torch.device("cuda")
    
    resume_path= args.weight_path
    checkpoint = torch.load(resume_path, map_location='cpu')['model']
    for key in list(checkpoint.keys()):
        if 'retrieval_back_bone' in key:
            reduce = checkpoint[key]
            orig = checkpoint_backbone_model[key[20:]]
            print(key,torch.sum(reduce-orig))
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint.keys():
            print(f"Removing key {k} from pretrained model checkpoint")
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    #########

    # open image
    all_imgs = []
    all_img_paths = []
    if os.path.isfile(args.image_path):
        img = cv2.imread(args.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_imgs.append(img)
        all_img_paths.append(args.image_path.split('/')[-1])
    elif os.path.isdir(args.image_path):
        for file_path in os.listdir(args.image_path):
            all_img_paths.append(file_path)
            file_path = args.image_path + '/' + file_path
            img = cv2.imread(file_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_imgs.append(img)
            except:
                continue

    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    
    for im_path,img in zip(all_img_paths,all_imgs):
        im_h, im_w, _ = np.array(img).shape
        resize_h, resize_w = args.image_size[0] * im_h // max(im_h,im_w),  args.image_size[0] * im_w // max(im_h,im_w)
        resize_size = (resize_h, resize_w) if  args.keep_aspec_ratio else args.image_size
        transform = A.Compose([
                    A.Resize(resize_size[0],resize_size[1]),
                    A.Normalize(),
                    A.pytorch.transforms.ToTensorV2()
                ])
        img = transform(image=img)['image']

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size * args.up_factor
        h_featmap = img.shape[-1] // args.patch_size * args.up_factor
        with torch.no_grad():
            _, attentions, _, _ = model.forward_features(img.to(device))

        nh = attentions.shape[1] # number of head
        print(attentions.shape)
        # we keep only the output patch attention
        attentions = attentions[0]
        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size//args.up_factor, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size//args.up_factor, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            fname = os.path.join(args.output_dir, im_path + "-attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")

        if args.threshold is not None:
            image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
            for j in range(nh):
                display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)