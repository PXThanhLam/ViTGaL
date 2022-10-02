import copy
import io
import argparse
import os

from scipy import spatial
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io as skio
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from PIL import Image

from timm.models import create_model
import utils
import cv2
from tqdm import tqdm
import albumentations.pytorch
import albumentations as A
import pydegensac
import xcit_retrieval
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Returns:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
def extract_local_descriptors(model, input, patch_size, attn_threshold, min_attn_threshold, max_key_point, multiscale=None):
    if multiscale == None:
        multiscale = [1]
    selected_locations = []
    selected_features = []
    selected_scales = []
    all_local_scores = []
    _,_, input_h, input_w = input.shape
    for scale in multiscale:
        resize_h, resize_w = input_h * scale, input_w * scale
        resize_h, resize_w = ((resize_h -1 ) // patch_size + 1) * patch_size, ((resize_w -1 ) // patch_size + 1) * patch_size
        scale_h, scale_w = resize_h/input_h, resize_w/input_w
        input_scale = nn.functional.interpolate(input, scale_factor=(scale_h, scale_w), mode='bilinear', align_corners=False)
        with torch.no_grad():
            _, attentions, embeddings, _, _ = model.forward_features(input_scale.to(device))
        nh = attentions.shape[1] # number of head
        h_featmap = input_scale.shape[-2] // patch_size
        w_featmap = input_scale.shape[-1] // patch_size
        attentions = attentions[0]
        attentions = attentions/torch.max(attentions,dim=1).values.unsqueeze(1)
        attentions = torch.mean(attentions, dim =0)
        attentions = attentions / torch.sum(attentions)
        attentions = attentions * len(attentions)
        

        scale_attn_threshold = attn_threshold
        indices = None
        max_kp_scale = max_key_point * (scale_h * scale_w)
        while(indices is None or len(indices) == 0 or len(indices) < max_kp_scale):
            indices = torch.gt(attentions, scale_attn_threshold).nonzero().squeeze()
            try :
                len(indices)
            except:
                indices = None
            scale_attn_threshold = scale_attn_threshold * 0.5   # use lower threshold if no indexes are found.
            if scale_attn_threshold < min_attn_threshold:
                break
        embeddings = embeddings[0]
        embeddings = nn.functional.normalize(embeddings, dim=1, p=2)
        selected_features.extend(list(torch.index_select(embeddings, dim=0, index=indices).cpu().numpy()))
        all_local_scores.extend(list(torch.index_select(attentions, dim= 0, index=indices).cpu().numpy()))
        for indice in indices.cpu().numpy():
            indice_h = indice // w_featmap
            indice_w = indice % w_featmap
            x1 = int((indice_w * patch_size)/scale_w)
            y1 = int((indice_h * patch_size)/scale_h)
            x2 = int((indice_w * patch_size + patch_size)/scale_w)
            y2 = int((indice_h * patch_size + patch_size)/scale_h)
            selected_locations.append([y1,x1,y2,x2])
            selected_scales.append(scale)
    # overlap : 0.7 for 128 dim, 0.5 for 384 dim
    keep_idx, count = nms(torch.tensor(selected_locations), torch.tensor(all_local_scores), overlap=args.overlap, top_k=args.num_points + 200)
    keep_idx = keep_idx.cpu().numpy()
    keep_idx = keep_idx[:min(args.num_points,count)]
    selected_locations = np.array(selected_locations)[keep_idx]
    selected_locations = np.array([((y1+y2)//2,(x1+x2)//2) for (y1,x1,y2,x2) in selected_locations])
    return selected_locations,np.array(selected_features)[keep_idx],None,None


def preprocess_image(args, im_path, image_size = None) :
    patch_size = args.patch_size
    if image_size is None :
        image_size = args.image_size
    
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            
    im_h, im_w, _ = np.array(img).shape
    resize_h, resize_w = image_size * im_h // max(im_h,im_w),  image_size * im_w // max(im_h,im_w)
    resize_h, resize_w = resize_h - resize_h % patch_size, resize_w - resize_w % patch_size # divisible by the patch size
    resize_size = (resize_h, resize_w) if  args.keep_aspec_ratio else (image_size,image_size)
    img_numpy = cv2.resize(np.array(img),resize_size[::-1])

    transform = A.Compose([
                    A.Resize(resize_size[0],resize_size[1]),
                    A.Normalize(),
                    A.pytorch.transforms.ToTensorV2()
                ])    
    img = transform(image=img)['image'].cpu()
    img = img.unsqueeze(0)
    return img_numpy, img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing match pair')
    # model config
    parser.add_argument('--model', default='xcit_retrievalv2_small_12_p16', type=str, metavar='MODEL',help='')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    
    parser.add_argument("--attn_threshold", default = 60, type=float)
    parser.add_argument("--min_attn_threshold", default = 0.6 , type=float)
    parser.add_argument("--max_keypoint", default=400, type=int)

    # io config
    parser.add_argument("--keep_aspec_ratio", default=True, type=utils.bool_flag, help="Keep aspec_ratio.")
    parser.add_argument("--use_multi_scale", default=True, type=utils.bool_flag, help="")
    parser.add_argument("--image_folders", default='../test_datasets/roxford5k/jpg', type=str, help="")
    parser.add_argument("--list_extract_paths", default='GlobalRank/rerank_path.txt', type=str)
    parser.add_argument("--image_size", default=(512,512), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--descriptor_output_dir', default='LocalFeaturePath/Descriptions', help='')
    parser.add_argument('--location_output_dir', default='LocalFeaturePath/Locations', help='')
    parser.add_argument('--use_crop', default=True, help='use query croping or not', type = utils.bool_flag)
    parser.add_argument('--overlap', default=0.5, help='', type = float)
    parser.add_argument('--num_scale', default=5, help='', type = int)
    parser.add_argument('--num_points', default=1000, help='', type = int)
    parser.add_argument('--model_re', default='xcit_retrievalv2_reduction_ae')
    parser.add_argument('--weight_path', 
              default='model_checkpoint/xcit_small_12_p16_retrieval_reduction_ae/checkpoint.pth') #_model_epoch_24

    

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # ===================build model
#     model = create_model(
#         args.model,
#         pretrained=False,
#         num_classes=1,
#     )

#     checkpoint = torch.load(args.pretrained, map_location='cpu')
#     checkpoint_model = checkpoint['model']
#     state_dict = model.state_dict()
#     for k in ['head.weight', 'local_head.weight','local_head.bias']:
#         if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]
#     model.load_state_dict(checkpoint_model, strict=False)
#     model.to(device)
#     model.eval() 
    model_backbone = create_model(
        args.model,
        pretrained=False,
        num_classes=0,
    )
    model = create_model(
        args.model_re,
        pretrained=False,
        num_classes=0,
        retrieval_back_bone = model_backbone,
        drop_block_rate=None,
    )

    device = torch.device("cuda")
    resume_path=args.weight_path
    checkpoint = torch.load(resume_path, map_location='cpu')['model']
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint.keys():
            print(f"Removing key {k} from pretrained model checkpoint")
            del checkpoint[k]
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    #####################################
    
    multi_scale = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0] if args.num_scale == 7 else \
                [0.3535, 0.5, 0.7071, 1.0, 1.4142] if args.num_scale == 5 else [0.7071, 1.0, 1.4142]
    if not args.use_multi_scale: multi_scale = [1]
    bbxs = None
    if args.list_extract_paths != None:
        all_paths = []
        bbxs = {}
        f = open(args.list_extract_paths)
        for path in f.readlines():
            path,x1,y1,x2,y2 = path.split('\n')[0].split(' ')
            all_paths.append(path)
            bbxs[path] = [x1,y1,x2,y2]
        all_paths = list(set(all_paths))
        all_paths = sorted(all_paths)
    else:
        all_paths = [ args.image_folders + '/' + dir for dir in os.listdir(args.image_folders)]
    print('Extract ' + str(len(all_paths)) + ' file')
    for idx, im_path in tqdm(enumerate(all_paths)):
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if args.use_crop:
            try:
                if bbxs != None:
                    x1,y1,x2,y2 = np.array(bbxs[im_path], dtype=np.int32)
                if x1 !=-1 and x2!=-1 and y1!=-1 and y2!=-1:
                    img = img[y1:y2,x1:x2,:]
                    print('Querry cropping')

            except:
                print(img_path)
                continue
        
        im_h, im_w, _ = np.array(img).shape
        scale_factor = max(im_h,im_w) / np.sqrt(im_h * im_w)
        image_size = args.image_size[0]
        resize_h, resize_w = int(image_size * scale_factor * im_h / max(im_h,im_w)),  \
                         int(image_size * scale_factor * im_w / max(im_h,im_w))
        resize_h, resize_w = resize_h - resize_h % args.patch_size, resize_w - resize_w % args.patch_size 
        resize_size = (resize_h, resize_w) if  args.keep_aspec_ratio else args.image_size


        transform = A.Compose([
                    A.Resize(resize_size[0],resize_size[1]),
                    A.Normalize(),
                    A.pytorch.transforms.ToTensorV2()
                ])    
        img = transform(image=img)['image'].to(device)
        
        img = img.unsqueeze(0)
        
        locations, descriptions, scores, scales = extract_local_descriptors(model, 
                img,args.patch_size,args.attn_threshold,args.min_attn_threshold, args.max_keypoint, multi_scale)
        im_path = im_path.replace('../','')
        im_path = im_path.replace('/', '__')
        out_file_location = args.location_output_dir + '/' + im_path
        np.save(out_file_location, locations)

        out_file_descriptions = args.descriptor_output_dir + '/' + im_path
        np.save(out_file_descriptions, descriptions)
        
