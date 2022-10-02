import copy
import io
import argparse
import os

import torch
print(torch.__version__)
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms

from PIL import Image

from scipy import spatial
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io as skio
import numpy as np

import utils
import cv2
from tqdm import tqdm
import pickle
from timm.models import create_model

import albumentations.pytorch
import albumentations as A
import pydegensac
from functools import cmp_to_key

import xcit_retrieval
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




def compute_putative_matching_keypoints(test_keypoints,
                                        test_descriptors,
                                        train_keypoints,
                                        train_descriptors,
                                        use_ratio_test,
                                        matching_threshold,
                                        max_distance):
    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""
    train_descriptor_tree = spatial.cKDTree(train_descriptors)
    if use_ratio_test:
        distances,matches=train_descriptor_tree.query(
            test_descriptors,k=2,n_jobs=-1
        )
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints=np.array([
            test_keypoints[i,]
            for i in range(test_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])
        train_matching_keypoints=np.array([
            train_keypoints[matches[i][0],]
            for i in range(test_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])

    else:
        dist, matches = train_descriptor_tree.query(
              test_descriptors, distance_upper_bound=max_distance)
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints = np.array([
              test_keypoints[i,]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
        train_matching_keypoints = np.array([
              train_keypoints[matches[i],]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
    return test_matching_keypoints, train_matching_keypoints 



def compute_num_inliers(test_keypoints, 
                        test_descriptors, 
                        train_keypoints,
                        train_descriptors,
                        max_reprojection_error,
                        homography_confidence,
                        max_ransac_iteration,
                        use_ratio_test,
                        matching_threshold,
                        max_distance,
                        query_im_array=None,
                        index_im_array=None,
                        test_scales=None, test_scores=None, 
                        train_scales=None, train_scores=None,
                      ):
    """Returns the number of RANSAC inliers.""" 
        
    test_match_kp, train_match_kp = \
            compute_putative_matching_keypoints(test_keypoints, 
                                                test_descriptors, 
                                                train_keypoints, 
                                                train_descriptors,
                                                use_ratio_test=use_ratio_test,
                                                matching_threshold = matching_threshold,
                                                max_distance =  max_distance,
                                                )
    if test_match_kp.shape[0] <= 4:  
        return 0, b''

    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            max_reprojection_error,
                                            homography_confidence,
                                            max_ransac_iteration)
    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
        return 0, b''

    inliers = mask if mask is not None else []

    match_viz_bytes = b''
    if isinstance(query_im_array, np.ndarray) and isinstance(index_im_array, np.ndarray) :
        query_im_scale_factors = [1.0, 1.0]
        index_im_scale_factors = [1.0, 1.0]
        inlier_idxs = np.nonzero(inliers)[0]
        _, ax = plt.subplots()
        ax.axis('off')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        feature.plot_matches(
            ax,
            query_im_array,
            index_im_array,
            test_match_kp * query_im_scale_factors,
            train_match_kp * index_im_scale_factors,
            np.column_stack((inlier_idxs, inlier_idxs)),
            only_matches=True)

        match_viz_io = io.BytesIO()
        plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
        match_viz_bytes = match_viz_io.getvalue()
    
    return int(copy.deepcopy(mask).astype(np.float32).sum()), match_viz_bytes
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
            
    keep_idx, count = nms(torch.tensor(selected_locations), torch.tensor(all_local_scores), overlap=0.5, top_k=1200)
    keep_idx = keep_idx.cpu().numpy()
    keep_idx = keep_idx[:min(1000,count)]
    selected_locations = np.array(selected_locations)[keep_idx]
    selected_locations = np.array([((y1+y2)//2,(x1+x2)//2) for (y1,x1,y2,x2) in selected_locations])
    print('Num_feature detected,' ,count)
    return selected_locations,np.array(selected_features)[keep_idx],None,None

    
    

def preprocess_image(args, im_path, image_size = None, box = None) :
    patch_size = args.patch_size
    if image_size is None :
        image_size = args.image_size
    
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if box is not None:
        x1,y1,x2,y2 = box
        img = img[y1:y2,x1:x2]
            
    im_h, im_w, _ = np.array(img).shape
    scale_factor = max(im_h,im_w) / np.sqrt(im_h * im_w)
    resize_h, resize_w = int(image_size * scale_factor * im_h / max(im_h,im_w)),  \
                         int(image_size * scale_factor * im_w / max(im_h,im_w))
    resize_h, resize_w = resize_h - resize_h % patch_size, resize_w - resize_w % patch_size # divisible by the patch size
    resize_size = np.array([resize_h, resize_w]) if  args.keep_aspec_ratio else np.array([image_size,image_size])
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
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_original', default=384, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--layer_names', default=['self','cross'] * 6, type=list)
                        
    
    # io config
    parser.add_argument("--keep_aspec_ratio", default=True, type=utils.bool_flag, help="Keep aspec_ratio.")
    parser.add_argument("--use_multi_scale", default=True, type=utils.bool_flag, help="Keep aspec_ratio.")
    parser.add_argument('--data_path', default='../test_datasets', type=str)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--image_root_paths', default='../test_datasets/roxford5k/jpg', help='')
    parser.add_argument("--image_size", default= 512, type=int, nargs="+", help="Resize image.")
    parser.add_argument("--quer_idx", default= 0, type=int, help="Querry idx to visualize")
    parser.add_argument('--output_dir', default='MatchPairVis',type=str, help='Path where to save visualizations.')
    
    parser.add_argument('--descriptor_output_dir', default='', help='')

    # ransac config
    parser.add_argument("--max_reprojection_error", default=16, type=float)
    parser.add_argument("--max_ransac_iteration", default=2000, type=int)
    parser.add_argument("--homography_confidence", default=1.0, type=float)
    parser.add_argument("--max_distance", default=0.95, type=float)
    
    parser.add_argument("--matching_ratio_threshold", default=0.97, type=float)
    parser.add_argument("--use_ratio_test", default=False, type=utils.bool_flag)
    parser.add_argument("--attn_threshold", default = 60, type=float)
    parser.add_argument("--min_attn_threshold", default = 0.6 , type=float)
    parser.add_argument("--max_keypoint", default=400, type=int)
    
    device = torch.device("cuda")
    args = parser.parse_args()
    model_backbone = create_model(
        args.model,
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
    
    resume_path='../model_checkpoint/xcit_small_12_p16_retrieval_reduction_ae/checkpoint_model_epoch_24.pth'
    checkpoint = torch.load(resume_path, map_location='cpu')['model']
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint.keys():
            print(f"Removing key {k} from pretrained model checkpoint")
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    ################
    
                        
    galery_paths = []
    all_paths = []
    pair_descriptors = []
    pair_locations = []
    pair_scales = []
    pair_scores = []
    pair_imgs = []
    multi_scale = [0.3535, 0.5, 0.7071, 1.0, 1.4142] if args.use_multi_scale else None
    # specific query and galery path
    quer_idx = int(args.quer_idx)
    assert args.data_path != '', " not find data path"
    gnd_fname = os.path.join(args.data_path, args.dataset, 'gnd_{}.pkl'.format(args.dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
        
    querry_image_path = os.path.join(args.data_path, args.dataset, 'jpg', cfg["qimlist"][quer_idx] + ".jpg")
    gt_for_querry_image_paths =  np.concatenate([cfg['gnd'][quer_idx]['hard']])[:75] 

    print('num hard', len(cfg['gnd'][quer_idx]['hard']))
    all_paths.append(querry_image_path)
    for im_path in gt_for_querry_image_paths: 
        all_paths.append(os.path.join(args.data_path, args.dataset, 'jpg', cfg["imlist"][int(im_path)] + ".jpg"))
    for im_idx,im_path in tqdm(enumerate(all_paths)):
        box = None
        if im_idx == 0 :
            box =  np.array(cfg['gnd'][quer_idx]['bbx'], dtype= np.int32)
            x1,y1,x2,y2 = box    
            cv2.imwrite('MATCHING_PLOT_IM/quer.jpg',
                        cv2.rectangle(cv2.imread(im_path), (x1,y1), (x2,y2), (0, 0, 255), 2))
        cv2.imwrite('MATCHING_PLOT_IM/'+ str(im_idx)+'.jpg',cv2.imread(im_path))
          
        orig_img ,img_input = preprocess_image(args, im_path, box=box)
        location, description, scores, scales = extract_local_descriptors(model, img_input, 
                    args.patch_size,args.attn_threshold,args.min_attn_threshold, args.max_keypoint,multi_scale)
            
        
        
        pair_descriptors.append(description)
        pair_locations.append(location)
        pair_imgs.append(orig_img)
        pair_scales.append(scales)
        pair_scores.append(scores)



    querry_location = pair_locations[0]
    querry_description = pair_descriptors[0]
    querry_scales = pair_scales[0]
    querry_scores = pair_scores[0]
    query_img = pair_imgs[0]
    
    num_inliers = []
    for idx in range(len(pair_descriptors) -1):
        gal_img = pair_imgs[idx + 1]
        num_inlier, match_vis_bytes = compute_num_inliers(querry_location, querry_description,
                                                        pair_locations[idx + 1], pair_descriptors[idx + 1],
                                                        max_reprojection_error = args.max_reprojection_error,
                                                        homography_confidence = args.homography_confidence,
                                                        max_ransac_iteration = args.max_ransac_iteration,
                                                        use_ratio_test = args.use_ratio_test,
                                                        matching_threshold = args.matching_ratio_threshold,
                                                        max_distance = args.max_distance,
                                                        query_im_array = query_img,
                                                        index_im_array = gal_img,
                                                        test_scales=querry_scales,test_scores=querry_scores, 
                                                        train_scales=pair_scales[idx + 1],
                                                        train_scores=pair_scores[idx + 1],
                                                         )
        print(idx,'Num inlier : ', num_inlier)
        with open(os.path.join(args.output_dir, f'save_match_{idx}.jpg'),"wb") as fout:
            fout.write(match_vis_bytes)
        num_inliers.append(num_inlier)
    print('query_num_inlier', num_inliers)
    ######
