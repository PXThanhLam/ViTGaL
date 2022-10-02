import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from shutil import copyfile

def get_args_parser():
    parser = argparse.ArgumentParser('Create split', add_help=False)
    parser.add_argument('--data_path', default='../gglandmark-v2-clean', type=str)
    parser.add_argument('--split_ratio', default=0.2, type=float)
    
    return parser
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('XCiT Retrieval training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join(args.data_path,'train.csv'))
    train_list = []
    val_list = []
    current_label = -1
    current_label_idx = -1
    labels = df['landmark_id'].values
    im_files = df['id'].values
    for idx in tqdm(np.argsort(labels)):
        imfile, label = im_files[idx], labels[idx]
        imfile = 'train/' + imfile[0] + '/' + imfile[1] + '/' + imfile[2] + '/' + imfile +'.jpg'
        if label > current_label :
            current_label_idx += 1
            current_label = label
            train_list.append([imfile, current_label_idx])
        else:
            if np.random.uniform() >= args.split_ratio:
                train_list.append([imfile, current_label_idx])
            else:
                val_list.append([imfile, current_label_idx])
        # if current_label_idx >= 17999:
        #     break
    f_train = open(args.data_path + '/train_split.txt', 'w+')
    f_test = open(args.data_path + '/test_split.txt', 'w+')
    for train_data in tqdm(train_list):
        file, label = train_data
        f_train.write(file + ' ' + str(label) + '\n')
    for test_data in tqdm(val_list):
        file, label = test_data
        f_test.write(file + ' ' + str(label) + '\n')
        
        
    #     for sample in train_list :
#         if not os.path.exists(args.data_path + '/view/' + str(sample[1])):
#             os.mkdir(args.data_path + '/view/' + str(sample[1]))
#         copyfile(args.data_path + '/' + sample[0], args.data_path + '/view/' + str(sample[1]) + '/' +                                  sample[0].split('/')[-1])

                
                
            
        
        
    
    