# ViTGaL-pytorch
Pytorch Implementation of ViTGaL (Patch Embedding as Local Features: Unifying Deep Local and Global Features Via Vision Transformer for Image Retrieval)



## Prerequisites

+ PyTorch
+ python3
## Build pydegensac
We use pydegensac https://github.com/ducha-aiki/pydegensac for reranking, but unfortunately, the seed is not fixed, so you got different results at different running. To solve this, we fix all the seeds to 0 (see new code in the pydegensac folder). You can build it by the following command:
```
cd pydegensac
python3 ./setup.py install
```
Or choose to install pydegensac with pip install (the result will different minor from those report in the paper)
## Training
 Dataset:

+ Download GLD2-clean dataset from https://www.kaggle.com/competitions/landmark-retrieval-2021
+ Create train-test split. The ouput will be two txt file containing images path and their label for training and evaluation
```
python create_train_split.py --data_path path_to_gldv2clean
```


Training a ViTGaL model:
+ Download pretrained weight on ImageNet from https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_224.pth (for S12P16) and https://dl.fbaipublicfiles.com/xcit/xcit_small_24_p16_224.pth (for S24P16)
+ Training ViTGaL

```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --model xcit_retrievalv2_small_12_p16 --batch-size 64  --output_dir model_checkpoint/xcit_retrievalv2_small_12_p16 --epochs 40 --train_split train_split.txt --val_split test_split.txt
```
+ Training Autoencoder
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_reduction_ae.py --model_backbone xcit_retrievalv2_small_12_p16 --model xcit_retrievalv2_reduction_ae --batch-size 64  --output_dir model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae --pretrained_backbone path_to_pretrained_vitgal --epochs 30 --train_split train_split.txt --val_split test_split.txt
```

## Trained Weights  on GLDv2

-XCiT-S12/16-ViTGaL+AutoEncoder(Download weight from https://drive.google.com/file/d/1IiNwn9HBkL4hHbOfJhxVQf8Plp5ET58m/view?usp=sharing and put it in model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae) 

-XCiT-S24/16-ViTGaL+AutoEncoder(Download weight from https://drive.google.com/file/d/1BflEX7BhVybi-Ioz2TFE36m5y7SXYdjg/view?usp=sharing and put it in model_checkpoint/xcit_retrievalv2_small_24_p16_reduction_ae)  





## Global Evaluation on ROxf and RPar

###  Download Roxford, Ppar and 1M distractor dataset
+ Follow instruction in https://github.com/filipradenovic/revisitop to download. Your folder will be as follow:
```
test_datasets
└── revisitop1m
    └── jpg (folder of images)
    └── .txt file
└── roxford5k
    └── jpg (folder of images)
    └── .pkl file
    └──
└── rparis6k
    └── jpg (folder of images)
    └── .pkl file
```


### Evaluate on roxford

+ First, create the folder to save your global reranking results. After running the evaluation code, this path will contain the path to the top 100 reranking images for later local feature extraction (so you only need to extract local features from that path only instead of extracting for all images). The saved global ranking results folder will be as follow:
```
GlobalRank
    └──rerank_path.txt (path to top 100 image of each querrym used to extract local feature)
    └──global_rank.npy (global ranking result)
```
+ Evaluate on Roxford5k using XCiT-S12/16-ViTGaL+AutoEncoder(similar for paris6k)
```
python eval_image_retrieval.py --r1m_path None --data_path ../test_datasets  --dataset roxford5k --save_global_rank GlobalRank/global_rank.npy --do_reranking False --model xcit_retrievalv2_small_12_p16 --weight_path model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae/checkpoint.pth
```

+ Evaluate on Roxford5k+1M using XCiT-S12/16-ViTGaL+AutoEncoder(similar for Rparis6k+1M)

```
python eval_image_retrieval.py --r1m_path ../test_datasets/revisitop1m/jpg --data_path ../test_datasets  --dataset roxford5k --save_global_rank GlobalRank/global_rank.npy --do_reranking False --model xcit_retrievalv2_small_12_p16 --weight_path model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae/checkpoint.pth
```

## Local Feature Extraction

#### Extract local features for reranking

+ Extract local feature from global ranking result. Local feature will be stored in a folder with format below :
```
LocalFeaturePath
    └──Descriptions
        └── image_path.npy (npy file store local feature description, name is image path)
    └──Locations
        └──image_path.npy (npy file store local feature location, name is image path)
```
+  Commandline for Roxford5k extraction using XCiT-S12/16-ViTGaL+AutoEncoder is as below (similar for other setting):



```
python extract_local_feature.py --image_folder ../test_datasets/roxford5k/jpg --list_extract_paths GlobalRank/rerank_path.txt --weight_path model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae/checkpoint.pth --descriptor_output_dir LocalFeaturePath/Descriptions --location_output_dir LocalFeaturePath/Locations
```

## Reranking evaluation

With global retrieval result and local feature extraction. Reranking for each settings is as follow:

+ Roxford5k using XCiT-S12/16-ViTGaL+AutoEncoder (similar for 4 other settings)

```
python eval_image_retrieval.py --r1m_path None --data_path ../test_datasets  --dataset roxford5k --save_global_rank GlobalRank/global_rank.npy --do_reranking True --model xcit_retrievalv2_small_12_p16 --weight_path model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae/checkpoint.pth --descriptor_output_dir LocalFeaturePath/Descriptions --location_output_dir LocalFeaturePath/Locations --max_distance 0.82 --load_precomputed_global_rank True
```

+ Roxford5k using XCiT-S24/16-ViTGaL+AutoEncoder (similar for 4 other settings)

```
python eval_image_retrieval.py --r1m_path None --data_path ../test_datasets  --dataset roxford5k --save_global_rank GlobalRank/global_rank.npy --do_reranking True --model xcit_retrievalv2_small_24_p16 --weight_path model_checkpoint/xcit_retrievalv2_small_12_p16_reduction_ae/checkpoint.pth --descriptor_output_dir LocalFeaturePath/Descriptions --location_output_dir LocalFeaturePath/Locations --max_distance 0.79 --load_precomputed_global_rank True
```


## Matching visualization

To visulaize matching between querry (quer_idx between 0 and 69) and hard gallery image, run
```
python match_pair.py --quer_idx 20 --output_dir MatchPairVis
```
