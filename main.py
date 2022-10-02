import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import RetrievalDataSet
from engine import train_one_epoch, evaluate
from losses import RetrievalLoss
import utils

import xcit_retrieval


def get_args_parser():
    parser = argparse.ArgumentParser('XCiT Retrieval training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    # Model parameters
    parser.add_argument('--model', default='xcit_retrievalv2_small_12_p16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default= 512, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.005,
                        help='weight decay (default: 0.005)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 3e-5), lr will be linear scale')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=3e-5, metavar='LR',
                        help='warmup learning rate (default: 3e-5)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                        help='epoch interval to decay LR, use in step learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=2, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=2, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--compress_lower', type=float, default=99, metavar='PCT',
                        help='Image compress lower (default: 99)')
    parser.add_argument('--compress_upper', type=float, default=100, metavar='PCT',
                        help='Image compress upper (default: 100)')
    parser.add_argument('--shift-limit', type=float, default=0.2, metavar='PCT',
                        help='shifting augment (default: 0.2)')
    parser.add_argument('--scale-limit', type=float, default=0.3, metavar='PCT',
                        help='scale augment (default: 0.3)')
    parser.add_argument('--rotate-limit', type=float, default=10, metavar='PCT',
                        help='rotate augment (default: 10)')
    parser.add_argument('--shift-roate-scale-p', type=float, default=0.7, metavar='PCT',
                        help='shift-roate-scale augment prop (default: 0.7)')
    parser.add_argument('--cutout_size', type=float, default=0.25, metavar='PCT',
                        help='max cutout relative size (default: 0.25)')
    parser.add_argument('--cutout_p', type=float, default=0.5, metavar='PCT',
                        help='cutout apply prob (default: 0.5)')

    parser.add_argument('--crop_scale', type=float, default=0.9, metavar='PCT',
                        help='min crop scale, defualt = 0.9)')
    parser.add_argument('--crop_p', type=float, default=0.5, metavar='PCT',
                        help='apply crop prob, defualt = 0.5')
        
    

    # Dataset parameters
    parser.add_argument('--data-path', default='../gglandmark-v2-clean', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default= 81313, type=int,
                        help='Number of class')
    parser.add_argument('--train_split', default= 'train_split.txt', type=str,
                        help='Number of class')
    parser.add_argument('--val_split', default= 'test_split.txt', type=str,
                        help='Number of class')
    parser.add_argument('--output_dir', default='model_checkpoint/xcit_small_12_p16_retrieval',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--test-freq', default=4, type=int, help='Number of epochs between \
                                                                  validation runs.')

    parser.add_argument('--full_crop', action='store_true', help='use crop_ratio=1.0 instead of the\
                                                                  default 0.875 (Used by CaiT).')
    parser.add_argument("--pretrained", default='../model_checkpoint/xcit_small_12_p16_224.pth', type=str, help='Path                                  to pretrained checkpoint')

    return parser



def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = RetrievalDataSet(args, mode = 'train')
    dataset_val = RetrievalDataSet(args, mode = 'val')

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(0.5*args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None
    )

    if args.pretrained:
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
           
    

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = RetrievalLoss()

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    resume_path = os.path.join(output_dir, 'checkpoint.pth') 
    if args.resume and os.path.exists(resume_path):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading from checkpoint ...")
            checkpoint = torch.load(resume_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint_optimizer = checkpoint['optimizer']
            checkpoint_scheduler = checkpoint['lr_scheduler']
            for opt_param in checkpoint_optimizer['param_groups']:
                opt_param['lr'] = args.lr
                opt_param['initial_lr'] = args.lr
            checkpoint_scheduler['lr_min'] = args.min_lr
            checkpoint_scheduler['t_initial'] = args.epochs
            checkpoint_scheduler['base_values'] = [args.lr,args.lr]
         
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler.step(checkpoint['epoch'])
            
            print(optimizer.state_dict()['param_groups'])
            print(lr_scheduler.state_dict())
            
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                print('Load scaler form checkpoint',loss_scaler.state_dict())

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, output_dir=output_dir )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, os.path.join(output_dir, f'checkpoint_model_epoch_{epoch}.pth'))
            
        if (epoch % args.test_freq == 0) or (epoch == args.epochs):
            test_stats = evaluate(data_loader_val, model, device)
            if test_stats["acc1"] >= max_accuracy:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(output_dir, 'best_model.pth'))

            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('XCiT Retrieval training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)