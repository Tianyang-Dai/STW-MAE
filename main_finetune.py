import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
assert timm.__version__ == "0.3.2"
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit
from engine_finetune import train_one_epoch
import MyDataset
import torchvision.transforms as transforms
import utils
import torch.nn as nn
from torch.autograd import Variable
import wandb
import scipy.io as io
from utils_sig import *


def get_args_parser():
    parser = argparse.ArgumentParser('STW-MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
  
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--finetune', default='/media/26d532/keke/mae-main/mae-main/pretrain_VIPLST_lossIMG_new/checkpoint-150.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--nb_classes', default=220, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./supervise_VIT_VIPL_LossCrossEntropy',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./supervise_VIT_VIPL_LossCrossEntropy',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
  
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dataname', type=str, default="VIPL", help='log and save model name')
    parser.add_argument('--Map_name1', type=str, default='STMap_RGB_Align_CSI_CHROM.png', help='log and save model name')
    parser.add_argument('--Map_name2', type=str, default='WaveletMap_RGB_Align_CSI.png', help='log and save model name')
    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=224,
                        help='the num of frames')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('--log', type=str, default="supervise_VIT_VIPL_LossCrossEntropy", help='log and save model name')
    parser.add_argument('--loss_type', type=str, default="rppg", help='loss type')
    parser.add_argument('-rD', '--reData', dest='reData', type=int, default=0,
                        help='re Data')
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--semi', type=str, default='')
    
    return parser


def main(args):
    # misc.init_distributed_mode(args)
    if args.dataname=='VIPL':
        fileRoot = r'/scratch/project_2006419/data/VIPL_processed'
        # saveRoot = r'/scratch/project_2006419/data/VIPL_Index/fs_VIPL_Map' + str(args.fold_num) + str(args.fold_index)
        saveRoot = r'/scratch/project_2006419/data/VIPL_Index/VIPL_Map50'
    if args.dataname=='PURE':
        fileRoot = r"/STW-MAE/data/Map/PURE"
        saveRoot = r"/STW-MAE/data/Map50/PURE"
    if args.dataname=='UBFC':
        fileRoot = r'/scratch/project_2006419/data/UBFC_Map/UBFC_ST'
        saveRoot = r'/scratch/project_2006419/data/UBFC_Map/UBFC_Index/UBFC_Map50'

    wandb.init(project=args.dataname+"_finetunenew", entity="rppg" ,name =args.log)
    wandb.config = {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size
                    }
    best_mae =20
    frames_num = args.frames_num
    dataname=args.dataname
    fold_num=args.fold_num
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    # resize = transforms.Resize(size=(64, frames_num))
    resize = transforms.Resize(size=(frames_num, frames_num))

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # device = 'cpu'
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
  
    if args.reData == 1:
        if args.semi:
            test_index, train_index, semi_withlabel_index, semi_withoutlabel_index = MyDataset.CrossValidation_semi(fileRoot, fold_num ,fold_index=0,semi=2, semi_index=0)
            semi_with = MyDataset.getIndex(fileRoot, semi_withlabel_index, saveRoot + '_1Train50%', 'Map.png', 5, frames_num)
            semi_without = MyDataset.getIndex(fileRoot, semi_withoutlabel_index, saveRoot + '_2Train50%', 'Map.png', 5, frames_num)
        else:
            test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=5, fold_index=0)
            Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'Map.png', 5, frames_num)
            Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'Map.png', 5, frames_num)
     
    if args.semi:
        dataset_train = MyDataset.Data_DG(root_dir=(saveRoot + '_Train'+ args.semi),dataName=dataname,Map1 =args.Map_name1, Map2 =args.Map_name2, \
            in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
        dataset_val = MyDataset.Data_DG(root_dir=(saveRoot + '_Test'),dataName=dataname,Map1 =args.Map_name1, Map2 =args.Map_name2, \
            in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    else:
        dataset_train = MyDataset.Data_DG(root_dir=(saveRoot + '_Train'),dataName=dataname,Map1 =args.Map_name1, Map2 =args.Map_name2, \
            in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
        dataset_val = MyDataset.Data_DG(root_dir=(saveRoot + '_Test'),dataName=dataname,Map1 =args.Map_name1, Map2 =args.Map_name2, \
            in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    print('trainLen:', len(dataset_train), 'testLen:', len(dataset_val))
    print('fold_num:', args.fold_num, 'fold_index', args.fold_index)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        in_chans=args.in_chans
    )

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
      
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
      
        model.eval()
        HR_pr_temp = []
        HR_rel_temp = []
        for step, (data1, bvp, HR_rel) in enumerate(data_loader_val):
            # data = Variable(data).float().to(device=device)
            data1 = Variable(data1).float().to(device=device)
            # data2 = Variable(data2).float().to(device=device)
            bvp = Variable(bvp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            Wave = bvp.unsqueeze(dim=1)
            Map = data1[:, :, :, 0:frames_num]
            Wave = Wave[:, :, 0:frames_num]
            b, _, _ = Wave.size()

            outputs = model(data1)  # [B,220]

            if args.loss_type=='rppg':
                loss_func_rPPG = utils.P_loss3().to(device)
                loss_func_SP = utils.SP_loss(device, low_bound=36, high_bound=240,clip_length=args.frames_num).to(device)
                _, hr_pr= loss_func_SP(outputs.unsqueeze(dim=1), HR_rel)
                _, hr_rel= loss_func_SP(Wave, HR_rel)
                loss = loss_func_rPPG(outputs.unsqueeze(dim=1), Wave)
                HR_pr_temp.extend(hr_pr.data.cpu().numpy())
                HR_rel_temp.extend(hr_rel.data.cpu().numpy())
            if args.loss_type=='SP':
                loss_func_SP = utils.SP_loss(device, low_bound=36,high_bound=240, clip_length=args.frames_num).to(device)
                loss, hr_pre= loss_func_SP(outputs.unsqueeze(dim=1), HR_rel)
                HR_pr_temp.extend(hr_pre.data.cpu().numpy())
                HR_rel_temp.extend(HR_rel.data.cpu().numpy())
        print('loss_test: ', loss)
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
        wandb.log({"MAE": MAE,'epoch': epoch})
        if best_mae > MAE:
            best_mae = MAE
            io.savemat(args.log+'/' + 'HR_pr.mat', {'HR_pr': HR_pr_temp})
            io.savemat(args.log+'/' + 'HR_rel.mat', {'HR_rel': HR_rel_temp})
            print('save best predict HR')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        # **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
