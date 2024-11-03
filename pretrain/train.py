import argparse
from datetime import datetime
import numpy as np
import os
import random
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from progress.bar import Bar
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from dataset import collate_batch, batch_process_coords, get_datasets, create_dataset
from model import create_model
from utils.utils import create_logger, load_default_config, load_config, AverageMeter
from utils.metrics import MSE_LOSS_MULTI_MODAL_UNIMOTION

def evaluate_loss(rank, model, dataloader, config):
    bar = Bar(f"EVAL", fill="#", max=len(dataloader))
    loss_avg = AverageMeter()
    dataiter = iter(dataloader)
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                joints, masks, padding_mask = next(dataiter)
            except StopIteration:
                break
            joints = joints.to(rank)
            masks = masks.to(rank)
            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config)
            padding_mask = padding_mask.to(rank)
            out_joints = out_joints.to(rank)

            loss,_ ,_ = compute_loss(rank, model, config, in_joints, out_joints, in_masks, out_masks, padding_mask)
            loss_avg.update(loss.item(), len(in_joints))
            
            summary = [
                f"({i + 1}/{len(dataloader)})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]

            bar.suffix = " | ".join(summary)
            bar.next()

        bar.finish()

    return loss_avg.avg

def compute_loss(rank, model, config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=None, mode='val', loss_last=True, optimizer=None):
    
    _, in_F, _, _ = in_joints.shape

    applymask = (mode == 'train')

    pred_trajs, pred_pose_3d, pose_mask, available_num_keypoints = model(in_joints, padding_mask, applymask=applymask)

    loss = MSE_LOSS_MULTI_MODAL_UNIMOTION(rank, pred_trajs[:,:,in_F:], pred_pose_3d[:,in_F:], out_joints, available_num_keypoints, pose_mask)

    return loss, pred_trajs, pred_pose_3d


def adjust_learning_rate(optimizer, epoch, config):
    """
    From: https://github.com/microsoft/MeshTransformer/
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs*2/3 = 100
    """
    lr = config['TRAIN']['lr'] * (config['TRAIN']['lr_decay'] ** epoch) #  (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
    if 'lr_drop' in config['TRAIN'] and config['TRAIN']['lr_drop']:
        lr = lr * (0.1 ** (epoch // (config['TRAIN']['epochs']*4./5.)  ))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print('lr: ',lr)
        
def save_checkpoint(model, optimizer, epoch, config, filename, logger):
    logger.info(f'Saving checkpoint to {filename}.')
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config
    }
    torch.save(ckpt, os.path.join(config['OUTPUT']['ckpt_dir'], filename))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def dataloader_for(dataset, config, **kwargs):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size']//config['DATA']['chunk_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      sampler=sampler, 
                      **kwargs)

def dataloader_for_val(dataset, config, **kwargs):

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size']//config['DATA']['chunk_size'],
                      num_workers=0,
                      collate_fn=collate_batch,
                      sampler=sampler, 
                      **kwargs)


def train(rank,config, logger,world_size, devices,experiment_name="", dataset_name=""):

    setup(rank, world_size)

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    dataset_train = ConcatDataset(get_datasets(config['DATA']['train_datasets'], config, logger))
    
    dataloader_train = dataloader_for(dataset_train, config, shuffle=False, pin_memory=True)
    logger.info(f"Training on a total of {config['DATA']['chunk_size']*len(dataset_train)} annotations.")

    dataset_val = create_dataset(config['DATA']['train_datasets'][0], logger, split="val", track_size=(in_F+out_F), track_cutoff=in_F)
    dataloader_val = dataloader_for(dataset_val, config, shuffle=False, pin_memory=True)


    writer_name = experiment_name + "_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    writer_train = SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_TRAIN"))
    writer_valid =  SummaryWriter(os.path.join(config["OUTPUT"]["runs_dir"], f"{writer_name}_VALID"))
    
    model = create_model(config, logger,rank).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if config["MODEL"]["checkpoint"] != "":
        logger.info(f"Loading checkpoint from {config['MODEL']['checkpoint']}")
        checkpoint = torch.load(os.path.join(config['OUTPUT']['ckpt_dir'], config["MODEL"]["checkpoint"]))
        model.load_state_dict(checkpoint["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} parameters.")
    print('num_parameters in the model: ',num_parameters)
    
    global_step = 0
    min_val_loss = 1e5 # init min_val_loss as a large number
    start_epoch = 0

    for epoch in range(start_epoch, config["TRAIN"]["epochs"]):

        start_time = time.time()
        dataiter = iter(dataloader_train)

        timer = {"DATA": 0, "FORWARD": 0, "BACKWARD": 0}

        loss_avg = AverageMeter()

        if config["TRAIN"]["optimizer"] == "adam":
            adjust_learning_rate(optimizer, epoch, config)

        train_steps =  len(dataloader_train)

        bar = Bar(f"TRAIN {epoch}/{config['TRAIN']['epochs'] - 1}", fill="#", max=train_steps)
        
        for i in range(train_steps): 
            model.train()
            optimizer.zero_grad()
            start = time.time()

            try:
                joints, masks, padding_mask = next(dataiter)

            except StopIteration:
                dataiter = iter(dataloader_train)
                joints, masks, padding_mask = next(dataiter)

            in_joints, in_masks, out_joints, out_masks, padding_mask = batch_process_coords(joints, masks, padding_mask, config, training=True)            
            timer["DATA"] = time.time() - start
            start = time.time()
            padding_mask = padding_mask.to(rank)
            in_joints = in_joints.to(rank)
            loss, _, _ = compute_loss(rank, model, config, in_joints, out_joints, in_masks, out_masks, padding_mask, epoch=epoch, mode='train', optimizer=None)            
            timer["FORWARD"] = time.time() - start


            start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["TRAIN"]["max_grad_norm"])
            optimizer.step()
                
            timer["BACKWARD"] = time.time() - start

            loss_avg.update(loss.item(), len(joints))
            
            summary = [
                f"{str(epoch).zfill(3)} ({i + 1}/{train_steps})",
                f"LOSS: {loss_avg.avg:.4f}",
                f"T-TOT: {bar.elapsed_td}",
                f"T-ETA: {bar.eta_td:}"
            ]
            

            for key, val in timer.items():
                 summary.append(f"{key}: {val:.2f}")

            bar.suffix = " | ".join(summary)
            bar.next()

         
            
        bar.finish()

        global_step += train_steps

        writer_train.add_scalar("loss", loss_avg.avg, epoch)

        if rank == 0:
            temp_name = 'pre_train'  
            if epoch % 5 == 0:
                val_loss = evaluate_loss(rank, model, dataloader_val, config)
                writer_valid.add_scalar("loss", val_loss, epoch)
                val_ade = val_loss/100
                if val_ade < min_val_loss:
                    min_val_loss = val_ade
                    print('------------------------------BEST MODEL UPDATED------------------------------')
                    print('Best MSE: ', val_ade)
                    save_checkpoint(model, optimizer, epoch, config, temp_name+'_best_val'+'_checkpoint.pth.tar', logger)

            print('time for training: ', time.time()-start_time)
            print('epoch ', epoch, ' finished!')
        
    save_checkpoint(model, optimizer, epoch, config, temp_name+'_checkpoint.pth.tar', logger)
    logger.info("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
    parser.add_argument("--cfg", type=str, default="", help="Config name. Otherwise will use default config")
    parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")

    args = parser.parse_args()

    if args.cfg != "":
        cfg = load_config(args.cfg, exp_name=args.exp_name)
    else:
        cfg = load_default_config()

    cfg['dry_run'] = args.dry_run

    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])

    devices = ['cuda:0','cuda:1','cuda:2','cuda:3','cuda:4','cuda:5','cuda:6','cuda:7']
    if torch.cuda.is_available():
        cfg["DEVICE"] = devices
    else:
        cfg["DEVICE"] = "cpu"

    dataset = cfg["DATA"]["train_datasets"]

    logger = create_logger(cfg["OUTPUT"]["log_dir"])
    logger.info("Initializing with config:")
    logger.info(cfg)

    num_gpus = len(devices)
    mp.spawn(train,
            nprocs=num_gpus,
            args=(cfg, logger, num_gpus, devices, args.exp_name, dataset))






