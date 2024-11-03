import argparse
import torch
import random
import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader

from dataset import create_dataset, collate_batch, batch_process_coords_eval
from model import  create_model_eval
from utils.utils import create_logger

def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()
    
    with torch.no_grad():
        pred_traj, pred_pose_3d, pose_mask, available_num_keypoints = model(input_joints, padding_mask)
    output_traj = pred_traj[:,:,-out_len:]

    return output_traj


def evaluate_ade_fde(model, dataloader, bs, config, logger, return_all=False, bar_prefix="", per_joint=False, show_avg=False):
    in_F, out_F = 10, 20
    bar = Bar(f"EVAL ADE_FDE", fill="#", max=len(dataloader))

    num_test_samples = 0
    batch_id = 0
    ade = 0
    fde = 0
    ade_batch = 0 
    fde_batch = 0

    for i, batch in enumerate(dataloader):
        joints, masks, padding_mask = batch
        padding_mask = padding_mask.to(config["DEVICE"])
   
        in_joints, in_masks, out_joints, out_masks, padding_mask, last_ego = batch_process_coords_eval(joints, masks, padding_mask, config)
        

        pred_joints = inference(model, config, in_joints, padding_mask, out_len=out_F)
        out_joints = out_joints.cpu() 

        pred_joints = pred_joints.cpu().permute(1,0,2,3) 

        for k in range(len(out_joints)):

            gt_xy = out_joints[k,:,0,:2] 
            pred_xy = pred_joints[k,:,:,:2]
            _,temp_K,temp_out_F,_ = pred_joints.shape
            norm_multi_modal = torch.full((temp_K, temp_out_F), float('nan')).to(pred_joints.device)
           
            for i in range(temp_K):
                norm_multi_modal[i,:] = torch.norm(pred_xy[i,:] - gt_xy, p=2, dim=-1)

            norm_multi_modal_mean = torch.mean(norm_multi_modal,dim=-1)
            norm,ind_min = torch.min(norm_multi_modal_mean,dim=-1) #out_F

            ade_batch += norm

            scene_fde,_ = torch.min(norm_multi_modal[:,-1],dim=0)

            fde_batch += scene_fde
            num_test_samples+=1

        if batch_id % 1e2 == 0:
            print(f'Finished {batch_id*bs*4/1000} k samples')

        batch_id+=1

    ade = ade_batch/num_test_samples
    fde = fde_batch/num_test_samples

    return ade, fde, num_test_samples


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,  help="checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Split to use. one of [train, test, valid]")
    parser.add_argument("--metric", type=str, default="vim", help="Evaluation metric. One of (vim, mpjpe)")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
        
    logger = create_logger('')
    logger.info(f'Loading checkpoint from {args.ckpt}') 
    ckpt = torch.load(args.ckpt, map_location = torch.device('cpu'))
    config = ckpt['config']
    
    if torch.cuda.is_available():
        config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
        torch.cuda.manual_seed(0)
    else:
        config["DEVICE"] = "cpu"


    logger.info("Initializing with config:")
    logger.info(config)

    model = create_model_eval(config, logger)
    pretrained_dict = ckpt['model']
    pretrained_dict={key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict) 
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    assert in_F == 10
    assert out_F == 20

    # (a) 1 foot = 0.3048 meters, (b) 94 feet = 28 meters
    # To fairly compare with leapfrog's implementation on NBA, we rescale the length unit from (a) to (b)
    # Leapfrog: https://github.com/MediaBrain-SJTU/LED/blob/aae048a85292a24c2218de5fe9b4d20d3542bf04/visualization/draw_mean_variance.ipynb#L34
    rescale = (1/0.3048)*(28/94) 

    name = config['DATA']['train_datasets']
    
    dataset = create_dataset(name[0], logger, split=args.split, track_size=(in_F+out_F), track_cutoff=in_F)

    bs = config['TRAIN']['batch_size']//4 # chunk_size = 4
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=config['TRAIN']['num_workers'], shuffle=False, collate_fn=collate_batch)

    ade,fde,num_samples = evaluate_ade_fde(model, dataloader, bs, config, logger, return_all=True)
    
    print('ADE: ', ade*rescale)
    print('FDE: ', fde*rescale)
    print('Test samples: ', num_samples)


