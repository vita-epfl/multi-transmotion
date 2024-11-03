import torch

def MSE_LOSS_MULTI_MODAL_UNIMOTION(rank, output_traj, output_pose_3d, target,available_num_keypoints,mask=None):

    pred_xy_k = output_traj.permute(1,0,2,3).to(rank) 
    B,K_modal, out_F,k = pred_xy_k.shape

    gt_traj = target[:,:,0,:2].to(rank)

    norm_multi_modal = torch.full((B,K_modal, out_F), float(1e3)).to(rank)
    for i in range(K_modal):
        norm_multi_modal[:,i,:] = torch.norm(pred_xy_k[:,i,:] - gt_traj, p=2, dim=-1)
    mean_K = torch.mean(norm_multi_modal, dim=-1)
    norm_traj,_ = torch.min(mean_K,dim=-1)

    _, ouf_F_pose, num_avail_j, _ = output_pose_3d.shape 
    

    gt_pose_3d_pred = target[:,:ouf_F_pose,3:42,:3].to(rank) 
    mask_3d_p = mask.unsqueeze(0).unsqueeze(0).repeat(B,ouf_F_pose,1).to(rank)
    gt_pose_3d_available = gt_pose_3d_pred[mask_3d_p].view(B,ouf_F_pose,available_num_keypoints,3).to(rank)

    nan_mask = ~torch.isnan(gt_pose_3d_available[:,:,:,0]).to(rank) 
    #####
    # Supervise 1 future second of pose
    #####
    output_pose_3d = output_pose_3d.to(rank) 
    norm_pose_3d = torch.norm(output_pose_3d[nan_mask] - gt_pose_3d_available[nan_mask], p=2, dim=-1)
    mean_B_traj = torch.mean(norm_traj)*1
    mean_B_pose_3d = torch.nan_to_num(torch.mean(norm_pose_3d)*1)
    return (mean_B_traj+mean_B_pose_3d)*100
