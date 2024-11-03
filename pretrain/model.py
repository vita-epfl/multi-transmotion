import torch
import torch.nn as nn
import numpy as np

class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        return output
     

class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_obs_len = 200, max_pred_len = 300, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model//2, max_norm=True).to(device)
        self.learned_encoding_pred = nn.Embedding(max_pred_len, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F, num_people=1) -> torch.Tensor:
        
        half = x.size(3)//2
        # Bi-directional encoding
        x[:,:in_F,:,0:half*2:2] = x[:,:in_F,:,0:half*2:2] + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,in_F:,:,0:half*2:2] = x[:,in_F:,:,0:half*2:2] + self.learned_encoding_pred(torch.arange(out_F).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,:,:,1:half*2:2] = x[:,:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(x.size(1), dim=0).to(self.device)).unsqueeze(0)

        return self.dropout(x)

class Learnedbb3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_obs_len = 200, max_pred_len = 300, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:
   
        x = x + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)


class Learnedbb2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_obs_len = 200, max_pred_len = 300, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:
      
        x = x + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)


class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000, max_obs_len = 8000, max_pred_len = 12000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)
        self.learned_encoding_pred = nn.Embedding(max_pred_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:

        ## Bi-directional encoding
        x[:,:in_F] = x[:,:in_F] + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,in_F:] = x[:,in_F:] + self.learned_encoding_pred(torch.arange(out_F).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedpose2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000, max_obs_len = 8000, max_pred_len = 12000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:

        x = x + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class TransMotion(nn.Module):
    def __init__(self, tok_dim=540, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0'):

        super(TransMotion, self).__init__()
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.device = device
        self.joints_pose = 39
        self.obs_and_pred = 30
        self.obs_and_pred_pose = 15

        self.max_fps = 50

        self.fc_in_traj = nn.Linear(2,nhid)
    
        self.predict_head_traj = []
        ## Multiple prediction heads for trajectory
        for i in range(20):
            self.predict_head_traj.append(nn.Linear(nhid, 2, bias=False))
        self.predict_head_traj = nn.ModuleList(self.predict_head_traj)

        self.fc_out_pose_3d = nn.Linear(nhid, 3)

        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, device=device) 

        self.fc_in_3dbb = nn.Linear(4,nhid)
        self.bb3d_encoder = Learnedbb3dEncoding(nhid, dropout, device=device)

        self.fc_in_2dbb = nn.Linear(4,nhid)
        self.bb2d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)

        self.fc_in_2dpose = nn.Linear(2, nhid)
        self.pose2d_encoder = Learnedpose2dEncoding(nhid, dropout, device=device)


        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)
        

    
    def forward(self, tgt, padding_mask, applymask=None):
        
        B, in_F, NJ, K = tgt.shape 

        all_F = self.obs_and_pred 
        J = self.token_num
        out_F = all_F - in_F
        N = NJ // J
        
        all_F_pose = self.obs_and_pred_pose
        out_F_pose = all_F_pose - in_F

        # real fps here
        fps = 5

        sampling_stride = int(self.max_fps/fps) 

        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)  
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,all_F,N,J,K)
    
        ## temporal mask
        mask_ratio_traj = 0.1
        mask_ratio_modality = 0.3
        ## spatial mask for dropped pose tokens
        mask_ratio_pose = torch.rand(1) 

        if not applymask:
            mask_ratio_traj = 0.0
            mask_ratio_modality = 0.0
            mask_ratio_pose = 0.0

        num_false = int(self.joints_pose * mask_ratio_pose)
        available_num_keypoints = self.joints_pose - num_false

        tgt_traj = tgt[:,:,:,0,:2].to(self.device)
        tgt_traj = torch.nan_to_num(tgt_traj)             

        traj_mask = torch.rand((B,all_F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        modality_selection_3dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).repeat(1,in_F,1,4)
        modality_selection_2dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).repeat(1,in_F,1,4)


        tgt_vis = tgt[:,:,:,1:]
        
        tgt_3dbb = tgt_vis[:,:in_F,:,0,:4].to(self.device)
        tgt_2dbb = tgt_vis[:,:in_F,:,1,:4].to(self.device)
        tgt_3dpose = tgt_vis[:,:,:,2:2+self.joints_pose,:3].to(self.device)
        tgt_2dpose = tgt_vis[:,:,:,2+self.joints_pose:,:2].to(self.device) 

        tgt_traj = self.fc_in_traj(tgt_traj)

        ## Up-sampling padding
        tgt_traj_temp = tgt_traj.repeat_interleave(sampling_stride, dim=1)
        tgt_traj = self.double_id_encoder(tgt_traj_temp, in_F*sampling_stride, out_F*sampling_stride, num_people=N).reshape(B,all_F,sampling_stride,N,self.nhid)[:,:,0] 

        tgt_3dbb = torch.nan_to_num(tgt_3dbb*modality_selection_3dbb)
        tgt_2dbb = torch.nan_to_num(tgt_2dbb*modality_selection_2dbb)
        tgt_3dpose = torch.nan_to_num(tgt_3dpose)
        tgt_2dpose = torch.nan_to_num(tgt_2dpose)

        tgt_3dbb = self.fc_in_3dbb(tgt_3dbb[:,:in_F])  
        tgt_3dbb_temp = tgt_3dbb.repeat_interleave(sampling_stride, dim=1)
        tgt_3dbb = self.bb3d_encoder(tgt_3dbb_temp, in_F*sampling_stride, out_F*sampling_stride).reshape(B,in_F,sampling_stride,N,self.nhid)[:,:,0]

        tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:in_F]) 
        tgt_2dbb_temp = tgt_2dbb.repeat_interleave(sampling_stride, dim=1)
        tgt_2dbb = self.bb2d_encoder(tgt_2dbb_temp, in_F*sampling_stride, out_F*sampling_stride).reshape(B,in_F,sampling_stride,N,self.nhid)[:,:,0]

        tgt_3dpose = tgt_3dpose[:,:all_F_pose].transpose(2,3).reshape(B,-1,N,3)
        tgt_3dpose = self.fc_in_3dpose(tgt_3dpose) 
        tgt_3dpose_temp = tgt_3dpose.repeat_interleave(sampling_stride, dim=1) 
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose_temp, in_F * self.joints_pose * sampling_stride, out_F_pose * self.joints_pose * sampling_stride).reshape(B,all_F_pose*(self.joints_pose),sampling_stride, N, self.nhid)[:,:,0]

        tgt_2dpose = tgt_2dpose[:,:in_F].transpose(2,3).reshape(B,-1,N,2) 
        tgt_2dpose = self.fc_in_2dpose(tgt_2dpose) 
        tgt_2dpose_temp = tgt_2dpose.repeat_interleave(sampling_stride, dim=1) 
        tgt_2dpose = self.pose2d_encoder(tgt_2dpose_temp, in_F * self.joints_pose * sampling_stride, out_F_pose * self.joints_pose * sampling_stride).reshape(B,in_F*(self.joints_pose),sampling_stride, N, self.nhid)[:,:,0] 

        ## drop spatial masked pose tokens

        joint_mask = torch.ones(self.joints_pose, dtype=torch.bool)
        false_indices = torch.randperm(self.joints_pose)[:num_false]
        joint_mask[false_indices] = False
        temp_mask_P_3d = joint_mask.unsqueeze(0).unsqueeze(0).repeat(B,all_F_pose,1).reshape(B,-1).to(self.device)  
        temp_mask_P_2d = joint_mask.unsqueeze(0).unsqueeze(0).repeat(B,in_F,1).reshape(B,-1).to(self.device) 

        tgt_3dpose = tgt_3dpose[temp_mask_P_3d].view(B,all_F_pose * available_num_keypoints,N,self.nhid)
        tgt_2dpose = tgt_2dpose[temp_mask_P_2d].view(B,in_F * available_num_keypoints,N,self.nhid)

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(all_F,-1,self.nhid) # [all_F, B*N, nhid]
        tgt_3dbb = torch.transpose(tgt_3dbb,0,1).reshape(in_F,-1,self.nhid) # [in_F, B*N, nhid]
        tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid) # [in_F, B*N, nhid]
        tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(all_F_pose * available_num_keypoints, -1, self.nhid) # [all_F*avail_J, B*N, nhid]
        tgt_2dpose = torch.transpose(tgt_2dpose, 0,1).reshape(in_F * available_num_keypoints, -1, self.nhid) # [in_F*avail_J, B*N, nhid]
        
        tgt = torch.cat((tgt_traj,tgt_3dbb,tgt_2dbb,tgt_3dpose,tgt_2dpose),0) 


        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(tgt.size(0),dim=1) 

        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)

        ##### local residual ######
        out_local = out_local * self.output_scale + tgt

        out_traj_pose3d = torch.cat((out_local[:all_F],out_local[(all_F+in_F*2):(all_F + in_F*2 + all_F_pose * available_num_keypoints)]),dim=0) # [all_F+all_F*available_num_keypoints, B*N, nhid]
        out_traj_pose3d = out_traj_pose3d.reshape((all_F+ all_F_pose * available_num_keypoints),B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid) # N*(all_F + all_F_pose * avail_J), B, nhid
        
        tgt_padding_mask_global = padding_mask.repeat_interleave(all_F + all_F_pose * available_num_keypoints, dim=1) 
        out_global = self.global_former(out_traj_pose3d, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        ##### global residual ######
        out_global = out_global * self.output_scale + out_traj_pose3d
        out_primary = out_global.reshape(N,all_F+ all_F_pose *available_num_keypoints,B,self.nhid)[0] # [all_F+all_F*available_num_keypoints, B, nhid]
        

        out_traj = []
        for k in range(20):
            temp_out = self.predict_head_traj[k](out_primary[:all_F]) 
            temp_out = temp_out.transpose(0,1)
            out_traj.append(temp_out)

        out_primary_pose_3d = self.fc_out_pose_3d(out_primary[all_F:]) #[(self.obs_and_pred_pose)*available_num_keypoints, B, 3]
        out_pose_3d = out_primary_pose_3d.transpose(0,1).reshape(B,all_F_pose,available_num_keypoints,3)

        return torch.stack(out_traj), out_pose_3d, joint_mask, available_num_keypoints

def create_model(config, logger, devices):
    token_num = config["MODEL"]["token_num"]
    n_hid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers_local=config["MODEL"]["num_layers_local"]
    nlayers_global=config["MODEL"]["num_layers_global"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]

    if config["MODEL"]["type"] == "transmotion":
        logger.info("Creating bert model.")
        model = TransMotion(
            nhid=n_hid,
            nhead=nhead,
            dim_feedfwd=dim_feedforward,
            nlayers_local=nlayers_local,
            nlayers_global=nlayers_global,
            output_scale=config["MODEL"]["output_scale"],
            obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
            num_tokens=token_num,
            device=devices
        ).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model
  