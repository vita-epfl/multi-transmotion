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
        

    
    def forward(self, tgt, padding_mask,metamask=None):
        
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
    
        ## add mask
        mask_ratio_traj = 0.0 # 0.1 for pre-training, 

        if not metamask:
            mask_ratio_traj = 0.0
        
        

        # ## Augment Traj by masking
        tgt_traj = tgt[:,:,:,0,:2].to(self.device)       
        traj_mask = torch.rand((B,all_F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        
        tgt_traj = self.fc_in_traj(tgt_traj)

        ## Up-sampling padding
        tgt_traj_temp = tgt_traj.repeat_interleave(sampling_stride, dim=1)
        tgt_traj = self.double_id_encoder(tgt_traj_temp, in_F*sampling_stride, out_F*sampling_stride, num_people=N).reshape(B,all_F,sampling_stride,N,self.nhid)[:,:,0] #[B,all_F,N,128]

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(all_F,-1,self.nhid) # [all_F, B*N, nhid]

        tgt = tgt_traj

        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(tgt.size(0),dim=1) 


        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)

        ##### local residual ######
        out_local = out_local * self.output_scale + tgt
        out_traj_pose3d = out_local[:all_F]
        out_traj_pose3d = out_traj_pose3d.reshape((all_F),B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)

        tgt_padding_mask_global = padding_mask.repeat_interleave(all_F, dim=1) 
        out_global = self.global_former(out_traj_pose3d, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        ##### global residual ######
        out_global = out_global * self.output_scale + out_traj_pose3d
        
        out_primary = out_global.reshape(N,all_F,B,self.nhid)[0] 


        out_traj = []
        for k in range(20):
            temp_out = self.predict_head_traj[k](out_primary[:all_F]) 
            temp_out = temp_out.transpose(0,1)
            out_traj.append(temp_out)

        out_pose_3d = None
        joint_mask = None
        available_num_keypoints = None
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
  
def create_model_eval(config, logger):
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
            device=config["DEVICE"]
        ).to(config["DEVICE"]).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model