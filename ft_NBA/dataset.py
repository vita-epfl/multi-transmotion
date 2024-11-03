import torch
import pickle
import os
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    joints_list = []
    masks_list = []
    num_people_list = []

    real_batch = []
    for mini_batch in batch:
        real_batch += mini_batch


    for joints, masks in real_batch:
        joints_list.append(torch.from_numpy(joints[:,:,:])) 
        masks_list.append(torch.from_numpy(masks[:,:,:]))
        num_people_list.append(torch.zeros(joints.shape[0]))

    joints = pad_sequence(joints_list, batch_first=True)
    masks = pad_sequence(masks_list, batch_first=True)
    padding_mask = pad_sequence(num_people_list, batch_first=True, padding_value=1).bool()
    return joints, masks, padding_mask


def batch_process_coords(joints, masks, padding_mask, config,training=False, multiperson=True):
    
    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]   
    joints[:,:,:,0] = joints[:,:,:,0] - joints[:,0:1, (in_F-1):in_F, 0]

    joints[:,:,:,1:] = joints[:,:,:,1:] - joints[:,:,(in_F-1):in_F,1:]

    B, N, F, J, K = joints.shape 

    joints = joints.transpose(1, 2).reshape(B, F, N*J, K)
    masks = masks.transpose(1, 2).reshape(B, F, N*J)
    in_joints = joints[:,:in_F].float()
    out_joints = joints[:,in_F:in_F+out_F].float()
    in_masks = masks[:,:in_F].float()
    out_masks = masks[:,in_F:in_F+out_F].float()

    return in_joints, in_masks, out_joints, out_masks, padding_mask.float()

def batch_process_coords_eval(joints, masks, padding_mask, config,training=False, multiperson=True):
    in_F, out_F = config["TRAIN"]["input_track_size"], config["TRAIN"]["output_track_size"]  
    in_joints_pelvis = joints[:,:, (in_F-1):in_F, 0:1, :].clone()

    joints[:,:,:,0] = joints[:,:,:,0] - joints[:,0:1, (in_F-1):in_F, 0]

    joints[:,:,:,1:] = joints[:,:,:,1:] - joints[:,:,(in_F-1):in_F,1:]

    B, N, F, J, K = joints.shape 

    joints = joints.transpose(1, 2).reshape(B, F, N*J, K)
    in_joints_pelvis = in_joints_pelvis.reshape(B, 1, N, K)
    masks = masks.transpose(1, 2).reshape(B, F, N*J)
    in_joints = joints[:,:in_F].float()
    out_joints = joints[:,in_F:in_F+out_F].float()
    in_masks = masks[:,:in_F].float()
    out_masks = masks[:,in_F:in_F+out_F].float()

    return in_joints, in_masks, out_joints, out_masks, padding_mask.float(), in_joints_pelvis[:,0,0,:2].float()


class MultiPersonTrajPoseDataset(torch.utils.data.Dataset):

    def __init__(self, name, split="train", track_size=21, track_cutoff=9, segmented=True,
                 add_flips=False, frequency=1):
        self.name = [item.strip() for item in name.split(',')]
        self.split = split
        self.track_size = track_size
        self.track_cutoff = track_cutoff
        self.frequency = frequency
        self.data_chunk_size = 4
        self.datalist = []
        self.dataset_idx = {}
        self.load_data()
        
    def load_data(self):

        split = self.split
        data_names = self.name

        for data_name in data_names:
            directory = 'data/cache/'+data_name+'/'+split+'/'
            for filename in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, filename)):
                    self.datalist.append(directory+filename)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        with open(self.datalist[idx], 'rb') as f:
            read = pickle.load(f)
            return read


class UniHuMotionDataset(MultiPersonTrajPoseDataset):
    def __init__(self, dataset_name, **args):
        super(UniHuMotionDataset, self).__init__(dataset_name, frequency=1, **args)

def create_dataset(dataset_name, logger, **args):
    logger.info("Loading dataset " + dataset_name)

    dataset = UniHuMotionDataset(dataset_name, **args)
    return dataset

def get_datasets(datasets_list, config, logger):

    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    datasets = []
    for dataset_name in datasets_list:
        datasets.append(create_dataset(dataset_name, logger, split="train", track_size=(in_F+out_F), track_cutoff=in_F))
    return datasets

def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as f: # 'wb' cover content and create a new one, 'ab' write one by one
        pickle.dump(data, f)

def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


