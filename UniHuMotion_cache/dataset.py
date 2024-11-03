import torch
import numpy as np
import pickle
import os
from utils import Reader_UniHuMotion


class MultiPersonTrajPoseDataset(torch.utils.data.Dataset):



    def __init__(self, name, split="train", track_size=21, track_cutoff=9, segmented=True,
                 add_flips=False, frequency=1):
        self.name = name
        self.split = split
        self.track_size = track_size
        self.track_cutoff = track_cutoff
        self.frequency = frequency
        self.data_chunk_size = 4
        self.datalist = []
        self.dataset_idx = {}
        self.load_data()

    def load_data(self):
        self.datalist = []
        self.dataset_idx = {}
        split = self.split
        joint_and_mask_temp = []
      

        name = self.name
        data_name = name[4:] ## short name for the dataset
        if not os.path.exists('data/cache/'+data_name+'/'+split+'/'):
            os.makedirs('data/cache/'+data_name+'/'+split+'/')

        pkl_name = 'data/cache/'+data_name+'/'+split+'/'+data_name+'_idx_'
        train_scenes, _, _ = prepare_data('data/UniHuMotion/'+name+'/', subset=split, sample=1.0, goals=False, dataset_name=name)
        loaded_sample = 0
        idx = 0
        total_samples_this_file = len(train_scenes)

        for scene_i, (filename, scene_id, paths) in enumerate(train_scenes):
            scene_train = Reader_UniHuMotion.paths_to_xy(paths)
            scene_train = drop_ped_with_missing_frame(scene_train, obs=10)
            scene_train = drop_distant(scene_train, obs=10, max_num_peds=8)

            t,n = scene_train.shape[0],scene_train.shape[1]
            traj_4_col = np.pad(scene_train[:,:,:2].reshape(t,n,-1,2),((0, 0), (0, 0), (0, 0), (0, 2)), mode='constant')

            bb3d_4_col = scene_train[:,:,2:6].reshape(t,n,-1,4)
            bb2d_4_col = scene_train[:,:,6:10].reshape(t,n,-1,4)

            pose_3d_4_col = np.pad(np.transpose(scene_train[:,:,10:127].reshape(t,n,-1,39),(0,1,3,2)), ((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant')
            pose_2d_4_col = np.pad(np.transpose(scene_train[:,:,127:205].reshape(t,n,-1,39),(0,1,3,2)),((0, 0), (0, 0), (0, 0), (0, 2)), mode='constant')

            scene_train_real = np.concatenate((traj_4_col, bb3d_4_col, bb2d_4_col, pose_3d_4_col, pose_2d_4_col), axis=2)

            scene_train_real_ped = np.transpose(scene_train_real,(1,0,2,3)) #[N,F,81,4]

            scene_train_mask = np.ones(scene_train_real_ped.shape[:-1])

            joints = np.asarray(scene_train_real_ped)
            mask = np.asarray(scene_train_mask)


            joint_and_mask_temp.append((joints, mask))

            ## save pkl file based on the chunk size
            if loaded_sample%self.data_chunk_size == 0:
                save_data_to_pickle(joint_and_mask_temp, pkl_name+str(idx)+'.pkl')
                idx+=1
                joint_and_mask_temp = []
                
            ## Special case: total number of samples cannot be divided by 4, save last few samples to pkl directly
            if total_samples_this_file == (loaded_sample+1) and loaded_sample%4 != 0:
                save_data_to_pickle(joint_and_mask_temp, pkl_name+str(idx)+'.pkl')
                idx+=1
                joint_and_mask_temp = []


            self.datalist.append(('0'))
            loaded_sample+=1
            if (loaded_sample%1e3 == 0): print('finished loading ', int(loaded_sample/1e3), ' k samples')

        print('loaded ', loaded_sample, 'samples in total.')
    

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
     
        exit()
 



class UniHuMotionDataset(MultiPersonTrajPoseDataset):
    def __init__(self, dataset_name, split):
        super(UniHuMotionDataset, self).__init__(dataset_name, split=split, frequency=1)




def create_dataset(dataset_name, split="train"):
    if dataset_name[:3] == 'UHM':
        dataset = UniHuMotionDataset(dataset_name, split=split)
    else:
        raise ValueError(f"Dataset with name '{dataset_name}' not found.")
        
    return dataset


def get_datasets(datasets_list):

    datasets = []
    for dataset_name in datasets_list:
        datasets.append(create_dataset(dataset_name, split="train"))
    return datasets

def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as f: 
        pickle.dump(data, f)

def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def drop_ped_with_missing_frame(xy, obs):
    xy_n_t = np.transpose(xy, (1, 0, 2)) 
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(xy_n_t.shape[0]-1):
        for t in range(obs):
            if np.isnan(xy_n_t[n+1, t, 0]) == True:
                mask[n+1] = False
                break
    return np.transpose(xy_n_t[mask], (1, 0, 2))

def drop_distant(xy, obs, max_num_peds=8):
    """
    Only Keep the max_num_peds closest pedestrians
    """
    distance_2 = np.sum(np.square(xy[:obs,:,0:2] - xy[:obs,0:1,0:2]), axis=2)
    smallest_dist_to_ego = np.nanmin(distance_2, axis=0)
    return xy[:, np.argsort(smallest_dist_to_ego)[:(max_num_peds)]]

def prepare_data(path, subset='/train/', sample=1.0, goals=True, dataset_name=''):

    all_scenes = []
    files = [f.replace('.ndjson', '') for f in os.listdir(path + subset) if f.endswith('.ndjson')]

    for file in files:
            reader = Reader_UniHuMotion(path + subset + '/' + file + '.ndjson', scene_type='paths')
            scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
            all_scenes += scene
    return all_scenes, None, True