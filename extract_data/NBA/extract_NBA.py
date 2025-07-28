import numpy as np
import csv
import os
import argparse

current_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='train or val or test')
args = parser.parse_args()
split = args.split
scale_foot2meter = 0.3048 # convert foot to meter; to fairly compare with LED, we have a rescale in the line 114 of 'main/ft_NBA/evaluate.py'.

## extract partial data from train split as validation data
if args.split == 'val':
    split = 'train'
array_data = np.load(os.path.join(current_dir, 'raw','nba_'+split+'.npy'))


if args.split == 'train':
    array_data = array_data[:32500]
elif args.split == 'test':
    array_data = array_data[:12500] # extract the same samples as LED's dataloader_nba.py in https://drive.google.com/drive/folders/1NQffxbaEKeAa8pOjlmNqCdMzMmB68V04
elif args.split == 'val':
    array_data = array_data[32500:33500]
    val_size = array_data.shape[0]

else:
    raise ValueError('split should be train, val or test')

if not os.path.exists(current_dir+'/'+'output_csv'+'/'):
    os.makedirs(current_dir+'/'+'output_csv'+'/')

with open(os.path.join(current_dir,'output_csv','nba_'+args.split+'.csv'), 'w') as result:
    writer = csv.writer(result)

    num_batch = array_data.shape[0]

    for batch in range(num_batch):
        if batch % 100 == 0:
            print(batch, '/', num_batch)

        for frame in range(30):
            for N in range(11):
                traj = (batch*30+frame, batch*11+N, array_data[batch, frame, N, 0]*scale_foot2meter, array_data[batch, frame, N, 1]*scale_foot2meter)
                writer.writerow(traj)

