"""Create Trajnet data from original datasets."""
import argparse
import shutil
import os
import pysparkling
import scipy.io
from . import readers
from .scene import Scenes

import warnings
warnings.filterwarnings("ignore")

def UniHuMotion(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.UHM)
            .cache())


def write(input_rows, output_file, args):
    """ Write Valid Scenes without categorization """

    print(" Entering Writing ")
    if args.order_frames:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()),
                        key=lambda frame: frame % 100000)
    else:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))

    # split
    train_split_index = int(len(frames) * args.train_fraction)
    val_split_index = train_split_index + int(len(frames) * args.val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes(fps=args.fps, start_scene_id=0, args=args).rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(fps=args.fps, start_scene_id=train_scenes.scene_id, args=args).rows_to_file(val_rows, val_output)

    # test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(fps=args.fps, start_scene_id=train_scenes.scene_id, args=args).rows_to_file(test_rows, test_output)


def edit_goal_file(old_filename, new_filename):
    """ Rename goal files. 
    The name of goal files should be identical to the data files
    """

    shutil.copy("goal_files/train/" + old_filename, "goal_files/train/" + new_filename)
    shutil.copy("goal_files/val/" + old_filename, "goal_files/val/" + new_filename)
    shutil.copy("goal_files/test_private/" + old_filename, "goal_files/test_private/" + new_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=9,
                        help='Length of observation')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Length of prediction')
    parser.add_argument('--train_fraction', default=0.6, type=float,
                        help='Training set fraction')
    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='Validation set fraction')
    parser.add_argument('--fps', default=2.5, type=float,
                        help='fps')
    parser.add_argument('--order_frames', action='store_true',
                        help='For CFF')
    parser.add_argument('--chunk_stride', type=int, default=2,
                        help='Sampling Stride')
    parser.add_argument('--min_length', default= -0.1, type=float,
                        help='Min Length of Primary Trajectory')
    parser.add_argument('--all_present', action='store_true',
                        help='filter scenes where all pedestrians present at all times')
    parser.add_argument('--goal_file', default=None,
                        help='Pkl file for goals (required for ORCA sensitive scene filtering)')
    parser.add_argument('--mode', default='default', choices=('default', 'trajnet'),
                        help='mode of ORCA scene generation (required for ORCA sensitive scene filtering)')

    ## For Trajectory categorizing and filtering
    categorizers = parser.add_argument_group('categorizers')
    categorizers.add_argument('--static_threshold', type=float, default=1.0,
                              help='Type I static threshold')
    categorizers.add_argument('--linear_threshold', type=float, default=0.5,
                              help='Type II linear threshold (0.3 for Synthetic)')
    categorizers.add_argument('--inter_dist_thresh', type=float, default=5,
                              help='Type IIId distance threshold for cone')
    categorizers.add_argument('--inter_pos_range', type=float, default=15,
                              help='Type IIId angle threshold for cone (degrees)')
    categorizers.add_argument('--grp_dist_thresh', type=float, default=0.8,
                              help='Type IIIc distance threshold for group')
    categorizers.add_argument('--grp_std_thresh', type=float, default=0.2,
                              help='Type IIIc std deviation for group')
    categorizers.add_argument('--acceptance', nargs='+', type=float, default=[0.1, 1, 1, 1],
                              help='acceptance ratio of different trajectory (I, II, III, IV) types')

    args = parser.parse_args()
    sc = pysparkling.Context()

    if args.train_fraction == 1.0 and args.val_fraction == 0.0:
        ## NBA dataset
        temp_name = 'nba_train'
        ## train
        write(UniHuMotion(sc,'data/UniHuMotion_NBA/'+temp_name+'.csv'),
                'output_pre/{split}/'+temp_name+'.ndjson', args)
    elif args.train_fraction == 0.0 and args.val_fraction == 1.0:
        temp_name = 'nba_val'
        ## val
        write(UniHuMotion(sc,'data/UniHuMotion_NBA/'+temp_name+'.csv'),
                'output_pre/{split}/'+temp_name+'.ndjson', args)
    elif args.train_fraction == 0.0 and args.val_fraction == 0.0:
        temp_name = 'nba_test'
        ## test
        write(UniHuMotion(sc,'data/UniHuMotion_NBA/'+temp_name+'.csv'),
                'output_pre/{split}/'+temp_name+'.ndjson', args)
    else:
        raise ValueError("Invalid train and val fractions for UniHuMotion")
    

if __name__ == '__main__':
    main()