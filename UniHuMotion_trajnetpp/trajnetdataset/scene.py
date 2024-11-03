""" Preparng Scenes for TrajNet """
import os
from collections import defaultdict

from .trajnetpp_data import SceneRow
from . import trajnetpp_writers



class Scenes(object):
    def __init__(self, fps, start_scene_id=0, args=None):
        self.scene_id = start_scene_id
        self.chunk_size = args.obs_len + args.pred_len
        self.chunk_stride = args.chunk_stride
        self.obs_len = args.obs_len
        self.visible_chunk = None
        self.frames = set()
        self.fps = fps
        self.min_length = args.min_length

    @staticmethod
    def euclidean_distance_2(row1, row2):
        """Euclidean distance squared between two rows."""
        return (row1.x - row2.x)**2 + (row1.y - row2.y)**2

    @staticmethod
    def close_pedestrians(rows, cell_size=10):
        """Fast computation of spatially close pedestrians.

        By frame, get the list of pedestrian ids that or close to other
        pedestrians. Approximate with multi-occupancy of discrete grid cells.
        """
        sparse_occupancy = defaultdict(list)
        for row in rows:
            x = int(row.x // cell_size * cell_size)
            y = int(row.y // cell_size * cell_size)
            sparse_occupancy[(x, y)].append(row.pedestrian)
        return {ped_id
                for cell in sparse_occupancy.values() if len(cell) > 1
                for ped_id in cell}

    @staticmethod
    def continuous_frames(frames, tolerance=1.5):
        increments = [f2 - f1 for f1, f2 in zip(frames[:-1], frames[1:])]
        median_increment = sorted(increments)[int(len(increments) / 2)]
        ok = median_increment * tolerance > max(increments)

        return ok

    def from_rows(self, rows):
        count_by_frame = rows.groupBy(lambda r: r.frame).mapValues(len).collectAsMap()
        occupancy_by_frame = (rows
                              .groupBy(lambda r: r.frame)
                              .mapValues(self.close_pedestrians)
                              .collectAsMap())

        def to_scene_row(ped_frames):
            ped_id, scene_frames = ped_frames
            row = SceneRow(self.scene_id, ped_id, scene_frames[0], scene_frames[-1], self.fps, 0)
            self.scene_id += 1
            return row

        # scenes: pedestrian of interest, [frames]
        scenes = (
            rows
            .groupBy(lambda r: r.pedestrian)
            .filter(lambda p_path: len(p_path[1]) >= self.chunk_size)
            .mapValues(lambda path: sorted(path, key=lambda p: p.frame))
            .flatMapValues(lambda path: [
                [path[ii].frame for ii in range(i, i + self.chunk_size)]
                for i in range(0, len(path) - self.chunk_size + 1, self.chunk_stride)
                # filter for pedestrians moving by more than min_length meter
                if self.euclidean_distance_2(path[i], path[i+self.chunk_size-1]) > self.min_length
            ])

            # filter out scenes with large gaps in frame numbers
            .filter(lambda ped_frames: self.continuous_frames(ped_frames[1]))

            .cache()
        )

        self.frames |= set(scenes
                           .flatMap(lambda ped_frames:
                                    ped_frames[1]
                                    if self.visible_chunk is None
                                    else ped_frames[1][:self.visible_chunk])
                           .toLocalIterator())

        return scenes.map(to_scene_row)


    def rows_to_file(self, rows, output_file):
        # if '/test/' in output_file:
            # print('Output File: ', output_file)
            # self.visible_chunk = self.obs_len
        # else:
            # self.visible_chunk = None
        self.visible_chunk = None
        scenes = self.from_rows(rows)
        tracks = rows.filter(lambda r: r.frame in self.frames)
        all_data = rows.context.union((scenes, tracks))

        ## removes the file, if previously generated
        if os.path.isfile(output_file):
            os.remove(output_file)

        ## write scenes and tracks
        all_data.map(trajnetpp_writers.trajnet).saveAsTextFile(output_file)

        return self
