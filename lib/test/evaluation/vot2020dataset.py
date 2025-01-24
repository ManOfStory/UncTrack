import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VOT2020Dataset(BaseDataset):
    """ VOT2020 dataset.
        Write by Guo Yang
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot2020_path
        self.data_path = os.path.join(self.base_path,'data')
        self.anno_path = os.path.join(self.base_path,'annotations')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def xyxy2xywh(self, box: list):
        xmin, ymin, xmax, ymax = box
        return [xmin, ymin, xmax-xmin, ymax-ymin]

    def read_vot_ground_truth(self, anno_path):
        from vot.region.io import parse_region
        with open(anno_path, 'r') as f:
            contents = f.readlines()
        gt_bboxes = []
        for c in contents:
            mask = parse_region(c)
            gt_bboxes.append(self.xyxy2xywh(mask.bounds()))
        return np.array(gt_bboxes)  #x1 y1 x2 y2

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.anno_path, sequence_name)

        ground_truth_rect = self.read_vot_ground_truth(anno_path)

        frames_path = '{}/{}'.format(self.data_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'vot2020', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        return sequence_list


