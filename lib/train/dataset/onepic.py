import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAVDataset(BaseDataset):
    """ UAV123 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "uav_bike1", "path": "data_seq/UAV123/bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bike1.txt", "object_class": "vehicle"},
            {"name": "uav_bike2", "path": "data_seq/UAV123/bike2", "startFrame": 1, "endFrame": 553, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bike2.txt", "object_class": "vehicle"},
            {"name": "uav_bike3", "path": "data_seq/UAV123/bike3", "startFrame": 1, "endFrame": 433, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bike3.txt", "object_class": "vehicle"},
            {"name": "uav_bird1_1", "path": "data_seq/UAV123/bird1", "startFrame": 1, "endFrame": 253, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bird1_1.txt", "object_class": "bird"},
            {"name": "uav_bird1_2", "path": "data_seq/UAV123/bird1", "startFrame": 775, "endFrame": 1477, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bird1_2.txt", "object_class": "bird"},
            {"name": "uav_bird1_3", "path": "data_seq/UAV123/bird1", "startFrame": 1573, "endFrame": 2437, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/bird1_3.txt", "object_class": "bird"},
            {"name": "uav_boat1", "path": "data_seq/UAV123/boat1", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat1.txt", "object_class": "vessel"},
            {"name": "uav_boat2", "path": "data_seq/UAV123/boat2", "startFrame": 1, "endFrame": 799, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat2.txt", "object_class": "vessel"},
            {"name": "uav_boat3", "path": "data_seq/UAV123/boat3", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat3.txt", "object_class": "vessel"},
            {"name": "uav_boat4", "path": "data_seq/UAV123/boat4", "startFrame": 1, "endFrame": 553, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat4.txt", "object_class": "vessel"},
            {"name": "uav_boat5", "path": "data_seq/UAV123/boat5", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat5.txt", "object_class": "vessel"},
            {"name": "uav_boat6", "path": "data_seq/UAV123/boat6", "startFrame": 1, "endFrame": 805, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat6.txt", "object_class": "vessel"},
            {"name": "uav_boat7", "path": "data_seq/UAV123/boat7", "startFrame": 1, "endFrame": 535, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat7.txt", "object_class": "vessel"},
            {"name": "uav_boat8", "path": "data_seq/UAV123/boat8", "startFrame": 1, "endFrame": 685, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat8.txt", "object_class": "vessel"},
            {"name": "uav_boat9", "path": "data_seq/UAV123/boat9", "startFrame": 1, "endFrame": 1399, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/boat9.txt", "object_class": "vessel"},
            {"name": "uav_building1", "path": "data_seq/UAV123/building1", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building1.txt", "object_class": "other"},
            {"name": "uav_building2", "path": "data_seq/UAV123/building2", "startFrame": 1, "endFrame": 577, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building2.txt", "object_class": "other"},
            {"name": "uav_building3", "path": "data_seq/UAV123/building3", "startFrame": 1, "endFrame": 829, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building3.txt", "object_class": "other"},
            {"name": "uav_building4", "path": "data_seq/UAV123/building4", "startFrame": 1, "endFrame": 787, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building4.txt", "object_class": "other"},
            {"name": "uav_building5", "path": "data_seq/UAV123/building5", "startFrame": 1, "endFrame": 481, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/building5.txt", "object_class": "other"},
            {"name": "uav_car1_1", "path": "data_seq/UAV123/car1", "startFrame": 1, "endFrame": 751, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_1.txt", "object_class": "car"},
            {"name": "uav_car1_2", "path": "data_seq/UAV123/car1", "startFrame": 751, "endFrame": 1627, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_2.txt", "object_class": "car"},
            {"name": "uav_car1_3", "path": "data_seq/UAV123/car1", "startFrame": 1627, "endFrame": 2629, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_3.txt", "object_class": "car"},
            {"name": "uav_car10", "path": "data_seq/UAV123/car10", "startFrame": 1, "endFrame": 1405, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car10.txt", "object_class": "car"},
            {"name": "uav_car11", "path": "data_seq/UAV123/car11", "startFrame": 1, "endFrame": 337, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car11.txt", "object_class": "car"},
            {"name": "uav_car12", "path": "data_seq/UAV123/car12", "startFrame": 1, "endFrame": 499, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car12.txt", "object_class": "car"},
            {"name": "uav_car13", "path": "data_seq/UAV123/car13", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car13.txt", "object_class": "car"},
            {"name": "uav_car14", "path": "data_seq/UAV123/car14", "startFrame": 1, "endFrame": 1327, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car14.txt", "object_class": "car"},
            {"name": "uav_car15", "path": "data_seq/UAV123/car15", "startFrame": 1, "endFrame": 469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car15.txt", "object_class": "car"},
            {"name": "uav_car16_1", "path": "data_seq/UAV123/car16", "startFrame": 1, "endFrame": 415, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car16_1.txt", "object_class": "car"},
            {"name": "uav_car16_2", "path": "data_seq/UAV123/car16", "startFrame": 415, "endFrame": 1993, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car16_2.txt", "object_class": "car"},
            {"name": "uav_car17", "path": "data_seq/UAV123/car17", "startFrame": 1, "endFrame": 1057, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car17.txt", "object_class": "car"},
            {"name": "uav_car18", "path": "data_seq/UAV123/car18", "startFrame": 1, "endFrame": 1207, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car18.txt", "object_class": "car"},
            {"name": "uav_car1_s", "path": "data_seq/UAV123/car1_s", "startFrame": 1, "endFrame": 1475, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car1_s.txt", "object_class": "car"},
            {"name": "uav_car2", "path": "data_seq/UAV123/car2", "startFrame": 1, "endFrame": 1321, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car2.txt", "object_class": "car"},
            {"name": "uav_car2_s", "path": "data_seq/UAV123/car2_s", "startFrame": 1, "endFrame": 320, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car2_s.txt", "object_class": "car"},
            {"name": "uav_car3", "path": "data_seq/UAV123/car3", "startFrame": 1, "endFrame": 1717, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car3.txt", "object_class": "car"},
            {"name": "uav_car3_s", "path": "data_seq/UAV123/car3_s", "startFrame": 1, "endFrame": 1300, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car3_s.txt", "object_class": "car"},
            {"name": "uav_car4", "path": "data_seq/UAV123/car4", "startFrame": 1, "endFrame": 1345, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car4.txt", "object_class": "car"},
            {"name": "uav_car4_s", "path": "data_seq/UAV123/car4_s", "startFrame": 1, "endFrame": 830, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car4_s.txt", "object_class": "car"},
            {"name": "uav_car5", "path": "data_seq/UAV123/car5", "startFrame": 1, "endFrame": 745, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car5.txt", "object_class": "car"},
            {"name": "uav_car6_1", "path": "data_seq/UAV123/car6", "startFrame": 1, "endFrame": 487, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_1.txt", "object_class": "car"},
            {"name": "uav_car6_2", "path": "data_seq/UAV123/car6", "startFrame": 487, "endFrame": 1807, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_2.txt", "object_class": "car"},
            {"name": "uav_car6_3", "path": "data_seq/UAV123/car6", "startFrame": 1807, "endFrame": 2953, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_3.txt", "object_class": "car"},
            {"name": "uav_car6_4", "path": "data_seq/UAV123/car6", "startFrame": 2953, "endFrame": 3925, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_4.txt", "object_class": "car"},
            {"name": "uav_car6_5", "path": "data_seq/UAV123/car6", "startFrame": 3925, "endFrame": 4861, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car6_5.txt", "object_class": "car"},
            {"name": "uav_car7", "path": "data_seq/UAV123/car7", "startFrame": 1, "endFrame": 1033, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car7.txt", "object_class": "car"},
            {"name": "uav_car8_1", "path": "data_seq/UAV123/car8", "startFrame": 1, "endFrame": 1357, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car8_1.txt", "object_class": "car"},
            {"name": "uav_car8_2", "path": "data_seq/UAV123/car8", "startFrame": 1357, "endFrame": 2575, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car8_2.txt", "object_class": "car"},
            {"name": "uav_car9", "path": "data_seq/UAV123/car9", "startFrame": 1, "endFrame": 1879, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/car9.txt", "object_class": "car"},
            {"name": "uav_group1_1", "path": "data_seq/UAV123/group1", "startFrame": 1, "endFrame": 1333, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_1.txt", "object_class": "person"},
            {"name": "uav_group1_2", "path": "data_seq/UAV123/group1", "startFrame": 1333, "endFrame": 2515, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_2.txt", "object_class": "person"},
            {"name": "uav_group1_3", "path": "data_seq/UAV123/group1", "startFrame": 2515, "endFrame": 3925, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_3.txt", "object_class": "person"},
            {"name": "uav_group1_4", "path": "data_seq/UAV123/group1", "startFrame": 3925, "endFrame": 4873, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group1_4.txt", "object_class": "person"},
            {"name": "uav_group2_1", "path": "data_seq/UAV123/group2", "startFrame": 1, "endFrame": 907, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group2_1.txt", "object_class": "person"},
            {"name": "uav_group2_2", "path": "data_seq/UAV123/group2", "startFrame": 907, "endFrame": 1771, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group2_2.txt", "object_class": "person"},
            {"name": "uav_group2_3", "path": "data_seq/UAV123/group2", "startFrame": 1771, "endFrame": 2683, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group2_3.txt", "object_class": "person"},
            {"name": "uav_group3_1", "path": "data_seq/UAV123/group3", "startFrame": 1, "endFrame": 1567, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_1.txt", "object_class": "person"},
            {"name": "uav_group3_2", "path": "data_seq/UAV123/group3", "startFrame": 1567, "endFrame": 2827, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_2.txt", "object_class": "person"},
            {"name": "uav_group3_3", "path": "data_seq/UAV123/group3", "startFrame": 2827, "endFrame": 4369, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_3.txt", "object_class": "person"},
            {"name": "uav_group3_4", "path": "data_seq/UAV123/group3", "startFrame": 4369, "endFrame": 5527, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/group3_4.txt", "object_class": "person"},
            {"name": "uav_person1", "path": "data_seq/UAV123/person1", "startFrame": 1, "endFrame": 799, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person1.txt", "object_class": "person"},
            {"name": "uav_person10", "path": "data_seq/UAV123/person10", "startFrame": 1, "endFrame": 1021, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person10.txt", "object_class": "person"},
            {"name": "uav_person11", "path": "data_seq/UAV123/person11", "startFrame": 1, "endFrame": 721, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person11.txt", "object_class": "person"},
            {"name": "uav_person12_1", "path": "data_seq/UAV123/person12", "startFrame": 1, "endFrame": 601, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person12_1.txt", "object_class": "person"},
            {"name": "uav_person12_2", "path": "data_seq/UAV123/person12", "startFrame": 601, "endFrame": 1621, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person12_2.txt", "object_class": "person"},
            {"name": "uav_person13", "path": "data_seq/UAV123/person13", "startFrame": 1, "endFrame": 883, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person13.txt", "object_class": "person"},
            {"name": "uav_person14_1", "path": "data_seq/UAV123/person14", "startFrame": 1, "endFrame": 847, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person14_1.txt", "object_class": "person"},
            {"name": "uav_person14_2", "path": "data_seq/UAV123/person14", "startFrame": 847, "endFrame": 1813, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person14_2.txt", "object_class": "person"},
            {"name": "uav_person14_3", "path": "data_seq/UAV123/person14", "startFrame": 1813, "endFrame": 2923,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person14_3.txt", "object_class": "person"},
            {"name": "uav_person15", "path": "data_seq/UAV123/person15", "startFrame": 1, "endFrame": 1339, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person15.txt", "object_class": "person"},
            {"name": "uav_person16", "path": "data_seq/UAV123/person16", "startFrame": 1, "endFrame": 1147, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person16.txt", "object_class": "person"},
            {"name": "uav_person17_1", "path": "data_seq/UAV123/person17", "startFrame": 1, "endFrame": 1501, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person17_1.txt", "object_class": "person"},
            {"name": "uav_person17_2", "path": "data_seq/UAV123/person17", "startFrame": 1501, "endFrame": 2347,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person17_2.txt", "object_class": "person"},
            {"name": "uav_person18", "path": "data_seq/UAV123/person18", "startFrame": 1, "endFrame": 1393, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person18.txt", "object_class": "person"},
            {"name": "uav_person19_1", "path": "data_seq/UAV123/person19", "startFrame": 1, "endFrame": 1243, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person19_1.txt", "object_class": "person"},
            {"name": "uav_person19_2", "path": "data_seq/UAV123/person19", "startFrame": 1243, "endFrame": 2791,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_2.txt", "object_class": "person"},
            {"name": "uav_person19_3", "path": "data_seq/UAV123/person19", "startFrame": 2791, "endFrame": 4357,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/person19_3.txt", "object_class": "person"},
            {"name": "uav_person1_s", "path": "data_seq/UAV123/person1_s", "startFrame": 1, "endFrame": 1600, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person1_s.txt", "object_class": "person"},
            {"name": "uav_person2_1", "path": "data_seq/UAV123/person2", "startFrame": 1, "endFrame": 1189, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person2_1.txt", "object_class": "person"},
            {"name": "uav_person2_2", "path": "data_seq/UAV123/person2", "startFrame": 1189, "endFrame": 2623, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person2_2.txt", "object_class": "person"},
            {"name": "uav_person20", "path": "data_seq/UAV123/person20", "startFrame": 1, "endFrame": 1783, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person20.txt", "object_class": "person"},
            {"name": "uav_person21", "path": "data_seq/UAV123/person21", "startFrame": 1, "endFrame": 487, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person21.txt", "object_class": "person"},
            {"name": "uav_person22", "path": "data_seq/UAV123/person22", "startFrame": 1, "endFrame": 199, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person22.txt", "object_class": "person"},
            {"name": "uav_person23", "path": "data_seq/UAV123/person23", "startFrame": 1, "endFrame": 397, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person23.txt", "object_class": "person"},
            {"name": "uav_person2_s", "path": "data_seq/UAV123/person2_s", "startFrame": 1, "endFrame": 250, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person2_s.txt", "object_class": "person"},
            {"name": "uav_person3", "path": "data_seq/UAV123/person3", "startFrame": 1, "endFrame": 643, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person3.txt", "object_class": "person"},
            {"name": "uav_person3_s", "path": "data_seq/UAV123/person3_s", "startFrame": 1, "endFrame": 505, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person3_s.txt", "object_class": "person"},
            {"name": "uav_person4_1", "path": "data_seq/UAV123/person4", "startFrame": 1, "endFrame": 1501, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person4_1.txt", "object_class": "person"},
            {"name": "uav_person4_2", "path": "data_seq/UAV123/person4", "startFrame": 1501, "endFrame": 2743, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person4_2.txt", "object_class": "person"},
            {"name": "uav_person5_1", "path": "data_seq/UAV123/person5", "startFrame": 1, "endFrame": 877, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person5_1.txt", "object_class": "person"},
            {"name": "uav_person5_2", "path": "data_seq/UAV123/person5", "startFrame": 877, "endFrame": 2101, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person5_2.txt", "object_class": "person"},
            {"name": "uav_person6", "path": "data_seq/UAV123/person6", "startFrame": 1, "endFrame": 901, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person6.txt", "object_class": "person"},
            {"name": "uav_person7_1", "path": "data_seq/UAV123/person7", "startFrame": 1, "endFrame": 1249, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person7_1.txt", "object_class": "person"},
            {"name": "uav_person7_2", "path": "data_seq/UAV123/person7", "startFrame": 1249, "endFrame": 2065, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person7_2.txt", "object_class": "person"},
            {"name": "uav_person8_1", "path": "data_seq/UAV123/person8", "startFrame": 1, "endFrame": 1075, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person8_1.txt", "object_class": "person"},
            {"name": "uav_person8_2", "path": "data_seq/UAV123/person8", "startFrame": 1075, "endFrame": 1525, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person8_2.txt", "object_class": "person"},
            {"name": "uav_person9", "path": "data_seq/UAV123/person9", "startFrame": 1, "endFrame": 661, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/person9.txt", "object_class": "person"},
            {"name": "uav_truck1", "path": "data_seq/UAV123/truck1", "startFrame": 1, "endFrame": 463, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck1.txt", "object_class": "truck"},
            {"name": "uav_truck2", "path": "data_seq/UAV123/truck2", "startFrame": 1, "endFrame": 385, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck2.txt", "object_class": "truck"},
            {"name": "uav_truck3", "path": "data_seq/UAV123/truck3", "startFrame": 1, "endFrame": 535, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck3.txt", "object_class": "truck"},
            {"name": "uav_truck4_1", "path": "data_seq/UAV123/truck4", "startFrame": 1, "endFrame": 577, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck4_1.txt", "object_class": "truck"},
            {"name": "uav_truck4_2", "path": "data_seq/UAV123/truck4", "startFrame": 577, "endFrame": 1261, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/truck4_2.txt", "object_class": "truck"},
            {"name": "uav_uav1_1", "path": "data_seq/UAV123/uav1", "startFrame": 1, "endFrame": 1555, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav1_1.txt", "object_class": "aircraft"},
            {"name": "uav_uav1_2", "path": "data_seq/UAV123/uav1", "startFrame": 1555, "endFrame": 2377, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav1_2.txt", "object_class": "aircraft"},
            {"name": "uav_uav1_3", "path": "data_seq/UAV123/uav1", "startFrame": 2473, "endFrame": 3469, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav1_3.txt", "object_class": "aircraft"},
            {"name": "uav_uav2", "path": "data_seq/UAV123/uav2", "startFrame": 1, "endFrame": 133, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav2.txt", "object_class": "aircraft"},
            {"name": "uav_uav3", "path": "data_seq/UAV123/uav3", "startFrame": 1, "endFrame": 265, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav3.txt", "object_class": "aircraft"},
            {"name": "uav_uav4", "path": "data_seq/UAV123/uav4", "startFrame": 1, "endFrame": 157, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav4.txt", "object_class": "aircraft"},
            {"name": "uav_uav5", "path": "data_seq/UAV123/uav5", "startFrame": 1, "endFrame": 139, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav5.txt", "object_class": "aircraft"},
            {"name": "uav_uav6", "path": "data_seq/UAV123/uav6", "startFrame": 1, "endFrame": 109, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav6.txt", "object_class": "aircraft"},
            {"name": "uav_uav7", "path": "data_seq/UAV123/uav7", "startFrame": 1, "endFrame": 373, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav7.txt", "object_class": "aircraft"},
            {"name": "uav_uav8", "path": "data_seq/UAV123/uav8", "startFrame": 1, "endFrame": 301, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/uav8.txt", "object_class": "aircraft"},
            {"name": "uav_wakeboard1", "path": "data_seq/UAV123/wakeboard1", "startFrame": 1, "endFrame": 421, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard1.txt", "object_class": "person"},
            {"name": "uav_wakeboard10", "path": "data_seq/UAV123/wakeboard10", "startFrame": 1, "endFrame": 469,
             "nz": 6, "ext": "jpg", "anno_path": "anno/UAV123/wakeboard10.txt", "object_class": "person"},
            {"name": "uav_wakeboard2", "path": "data_seq/UAV123/wakeboard2", "startFrame": 1, "endFrame": 733, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard2.txt", "object_class": "person"},
            {"name": "uav_wakeboard3", "path": "data_seq/UAV123/wakeboard3", "startFrame": 1, "endFrame": 823, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard3.txt", "object_class": "person"},
            {"name": "uav_wakeboard4", "path": "data_seq/UAV123/wakeboard4", "startFrame": 1, "endFrame": 697, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard4.txt", "object_class": "person"},
            {"name": "uav_wakeboard5", "path": "data_seq/UAV123/wakeboard5", "startFrame": 1, "endFrame": 1675, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard5.txt", "object_class": "person"},
            {"name": "uav_wakeboard6", "path": "data_seq/UAV123/wakeboard6", "startFrame": 1, "endFrame": 1165, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard6.txt", "object_class": "person"},
            {"name": "uav_wakeboard7", "path": "data_seq/UAV123/wakeboard7", "startFrame": 1, "endFrame": 199, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard7.txt", "object_class": "person"},
            {"name": "uav_wakeboard8", "path": "data_seq/UAV123/wakeboard8", "startFrame": 1, "endFrame": 1543, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard8.txt", "object_class": "person"},
            {"name": "uav_wakeboard9", "path": "data_seq/UAV123/wakeboard9", "startFrame": 1, "endFrame": 355, "nz": 6,
             "ext": "jpg", "anno_path": "anno/UAV123/wakeboard9.txt", "object_class": "person"}
        ]

        return sequence_info_list



import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class OnePic(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().onepic_dir if root is None else root
        super().__init__('OnePic', root, image_loader)


        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'onepic'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:06}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        path = self._get_frame_path(seq_path, frame_id)
        return self.image_loader(path)

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames
