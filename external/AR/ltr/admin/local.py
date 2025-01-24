class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/guoyang/Projects/UncFormer'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/guoyang/Projects/UncFormer/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/home/guoyang/Projects/UncFormer/pretrained_networks'
        #self.lasot_dir = "/home/guoyang/Dataset/RawDataSet/LaSOT/LaSOTBenchmark"
        self.lasot_dir = "/mnt/ssd2/guoyang/Dataset/RawDataSet/LaSOT/LaSOTBenchmark"
        #self.got10k_dir = "/home/guoyang/Dataset/RawDataSet/GOT-10k/train"
        self.got10k_dir = "/mnt/ssd2/guoyang/Dataset/RawDataSet/GOT-10k/train"
        self.lasot_lmdb_dir = '/home/guoyang/Projects/UncFormer/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/guoyang/Projects/UncFormer/data/got10k_lmdb'
        self.trackingnet_dir = "/mnt/ssd2/guoyang/Dataset/RawDataSet/TrackingNet"
        self.trackingnet_lmdb_dir = '/home/guoyang/Projects/UncFormer/data/trackingnet_lmdb'
        self.coco_dir = "/mnt/ssd2/guoyang/Dataset/RawDataSet/COCO"
        self.coco_lmdb_dir = '/home/guoyang/Projects/UncFormer/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/guoyang/Projects/UncFormer/data/vid'
        self.imagenet_lmdb_dir = '/home/guoyang/Projects/UncFormer/data/vid_lmdb'
        
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
