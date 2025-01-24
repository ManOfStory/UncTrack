from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/lasot_lmdb'
    settings.lasot_path = "/mnt/ssd2/guoyang/Dataset/RawDataSet/LaSOT/LaSOTBenchmark"
    settings.nat_dir = ''
    settings.network_path = '/home/guoyang/Projects/Rebuttal/UncTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/nfs'
    settings.otb_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/OTB2015'
    settings.prj_dir = '/home/guoyang/Projects/Rebuttal/UncTrack'
    settings.result_plot_path = '/home/guoyang/Projects/Rebuttal/UncTrack/test/result_plots'
    settings.results_path = '/home/guoyang/Projects/Rebuttal/UncTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/guoyang/Projects/Rebuttal/UncTrack'
    settings.segmentation_path = '/home/guoyang/Projects/Rebuttal/UncTrack/test/segmentation_results'
    settings.tc128_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/trackingNet'
    settings.uav_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/UAV123'
    settings.vot_path = '/home/guoyang/Projects/Rebuttal/UncTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

