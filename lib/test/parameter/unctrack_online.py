from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.unctrack_online.config import cfg, update_config_from_file


def parameters(yaml_name: str, model=None, search_area_scale=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/unctrack_online/{}.yaml'.format(yaml_name))
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    if search_area_scale is not None:
        params.search_factor = search_area_scale
    else:
        params.search_factor = cfg.TEST.SEARCH_FACTOR
    print("search_area_scale: {}".format(params.search_factor))
    params.search_size = cfg.TEST.SEARCH_SIZE

    if model is None:
        raise NotImplementedError("Please set proper model to test.")
    else:
        params.checkpoint = os.path.join(save_dir, "models/%s" % model)

    # whether to save boxes from all queries
    params.save_all_boxes = False
    params.debug = False
    params.search_factor = 4.5
    ###whether confident
    params.threshold = 0.8
    params.feat_sz = 18

    params.memory_size = 3
    params.topk = 3
    params.ppt = cfg.TEST.PPT #0.8 #prototype positive threshold

    return params
