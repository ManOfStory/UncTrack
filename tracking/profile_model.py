import argparse
import torch
import os
import time
import importlib
from thop import profile
from thop.utils import clever_format
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("\n".join(sys.path))
from lib.models.unctrack.unctrack_online import Attention
def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='unctrack_online', choices=['unctrack','unctrack_online'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--display_name', type=str, default='UncTrack-B')
    parser.add_argument('--online_skip', type=int, default=200, help='the skip interval of unctrack_online')
    args = parser.parse_args()

    return args

def evaluate_stage1(model, template, search, skip=200, display_info='UncTrack'):
    """Compute FLOPs, Params, and Speed"""
    # compute flops and params except for score prediction
    # custom_ops = {Attention: get_complexity_Attention}

    macs, params = profile(model, inputs=(template, template, search, False), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('==>Macs is ', macs)
    print('==>Params is ', params)
    model.eval()
    # test speed
    T_w = 10
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        for i in range(T_w):
            _ = model(template, template, search, run_score_head=False)
        start = time.time()
        for i in range(T_t):
            if i % skip == 0:
                _ = model.set_online(template, template)
            _ = model.forward_test(search)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("\033[0;32;40m The average overall FPS of {} Stage1 is {}.\033[0m".format(display_info, 1.0 / avg_lat))

def evaluate(model, template, search, prototype, skip=200, display_info='UncTrack'):
    """Compute FLOPs, Params, and Speed"""
    # compute flops and params except for score prediction
    #custom_ops = {Attention: get_complexity_Attention}
    macs, params = profile(model, inputs=(template, template, search, True, None, {
            'prototype': prototype,
        }), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('==>Macs is ', macs)
    print('==>Params is ', params)
    model.eval()
    # test speed
    T_w = 10
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        if display_info == "UncTrack-B" or display_info == "UncTrack-L":
            # overall
            memory_prototype = model.gen_memory_prototype(template, template, search, torch.tensor([[0.4,0.5,0.6,0.7]]).to(template.device))
            memory_prototype = memory_prototype.repeat(1, 3, 1)
            for i in range(T_w):
                _ = model(template, template, search, run_score_head=True, memory_info = {'prototype':memory_prototype})
            start = time.time()
            for i in range(T_t):
                if i % skip == 0:
                    _ = model.set_online(template, template)
                _ = model.forward_test(search, run_score_head=True, memory_info = {'prototype':memory_prototype})
            end = time.time()
            avg_lat = (end - start) / T_t
            print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m" .format(display_info, 1.0/avg_lat))
        else:
            raise ValueError("display_info error")


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    print("cfg: {}".format(cfg))
    '''set some values'''
    bs = 1
    ms = 3
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    p_sz = cfg.MODEL.HIDDEN_DIM
    cfg.MODEL.BACKBONE.FREEZE_BN = False
    cfg.MODEL.HEAD_FREEZE_BN = False
    '''import stark network module'''
    model_module = importlib.import_module('lib.models.unctrack')
    if args.script == "unctrack":
        model_constructor = model_module.build_unctrack
        model = model_constructor(cfg, train=False)
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        # evaluate the model properties
        evaluate_stage1(model, template, search, args.online_skip, args.display_name)
    elif args.script == "unctrack_online":
        model_constructor = model_module.build_unctrack_online
        model = model_constructor(cfg, train=False)
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        prototype = torch.randn(bs,ms,p_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        prototype = prototype.to(device)
        # evaluate the model properties
        evaluate(model, template, search, prototype, args.online_skip, args.display_name)


