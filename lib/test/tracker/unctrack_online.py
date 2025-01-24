import numpy as np
import numpy
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.unctrack import build_unctrack_online
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from lib.test.tracker.tracker_utils import vis_attn_maps
from lib.utils.box_ops import box_xyxy_to_cxcywh,box_cxcywh_to_xyxy,box_xywh_to_xyxy
from scipy.interpolate import griddata
from scipy.ndimage import zoom
class UncTrackOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(UncTrackOnline, self).__init__(params)
        network = build_unctrack_online(params.cfg, train=False)
        miss,unexpected = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = params.debug
        self.frame_id = 0
        self.memory_size = params.memory_size
        self.topk = params.topk
        self.ppt = params.ppt #prototype positive threshold
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        self.online_size = self.online_sizes[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0

        if hasattr(params, 'memory_update_interval'):
            self.memory_update_interval = params.memory_update_interval
        else:
            self.memory_update_interval = 1



        print("Search scale is: ", self.params.search_factor)
        print("Online size is: ", self.online_size)
        print("Update interval is: ", self.update_interval)
        print("Max score decay is ", self.max_score_decay)
        print("Memory update interval is : ", self.memory_update_interval)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        if self.params.vis_attn == 1:
            self.z_patch = z_patch_arr
            self.oz_patch = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        self.online_max_template = template
        self.base_search_factor = self.params.search_factor
        self.search_factor = self.base_search_factor
        print('search_factor:',self.base_search_factor)
        print('threshold:', self.params.threshold)
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.state = info['init_bbox']
        self.oz_patch_max = z_patch_arr
        self.max_pred_score = -1
        self.maxscore = 0
        self.online_min_template = template
        self.online_forget_id = 0
        # save states
        self.state = info['init_bbox']

        self.frame_id = 0

        ###forward memory
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        gt_box = np.array(info['init_bbox'])
        gt_box = box_cxcywh_to_xyxy(torch.from_numpy(gt_box)).detach().cpu().numpy()
        # if not np.isnan(gt_box).any():
        #     # x y w h --> cx cy w h
        #     gt_box = np.array(self.map_box(gt_box, resize_factor))
        #     gt_box = gt_box * resize_factor
        #     gt_box = box_cxcywh_to_xyxy(torch.from_numpy(gt_box)).detach().cpu().numpy()
        #     gt_box = np.clip(gt_box, a_min=0, a_max=x_patch_arr.shape[0])
        #     gt_box = gt_box / x_patch_arr.shape[0]
        # else:
        #     raise ValueError("init_bbox nan")
        gt_box = torch.tensor(gt_box).to(self.template.device).unsqueeze(0)
        memory_prototype = self.network.gen_memory_prototype(template, template, search, gt_box)
        self.nextprototype = memory_prototype.clone()
        self.nextscore = self.ppt
        self.memory_prototype = memory_prototype.repeat(1,self.memory_size,1)

        self.memory_score = torch.tensor([self.ppt] * self.memory_size)

        self.update = False

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

        ###initialize kalman filter
        self.init_kalman(gt_box)

    def init_kalman(self, init_box):
        initial_box_state = box_xyxy_to_cxcywh(init_box).cpu().detach().numpy()[0]
        #[cx, cy, w, h, dx, dy, dw, dh]
        initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3], 0, 0, 0, 0]]).T
        self.A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])
        self.A_ = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1]])
        # 状态观测矩阵
        self.H = np.eye(8)

        # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
        # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
        self.Q = np.eye(8) * 0.1
        # 观测噪声协方差矩阵R，p(v)~N(0,R)
        # 观测噪声来自于检测框丢失、重叠等
        self.R = np.eye(8) * 10
        # 控制输入矩阵B
        self.B = None
        # 状态估计协方差矩阵P初始化
        self.P = np.eye(8)
        self.X_posterior = np.array(initial_state)
        self.P_posterior = np.array(self.P)
        self.Z = np.array(initial_state)



    def kalman_update(self, obs_box):
        ###observation value input
        ###obs_box : List of observation coordinates
        obs_box = np.array(obs_box)
        dx = obs_box[0] - self.X_posterior[0]
        dy = obs_box[1] - self.X_posterior[1]
        dw = obs_box[2] - self.X_posterior[2]
        dh = obs_box[3] - self.X_posterior[3]
        self.Z[0:4] = np.array([obs_box]).T
        self.Z[4::] = np.array([dx, dy, dw, dh])

        ###prior estimation
        X_prior = np.dot(self.A, self.X_posterior)
        ###state estimation co-matrix P
        P_prior_1 = np.dot(self.A, self.P_posterior)
        P_prior = np.dot(P_prior_1, self.A.T) + self.Q
        ##kalman K
        k1 = np.dot(P_prior, self.H.T)
        k2 = np.dot(np.dot(self.H, P_prior), self.H.T) + self.R
        K = np.dot(k1, np.linalg.inv(k2))
        ##posterior estimation
        X_posterior_1 = self.Z - np.dot(self.H, X_prior)
        self.X_posterior = X_prior + np.dot(K, X_posterior_1)
        P_posterior_1 = np.eye(8) - np.dot(K, self.H)
        self.P_posterior = np.dot(P_posterior_1, P_prior)
        return self.X_posterior[0:4][:,0].tolist()

    def redetect(self, image, search_factor, info: dict):
        ###redetect
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        ##track again
        with torch.no_grad():

            if self.online_size==1:
                out_dict = self.network(self.template, self.online_template, search, run_score_head=True, memory_info = {
                    'prototype': self.memory_prototype
                })
            else:
                out_dict = self.network.forward_test(search, run_score_head=True, memory_info = {
                    'prototype': self.memory_prototype
                })
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()

        pred_bbox = out_dict['coord'].squeeze()

        pred_bbox = box_xyxy_to_cxcywh(pred_bbox)

        pred_bbox = (pred_bbox * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        pred_bbox = self.map_box_back(pred_bbox, resize_factor)

        if pred_score > self.params.threshold:
            ##kalman filter balance
            return self.kalman_update(pred_bbox)
        else:
            return self.kalman_update(self.state)

    def track(self, image, info: dict):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():

            if self.online_size==1:
                out_dict = self.network(self.template, self.online_template, search, run_score_head=True, memory_info = {
                    'prototype': self.memory_prototype
                })
            else:
                out_dict = self.network.forward_test(search, run_score_head=True, memory_info = {
                    'prototype': self.memory_prototype
                })

        if self.debug:
            unc_left = self.get_conf(torch.cat([out_dict['sigma_map_tl'],out_dict['sigma_map_br']],dim = 1), out_dict['coord'].squeeze()[:2])
            unc_right = self.get_conf(torch.cat([out_dict['sigma_map_tl'],out_dict['sigma_map_br']],dim = 1), out_dict['coord'].squeeze()[2:])
            conf_left = self.get_conf(torch.cat([1 - out_dict['sigma_map_tl'],1 - out_dict['sigma_map_br']],dim = 1), out_dict['coord'].squeeze()[:2])
            conf_right = self.get_conf(torch.cat([1 - out_dict['sigma_map_tl'],1 - out_dict['sigma_map_br']],dim = 1), out_dict['coord'].squeeze()[2:])

        pred_bbox = out_dict['coord'].squeeze()

        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()

        pred_bbox = box_xyxy_to_cxcywh(pred_bbox)

        pred_bbox = (pred_bbox * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        pred_bbox = self.map_box_back(pred_bbox, resize_factor)

        _ = self.kalman_update(torch.tensor(pred_bbox))

        if pred_score > self.params.threshold:
            self.state = clip_box(pred_bbox, H, W, margin=10)
        else:
            self.state = clip_box(self.redetect(image, 1.5 * self.search_factor, info), H , W, margin=10)

        if pred_score > self.params.threshold and pred_score > self.max_pred_score:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score
            self.oz_patch_max = z_patch_arr

        if pred_score > self.ppt and pred_score > self.nextscore:
            self.nextprototype = out_dict['prototype']
            self.nextscore = pred_score
            self.update = True
        # ###策略 FIFO
        if self.frame_id % self.memory_update_interval == 0 and self.update:
            assert self.nextprototype is not None
            new_memory_score = torch.cat([self.memory_score[1:],torch.tensor([self.nextscore])],dim = 0)
            new_memory_prototype = torch.cat([self.memory_prototype[:,1:,:], self.nextprototype],dim = 1)
            self.memory_score = new_memory_score
            self.memory_prototype = new_memory_prototype
            self.update = False
            self.nextprototype = None
            self.nextscore = self.ppt


        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size
            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online(self.template, self.online_template)

            self.max_pred_score = -1
        if self.debug:
            return {"target_bbox": [round(x, 3) for x in self.state],
                "pred_score": round(pred_score, 3),
                "unc_left": unc_left,
                "unc_right": unc_right,
                "conf_left": conf_left,
                "conf_right": conf_right,
                }
        else:
            return {"target_bbox": [round(x, 3) for x in self.state],
                "pred_score": round(pred_score, 3)}

    def get_conf(self,heat,coord):
        sz = heat.shape[-1]
        heat = heat.sum(dim = [0,1])
        heat = heat.cpu().detach().numpy()
        coord = coord.cpu().detach().numpy()
        coord = coord * sz
        x = coord[0]
        y = coord[1]
        x1 = np.floor(coord[0])
        x2 = np.ceil(coord[0])
        y1 = np.floor(coord[1])
        y2 = np.ceil(coord[1])

        Q11 = heat[int(np.clip(x1,a_min=0,a_max=sz-1))][int(np.clip(y1,a_min=0,a_max=sz-1))]
        Q12 = heat[int(np.clip(x1,a_min=0,a_max=sz-1))][int(np.clip(y2,a_min=0,a_max=sz-1))]
        Q21 = heat[int(np.clip(x2,a_min=0,a_max=sz-1))][int(np.clip(y1,a_min=0,a_max=sz-1))]
        Q22 = heat[int(np.clip(x2,a_min=0,a_max=sz-1))][int(np.clip(y2,a_min=0,a_max=sz-1))]
        R1 = Q11 * (x2 - x) + Q21 * (x - x1)
        R2 = Q12 * (x2 - x) + Q22 * (x - x1)
        P = R1 * (y2 - y) + R2 * (y - y1)
        return P
    def map_box(self, gt_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        gtcx,gtcy,gtw,gth = gt_box
        half_side = 0.5 * self.params.search_size / resize_factor
        w = gtw
        h = gth
        cx = gtcx - ((cx_prev - half_side) - 0.5 * w)
        cy = gtcy - ((cy_prev - half_side) - 0.5 * h)
        return [cx,cy,w,h]
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)#因为映射crop中心是(half,half)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

def get_tracker_class():
    return UncTrackOnline