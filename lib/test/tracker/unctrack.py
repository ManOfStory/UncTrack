import matplotlib.pyplot as plt

from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.unctrack import build_unctrack
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from lib.test.tracker.tracker_utils import vis_attn_maps
from lib.utils.box_ops import box_xyxy_to_cxcywh,box_cxcywh_to_xyxy
from scipy.ndimage import zoom
import numpy as np
class UncTrackOnline(BaseTracker):
    def __init__(self,params, dataset_name):
        super(UncTrackOnline, self).__init__(params)
        network = build_unctrack(params.cfg, train=False)
        miss,unexpected = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = 1 #params.debug
        self.frame_id = 0

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
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL

        self.update_interval = self.update_intervals[0]

        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        print("Update interval is: ", self.update_interval)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        self.z_patch_arr = z_patch_arr
        self.oz_patch_arr = z_patch_arr

        if self.params.vis_attn==1:
            self.z_patch = z_patch_arr
            self.oz_patch = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        self.search_factor = self.params.search_factor

        self.state = info['init_bbox']
        self.min_uncertainty = float('inf')
        self.maxscore = 0
        self.online_min_template = template
        self.online_forget_id = 0
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info:dict):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            out_dict = self.network(self.template, self.online_template, search)

        pred_bbox = out_dict['coord'].squeeze()

        pred_bbox_ = pred_bbox * x_patch_arr.shape[0]
        pred_bbox_ = pred_bbox_.detach().cpu().numpy().astype('uint64')

        pred_bbox = box_xyxy_to_cxcywh(pred_bbox)

        if self.debug:
            box_color = (0, 0, 255)
            gt_box_color = (0, 255, 0)
            import numpy as np
            gt_box = np.array(info['init_bbox'])
            if not np.isnan(gt_box).any():
                # x y w h 转换成 cx cy w h
                gt_box = np.array(self.map_box(gt_box, resize_factor))
                gt_box = gt_box * resize_factor
                gt_box = box_cxcywh_to_xyxy(torch.from_numpy(gt_box)).detach().cpu().numpy()
                gt_box = np.clip(gt_box, a_min=0, a_max=x_patch_arr.shape[0])
                gt_box = gt_box.astype('uint64')
                try:
                    cv2.rectangle(x_patch_arr, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color=gt_box_color,
                                  thickness=2)
                except:
                    print("error")
            try:
                cv2.rectangle(x_patch_arr, (pred_bbox_[0], pred_bbox_[1]), (pred_bbox_[2], pred_bbox_[3]), color=box_color,
                              thickness=2)
            except:
                print("error")
            x_patch_arr_ = cv2.cvtColor(x_patch_arr, cv2.COLOR_BGR2RGB)
            path = "/home/guoyang/Projects/UncFormer/test/tracking_results/uncformer_convmae/baseline_2/heatmap/" + \
                   info['seq_name'] + '/' + 'original_patch'
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(path + "/{}.jpg".format(self.frame_id), x_patch_arr_)


        pred_bbox = (pred_bbox * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        #uncertainty = uncertainties.max()

        self.state = clip_box(self.map_box_back(pred_bbox, resize_factor), H, W, margin=10)

        unc_left = self.get_conf(out_dict['sigma_map_tl'], out_dict['coord'].squeeze()[:2])
        unc_right = self.get_conf(out_dict['sigma_map_br'], out_dict['coord'].squeeze()[2:])
        conf_left = self.get_conf(1 - out_dict['sigma_map_tl'], out_dict['coord'].squeeze()[:2])
        conf_right = self.get_conf(1 - out_dict['sigma_map_br'], out_dict['coord'].squeeze()[2:])
        #show z_patch_arr oz_patch_arr x_patch_arr
        if self.debug:
            names = ['sigma_map_tl','sigma_map_br']


            # for name in names:
            #    self.get_heat_map(x_patch_arr, 1 - out_dict[name], 100, info['seq_name'], self.frame_id, 'rev' + name)
            path = "/home/guoyang/Projects/UncFormer/test/tracking_results/uncformer_convmae/baseline_2/heatmap/" + \
                            info['seq_name'] + '/'
            self.get_heat_map(x_patch_arr,2 - out_dict[names[0]] - out_dict[names[1]], self.frame_id, path + 'conf_map')

        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                self.online_template = self.preprocessor.process(z_patch_arr)
                self.oz_patch_arr = z_patch_arr

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)

        return {
            "target_bbox": [round(x, 1) for x in self.state],
            "unc_left":unc_left,
            "unc_right":unc_right,
            "conf_left":conf_left,
            "conf_right":conf_right,
            #"uncertainty": round(uncertainty.item(), 3),
            #"uncertainties": {self.frame_id: uncertainties.detach().cpu().numpy().tolist()},  # 10,
        }
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

    def show_heat_map(self,z,path):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        x = np.array(([[i for i in range(0,288)]]))
        x_r = np.repeat(x, [288], axis=0)

        y = np.array(([[i for i in range(0,288)]])).T
        y_r = np.repeat(y, [288], axis=1)

        c = plt.pcolormesh(x_r, y_r, z, cmap='viridis_r', shading='gouraud')  # 彩虹热力图
        # c = plt.pcolormesh(x_r, y_r, z, cmap='viridis_r')# 普通热力图
        #plt.colorbar(c, label='AUPR')
        plt.axis('off')
        plt.savefig(path, dpi=300)
        #plt.show()

    def show3d_map(self,origin_image,high_res_data,path,frame):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # 添加地面图片
        # 创建示例数据
        #plt.imshow(origin_image / 255.0)
        #plt.show()

        origin_image = np.flip(origin_image,axis=0) #.transpose(1,0,2)
        high_res_data = np.flip(high_res_data,axis=0) #.T

        min = 0
        max = origin_image.shape[0]
        x = np.linspace(min, max - 1, max)
        y = np.linspace(min, max - 1, max)
        x, y = np.meshgrid(x, y)
        z = high_res_data
        # 创建一个3D图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制3D密度图
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

        # 在底部添加平面
        xx, yy = np.meshgrid(np.linspace(min, max - 1, max), np.linspace(min, max - 1, max))
        zz = np.zeros_like(xx)  # 平面的 z 坐标
        ax.plot_surface(xx, yy, zz, facecolors=origin_image / 255.0, shade=False)
        # 去掉刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # 去掉网格
        ax.grid(False)
        # 设置坐标轴标签
        ax.set_xlabel('Y Axis')
        ax.set_ylabel('X Axis')
        ax.set_zlabel('Z Axis')

        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + '/{}.jpg'.format(frame))
        plt.close()
        #plt.show()
    def get_heat_map(self, origin_image, feature_map, frameid , path):
        """
        :param origin_image: [H,W,C],[H,W,C]
        :param feature_map:  [C',H',W']] or [[H',W']
        :return:None
        """
        # origin_image = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),origin_image))
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.sum(dim=[0, 1]).detach().cpu().numpy()

        high_res_data = zoom(feature_map, 4, order=1)

        #self.show3d_map(origin_image,high_res_data,path+'/3d/orig',frameid)

        scaled_data = cv2.normalize(high_res_data, None, 0, 255, cv2.NORM_MINMAX)

        #self.show3d_map(origin_image,scaled_data,path+'/3d/norm',frameid)

        smoothed_data = cv2.GaussianBlur(scaled_data, (0, 0), sigmaX=5, sigmaY=5)
        heatmap_smooth = cv2.applyColorMap(np.uint8(smoothed_data), cv2.COLORMAP_JET)

        #cv2.imshow('heatmap', heatmap)
        #cv2.imshow('smooth_heatmap', heatmap_smooth)
        #cv2.waitKey(5)

        if not os.path.exists(path + '/heatmap'):
            os.makedirs(path + '/heatmap')
        cv2.imwrite(path + '/heatmap/{}.jpg'.format(frameid), heatmap_smooth)

    def map_box(self, gt_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        gtcx,gtcy,gtw,gth = gt_box
        half_side = 0.5 * self.params.search_size / resize_factor
        w = gtw
        h = gth
        cx = gtcx - ((cx_prev - half_side) - 0.5 * w)
        cy = gtcy - ((cy_prev - half_side) - 0.5 * h)
        return [cx,cy,w,h]

def get_tracker_class():
    return UncTrackOnline