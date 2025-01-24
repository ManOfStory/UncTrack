from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_xywh
import torch
import copy
import cv2

class UncTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings , run_score_head = False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

    def __call__(self, data):
        out_dict = self.forward_pass(data, run_score_head=self.run_score_head)

        # process the groundtruth
        # gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        gt_bboxes = data['search_anno']

        labels = None
        if 'pred_scores' in out_dict:
            try:
                labels = data['label'].view(-1)  # (batch, ) 0 or 1
            except:
                raise Exception("Please setting proper labels for score branch.")

        # self.show_heat_map(data['search_origin_images'][0], out_dict, labels)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0], labels=labels)
        return loss, status

    def forward_pass(self,data,run_score_head = False):

        search_bboxes = box_xywh_to_xyxy(data['search_anno'][0].clone())
        # out_dict = self.net(data['template_images'][0], data['template_images'][1], data['search_images'][0],
        #                     run_score_head = run_score_head, gt_bboxes=search_bboxes)
        memory_info = None
        if data.get('memory_search_anno',None) is not None:
            try:
                memory_info = {
                    "memory_search_anno": data['memory_search_anno'],
                    "memory_search_images": data['memory_search_images'],
                    "memory_template_images": data['memory_template_images'],
                }
            except:
                raise ValueError("memory not complete!")
            out_dict = self.net(data['template_images'][0], data['template_images'][1], data['search_images'][0],
                            run_score_head=run_score_head, gt_bboxes=search_bboxes,memory_info = memory_info)
        else: #stage1 train
            out_dict = self.net(data['template_images'][0], data['template_images'][1], data['search_images'][0],
                                run_score_head=run_score_head, gt_bboxes=search_bboxes)

        # out_dict: B,4
        return out_dict

    def compute_losses(self,pred_dict,gt_bboxes,labels):

        coord = pred_dict['coord'].clone()
        if torch.isnan(coord).any():
            raise ValueError("coord Network outputs is NAN! Stop Training")

        pred_boxes_vec = coord.view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bboxes).view(-1, 4).clamp(min=0.0,max=1.0)

        B = coord.shape[0]
        prob_tl = pred_dict['prob_vec_tl'].clone().view(B, -1)  # B 72x72
        prob_br = pred_dict['prob_vec_br'].clone().view(B, -1)  # B 72x72
        sigma_x_tl = pred_dict['sigma_map_tl'][:, 0, :, :].clone().view(B, -1)
        sigma_y_tl = pred_dict['sigma_map_tl'][:, 1, :, :].clone().view(B, -1)
        sigma_x_br = pred_dict['sigma_map_br'][:, 0, :, :].clone().view(B, -1)
        sigma_y_br = pred_dict['sigma_map_br'][:, 1, :, :].clone().view(B, -1)
        gt_x_tl = gt_boxes_vec[:, 0].unsqueeze(-1).repeat(1, sigma_x_tl.shape[-1])
        gt_y_tl = gt_boxes_vec[:, 1].unsqueeze(-1).repeat(1, sigma_y_tl.shape[-1])
        gt_x_br = gt_boxes_vec[:, 2].unsqueeze(-1).repeat(1, sigma_x_br.shape[-1])
        gt_y_br = gt_boxes_vec[:, 3].unsqueeze(-1).repeat(1, sigma_y_br.shape[-1])

        coord_x = pred_dict['coord_map_x'].clone().unsqueeze(0).repeat(B, 1)
        coord_y = pred_dict['coord_map_y'].clone().unsqueeze(0).repeat(B, 1)



        data = {
            "prob_tl":prob_tl,
            "prob_br":prob_br,
            "sigma_x_tl":sigma_x_tl,
            "sigma_y_tl":sigma_y_tl,
            "sigma_x_br":sigma_x_br,
            "sigma_y_br":sigma_y_br,
            "gt_x_tl":gt_x_tl,
            "gt_y_tl":gt_y_tl,
            "gt_x_br":gt_x_br,
            "gt_y_br":gt_y_br,
            "coord_x":coord_x,
            "coord_y":coord_y
        }

        #ciou
        try:
            ciou_loss, iou = self.objective['ciou'](pred_boxes_vec, gt_boxes_vec)
        except:
            #raise ValueError('cious zero')
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        mean_iou = iou.mean()
        #unc
        try:
            loss_l_unc = self.objective['unc'](data)
        except:
            loss_l_unc = 0.0
        #l1
        try:
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        except:
            l1_loss = 0.0

        try:
            loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['unc'] * loss_l_unc
        except:
            loss = 0.0

        if 'pred_scores' in pred_dict:
            #Jitter the logits
            #eps = 10**-9
            #rand = torch.randn(pred_dict['pred_sigmas'].shape).to(pred_dict['pred_sigmas'].device)
            #sqrt = torch.sqrt(pred_dict['pred_sigmas'] + eps)
            #jitterd_scores = pred_dict['pred_scores'] + sqrt * rand
            #score_loss = self.objective['score'](jitterd_scores, labels)
            score_loss = self.objective['score'](pred_dict['pred_scores'],labels)
            loss = score_loss * self.loss_weight['score']

            status = {
                'Loss/total':loss.item(),
                "Loss/score_loss": score_loss.item(),
            }
        else:
            status = {"Loss/total": loss.item(),
                  "Loss/ciou": ciou_loss.item(),
                  "Loss/l1": l1_loss.item(),
                  "IoU": mean_iou.item()}
            if loss_l_unc:
                status.update({"Loss/unc": loss_l_unc.item()})

        return loss,status
