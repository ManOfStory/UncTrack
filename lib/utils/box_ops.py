import math
import torch
from torchvision.ops.boxes import box_area
import numpy as np
from torch.nn import functional as F
from torch import nn
from torch.nn.functional import l1_loss

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou

class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
    #[n,batch,C,H,W]
    def forward(self, pred, target, weights=None):
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)
        if target.dim() == 4:
            target = target.unsqueeze(0)
        if weights is not None and weights.dim() == 4:
            weights = weights.unsqueeze(0)
        pred = pred.permute(0, 1, 3, 4, 2).reshape(-1, 4) # nf x ns x x 4 x h x w
        target = target.permute(0, 1, 3, 4, 2).reshape(-1, 4) #nf x ns x 4 x h x w

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_union = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect + 1e-7
        ious = (area_intersect) / (area_union)
        gious = ious - (ac_union - area_union) / ac_union

        losses = 1 - gious

        if weights is not None and weights.sum() > 0:
            weights = weights.permute(0, 1, 3, 4, 2).reshape(-1) # nf x ns x x 1 x h x w
            loss_mean = losses[weights>0].mean()
            ious = ious[weights>0]
        else:
            loss_mean = losses.mean()

        return loss_mean, ious

class CIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def ltrb2xyxy(self,target):
        """
        :param target: N,4 ltrb
        #中心点默认(0,0)
        :return: target N,4 xyxy
        """
        target[:,:2] = -target[:,:2]
        return target
    def ciou_loss(self,bboxes1,bboxes2):
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        cious = torch.zeros((rows, cols))
        if rows * cols == 0:
            return cious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            cious = torch.zeros((cols, rows))
            exchange = True
        w1 = bboxes1[:, 2] - bboxes1[:, 0]
        h1 = bboxes1[:, 3] - bboxes1[:, 1]
        w2 = bboxes2[:, 2] - bboxes2[:, 0]
        h2 = bboxes2[:, 3] - bboxes2[:, 1]
        area1 = w1 * h1
        area2 = w2 * h2
        center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2.
        center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2.
        center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2.
        center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2.

        inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
        inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
        inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
        inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
        inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

        c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
        c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
        c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
        c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

        union = area1 + area2 - inter_area
        u = (inter_diag) / c_diag
        iou = inter_area / union
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        with torch.no_grad():
            S = (iou > 0.5).float()
            alpha = S * v / (1 - iou + v)
        cious = iou - u - alpha * v
        cious = torch.clamp(cious, min=-1.0, max=1.0)
        if exchange:
            cious = cious.T
        return 1 - cious, iou

    #[n,batch,C,H,W]
    def forward(self, pred, target, weights=None):
        assert pred.dim() == 4
        assert target.dim() == 4
        assert weights is None or weights.dim() == 4
        pred = pred.permute(0, 2, 3, 1).reshape(-1, 4) # B,C,H,W -> N,4
        target = target.permute(0, 2, 3, 1).reshape(-1, 4) # B,C,H,W -> N,4
        #假定中心是0,0 要把ltrb转换成xyxy
        pred = self.ltrb2xyxy(pred)
        target = self.ltrb2xyxy(target)
        if weights is not None:
            weights = weights.permute(0, 2, 3, 1).reshape(-1) # B,C,H,W -> N,
            pred = pred[weights]
            target = target[weights]

        losses,ious = self.ciou_loss(pred,target)

        loss_mean = losses.mean()

        return loss_mean, ious

def ciou_loss(bboxes1, bboxes2):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    eps = 1e-8
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 0] + bboxes1[:, 2]) / 2.
    center_y1 = (bboxes1[:, 1] + bboxes1[:, 3]) / 2.
    center_x2 = (bboxes2[:, 0] + bboxes2[:, 2]) / 2.
    center_y2 = (bboxes2[:, 1] + bboxes2[:, 3]) / 2.

    inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

    c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / (c_diag + eps)
    iou = inter_area / (union + eps)
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))), 2)
    with torch.no_grad():
        S = (iou>0.5).float()
        alpha= S*v/(1-iou+v+eps)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return torch.mean(1-cious), iou


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]


class REGLoss(nn.Module):
    def __init__(self, dim=4, loss_type='iou'):
        super(REGLoss, self).__init__()
        self.dim = dim
        if loss_type == 'iou':
            self.loss = IOULoss()
        else:
            raise ValueError("Only support iou loss.")

    def forward(self, output, ind, target, radius=1, norm=1/20.):
        width, height = output.size(-2), output.size(-1)
        output = output.view(-1, self.dim, width, height)
        # mask =  mask.view(-1, 2)
        target = target.view(-1, self.dim)
        ind = ind.view(-1, 1)
        center_w = (ind % width).int().float()
        center_h = (ind / width).int().float()

        # regress for the coordinates in the vicinity of the target center, the default radius is 1.
        if radius is not None:
            loss = []
            for r_w in range(-1 * radius, radius + 1):
                for r_h in range(-1 * radius, radius + 1):
                    target_wl = target[:, 0] + r_w * norm
                    target_wr = target[:, 1] - r_w * norm
                    target_ht = target[:, 2] + r_h * norm
                    target_hb = target[:, 3] - r_h * norm
                    if (target_wl < 0.).any() or (target_wr < 0.).any() or (target_ht < 0.).any() or (target_hb < 0.).any():
                        continue
                    if (center_h + r_h < 0.).any() or (center_h + r_h >= 1.0 * width).any() \
                            or (center_w + r_w < 0.).any() or (center_w + r_w >= 1.0 * width).any():
                        continue

                    target_curr = torch.stack((target_wl, target_wr, target_ht, target_hb), dim=1)  # [num_images * num_sequences, 4]
                    ind_curr = ((center_h + r_h) * width + (center_w + r_w)).long()
                    pred_curr = _tranpose_and_gather_feat(output, ind_curr)
                    loss_curr = self.loss(pred_curr, target_curr)
                    loss.append(loss_curr)
            if len(loss) == 0:
                pred = _tranpose_and_gather_feat(output, ind.long())  # pred shape: [num_images * num_sequences, 4]
                loss = self.loss(pred, target)
                return loss
            loss = torch.stack(loss, dim=0)
            loss = torch.mean(loss, dim=0)
            return loss
        pred = _tranpose_and_gather_feat(output, ind.long())     # pred shape: [num_images * num_sequences, 4]
        loss = self.loss(pred, target)

        return loss

class IOULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 2]
        pred_right = pred[:, 1]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 2]
        target_right = target[:, 1]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            if self.reduction == 'mean':
                return losses.mean()
            else:
                return losses.sum()

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(dim=1, index=ind)   # [num_images * num_sequences, 1, 2]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat.view(ind.size(0), dim)


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        # print("pred shape: {}, label shape: {}".format(prediction.shape, label.shape))
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


def Gaussian(y, mu, var):
    eps = 0.3
    epsilon = 1e-6
    result = (y-mu)/(var+epsilon)
    result = (result**2)/2*(-1)
    exp = torch.exp(result)
    result = exp/(math.sqrt(2*math.pi))/(var + eps)

    return result

def NLL_loss(bbox_gt, bbox_pred, bbox_var):
        #bbox_var = torch.sigmoid(bbox_var)
        prob = Gaussian(bbox_gt, bbox_pred, bbox_var)

        return prob

#先假设x和y独立
class UncBoxLoss_CORNER(nn.Module):
    def __init__(self,use_gpu=True):
        super(UncBoxLoss_CORNER, self).__init__()
        self.use_gpu = use_gpu

    def log_sum_exp(self,x):
        x_max = x.data.max()
        return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

    def forward(self, data):
        """
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
        """

        prob_tl,prob_br = data['prob_tl'],data['prob_br']
        sigma_x_tl,sigma_y_tl = data['sigma_x_tl'],data['sigma_y_tl']
        sigma_x_br,sigma_y_br = data['sigma_x_br'],data['sigma_y_br']
        gt_x_tl,gt_y_tl = data['gt_x_tl'],data['gt_y_tl']
        gt_x_br,gt_y_br = data['gt_x_br'],data['gt_y_br']
        coord_x,coord_y = data['coord_x'],data['coord_y']
        B = prob_tl.shape[0]
        epsi = 10 ** -9
        balance = 2.0

        loss_tl = NLL_loss(gt_x_tl,coord_x,sigma_x_tl) * NLL_loss(gt_y_tl,coord_y,sigma_y_tl)
        loss_tl = torch.clamp(loss_tl,min = 0.0,max = 1.0)
        # prob version
        loss_tl = -prob_tl * torch.log(loss_tl + epsi) / balance
        loss_tl = loss_tl.sum()


        loss_br = NLL_loss(gt_x_br, coord_x, sigma_x_br) * NLL_loss(gt_y_br, coord_y, sigma_y_br)
        loss_br = torch.clamp(loss_br, min=0.0, max=1.0)
        # prob version
        loss_br = -prob_br * torch.log(loss_br + epsi) / balance
        loss_br = loss_br.sum()


        loss_l = loss_tl + loss_br
        loss_l = loss_l / B
        return loss_l


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target, weights=None):
        assert pred.dim() == 4
        assert target.dim() == 4
        assert weights is None or weights.dim() == 4
        pred = pred.permute(0, 2, 3, 1).reshape(-1, 4)  # B,C,H,W -> N,4
        target = target.permute(0, 2, 3, 1).reshape(-1, 4)  # B,C,H,W -> N,4
        if weights is not None:
            weights = weights.permute(0, 2, 3, 1).reshape(-1) # B,C,H,W -> N,
            pred = pred[weights]
            target = target[weights]
        losses = l1_loss(pred, target)
        loss_mean = losses.mean()
        return loss_mean