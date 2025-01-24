"""
PMN: Prototype Memory Network
"""
import torch
import torch.nn as nn
from lib.models.unctrack.head import MLP
import torch.nn.functional as F

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), nn.ReLU(inplace=True),
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class QSCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads == 1, "currently only implement num_heads==1"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.drop_prob = 0.1

    def forward(self, prototype, q_x, qry_attn_mask):
        q = self.q_fc(prototype)
        k = self.k_fc(q_x)
        v = self.v_fc(q_x)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if qry_attn_mask is not None:
            qry_attn_mask = (1 - qry_attn_mask).unsqueeze(-2).float()
            qry_attn_mask = qry_attn_mask * -10000.0
            attn = attn + qry_attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MAtt(nn.Module):
    def __init__(self,
                 embed_dims=768,
                 dropout=0.1,
                 ):
        super(MAtt, self).__init__()
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.feedforward_channels = embed_dims * 3
        self.attn = QSCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout)
        self.LN1 = nn.LayerNorm(embed_dims)
        self.FFN = FFN(embed_dims, self.feedforward_channels, dropout=self.dropout)
        self.LN2 = nn.LayerNorm(embed_dims)
    def forward(self,q,kv,mask):
        prototype = self.attn(q,kv,mask)
        prototype = self.LN1(prototype)
        prototype = self.FFN(prototype)
        prototype = self.LN2(prototype)
        return prototype

class UncertaintyAwareScoreDecoder(nn.Module):
    #without roi

    def __init__(self, hidden_dim=768, c_dim = 384, nlayer_head=3, memo_group = 3, topk = 3):
        super().__init__()
        self.topk = topk
        self.memo_group = memo_group
        self.hidden_dim = hidden_dim
        #self.prototype = nn.Embedding(1, hidden_dim)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.search_embd = nn.Conv2d(hidden_dim + c_dim,hidden_dim,1)

        self.proto2search = MAtt(hidden_dim)

        self.q_fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.q_fc_temp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_fc_temp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_fc_temp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_temp = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.out_score = MLP(hidden_dim, hidden_dim, 1, nlayer_head)

    ###input:
    # confidence map
    # bbox mask
    # search token
    # template token

    def proto2memo(self, prototype, memo_prototype):
        ###B,1,768 B,3,768
        dim = prototype.shape[-1]
        scale = dim ** -0.5
        q = self.q_fc(prototype)
        k = self.k_fc(memo_prototype)
        v = self.v_fc(memo_prototype)
        attn = (q @ k.transpose(-2,-1)) * scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = self.proj(x)
        return prototype + x

    def proto2template(self, prototype, template):
        ###B,1,768 B,8x8,768
        dim = prototype.shape[-1]
        scale = dim ** -0.5
        q = self.q_fc_temp(prototype)
        k = self.k_fc_temp(template)
        v = self.v_fc_temp(template)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = self.proj_temp(x)
        return prototype + x

    def trans_box_to_mask(self, bbox, H, W):
        #trans bbox xyxy [B,4] to mask [B,1,H,W]
        B = bbox.shape[0]
        mask = torch.zeros(B, 1, H, W).to(bbox.device)
        box = bbox.clone()
        box = torch.clip(box, min=0.0, max=1.0)
        box[:,0::2] = box[:,0::2] * W
        box[:,1::2] = box[:,1::2] * H
        box[:,:2] = torch.floor(box[:,:2])
        box[:,2:] = torch.ceil(box[:,2:])
        for b in range(B):
            mask[b,:,int(box[b,1]):int(box[b,3]),int(box[b,0]):int(box[b,2])] = 1

        #a = mask[0,0].detach().cpu().numpy()
        return mask

    def gen_prototype(self, sfeat, tfeat, cmap, box):
        H, W = sfeat.shape[-2:]
        search_mask = self.trans_box_to_mask(box, H, W)
        conf_search_feat = self.search_embd(torch.cat([sfeat, cmap], dim=1))
        bs = conf_search_feat.shape[0]
        prototype = self.GAP(tfeat).permute(0,2,3,1).view(-1 , 1, self.hidden_dim)
        conf_search_feat = conf_search_feat.permute(0, 2, 3, 1).view(bs, -1, self.hidden_dim)
        search_mask = search_mask.permute(0, 2, 3, 1).view(bs, -1)
        prototype = self.proto2search(prototype, conf_search_feat, search_mask)
        return prototype

    def forward(self, search_feat, template_feat, search_box, confidence_map, memory_info = None):
        """
        :param search_box: with normalized coords. (x0, y0, x1, y1)
        :return:
        """
        #trans search_bbox -> bbox_mask
        H, W = search_feat.shape[-2:]
        bs = search_feat.shape[0]
        prototype = self.gen_prototype(search_feat, template_feat, confidence_map, search_box)
        search_prototype = prototype.clone()
        if 'prototype' not in memory_info:
            try:
                memo_gt_boxes = memory_info['memory_search_anno'].view(-1, 4)
                memo_gt_boxes[:,2:] = memo_gt_boxes[:,:2] + memo_gt_boxes[:,2:]
                memo_mask = self.trans_box_to_mask(memo_gt_boxes, H, W)
            except:
                memo_mask = self.trans_box_to_mask(memory_info['coord'], H, W)

            memo_search_feat = self.search_embd(torch.cat([memory_info['memory_search_feat'],memory_info['confidence_map']],dim = 1))

            memo_prototype = self.GAP(memory_info['memory_template_feat']).permute(0,2,3,1).view(self.memo_group, bs, -1, self.hidden_dim)

            memo_prototype = memo_prototype.permute(1,0,2,3)

            memo_search_feat = memo_search_feat.view([-1,bs] + list(memo_search_feat.shape[1:]))

            memo_search_feat = memo_search_feat.permute(1,0,3,4,2).contiguous()

            memo_search_feat = memo_search_feat.view(bs,self.memo_group,-1,self.hidden_dim)

            memo_mask = memo_mask.view([-1, bs] + list(memo_mask.shape[1:]))

            memo_mask = memo_mask.permute(1, 0, 3, 4, 2).contiguous()

            memo_mask = memo_mask.view(bs, self.memo_group, -1)

            memo_prototypes = []
            for i in range(self.memo_group):
                memo_prototypes += [self.proto2search(memo_prototype[:,i,...],memo_search_feat[:,i,...],memo_mask[:,i,...])]
            memo_prototype = torch.cat(memo_prototypes,dim = 1)
        else:
            assert not self.training
            memo_prototype = self.select_topk_prototype(memory_info['prototype'], prototype)

        prototype = self.proto2memo(prototype, memo_prototype) #B,1,768

        template_feat = template_feat.permute(0,2,3,1).view(bs,-1,self.hidden_dim)

        prototype = self.proto2template(prototype,template_feat)

        out_scores = self.out_score(prototype)

        return out_scores, search_prototype

    def select_topk_prototype(self, memory_prototype, prototype):
        ###choose topk prototype

        similarity = F.cosine_similarity(memory_prototype, prototype, dim=-1)  # (1, N)

        topk_values, topk_indices = torch.topk(similarity, k = self.topk, dim=-1)  # (1, topk)

        topk_memory_prototype = memory_prototype[:, topk_indices.squeeze(0), :]  # (1, topk, 768)

        return topk_memory_prototype


