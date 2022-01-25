import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from retinanet.utils import total_pixel_size


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from retinanet.utils import total_pixel_size


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):

    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage



class conv1x1(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(conv1x1, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv1 = nn.Conv2d(self.in_feat, self.out_feat, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)


class graph_fusion(nn.Module):
    def __init__(self, level):
        super(graph_fusion, self).__init__()
        self.level = level
        # self.dim = [2048, 1024, 512]  #실제 feature => [512, 1024, 2048] -> 채널 사이즈를 다 256으로 맞춰야함
        self.inter_dim = 256

        if level == 0:  # high-level
            self.resize_level_0 = add_conv(2048, self.inter_dim, 1, 1, leaky=False)
            self.resize_level_1 = add_conv(1024, self.inter_dim, 3, 2, leaky=False)  # 3x3 conv 한번
            self.resize_level_2 = nn.Sequential(  # max-pool ->3x3conv
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                add_conv(512, self.inter_dim, 3, 2, leaky=False),
            )
            self.FUB_level_0 = FUB(256, r=4, node_size=3, level=0)


        elif level == 1:  # middle-level
            self.resize_level_0 = nn.Sequential(
                add_conv(2048, self.inter_dim, 1, 1, leaky=False),  # stride_level_0 -> 차원수 줄이고 크기 한번 확장
                upsample(scale_factor=2, mode='nearest'),
            )
            self.resize_level_1 = add_conv(1024, self.inter_dim, 1, 1, leaky=False)  # 3x3 conv 한번
            self.resize_level_2 = add_conv(512, self.inter_dim, 3, 2, leaky=False)  # 3x3conv 한번
            self.FUB_level_1 = FUB(256, r=4, node_size=3, level=1)

        elif level == 2:  # low-level
            self.resize_level_0 = nn.Sequential(
                add_conv(2048, self.inter_dim, 1, 1, leaky=False),  # 채널 줄이고 scale factor - 4
                upsample(scale_factor=4, mode='nearest'),
            )
            self.resize_level_1 = nn.Sequential(
                add_conv(1024, self.inter_dim, 1, 1, leaky=False),  # 채널 줄이고 크기 1번확장
                upsample(scale_factor=2, mode='nearest'),
            )
            self.resize_level_2 = add_conv(512, self.inter_dim, 1, 1, leaky=False)  # 3x3conv 한번
            self.FUB_level_2 = FUB(256, r=4, node_size=3, level=2)

    def forward(self, c_3, c_4, c_5):  # input : [512, 1024, 2048]
        if self.level == 0:  # 최고 차원의 피쳐 (피라미드 꼭대기)
            lev_0_res = self.resize_level_0(c_5)
            lev_1_res = self.resize_level_1(c_4)
            lev_2_res = self.resize_level_2(c_3)
            out = self.FUB_level_0([lev_0_res, lev_1_res, lev_2_res]) # level_0의 사이즈에 모두 맞춰줌
            # leve_0 -> 작은 피쳐 , level_2 -> 큰피
            # out = RFC_0(c_5,out)

        elif self.level == 1:  # 어쩃든 피쳐 채널크기는 모두 256으로 맞춤
            lev_0_res = self.resize_level_0(c_5)
            lev_1_res = self.resize_level_1(c_4)
            lev_2_res = self.resize_level_2(c_3)
            out = self.FUB_level_1([lev_0_res, lev_1_res, lev_2_res])
            # out = RFC_0(c_4,out)

        elif self.level == 2:
            lev_0_res = self.resize_level_0(c_5)
            lev_1_res = self.resize_level_1(c_4)
            lev_2_res = self.resize_level_2(c_3)
            out = self.FUB_level_2([lev_0_res, lev_1_res, lev_2_res])
            # out = RFC_0(c_3,out)
        return out

class MS_CAM(nn.Module):

    def __init__(self, channels=64, r=2):
        super(MS_CAM, self).__init__()
        self.inter_channels = int(channels // r)
        self.bs_c = int(channels//3) # 256
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2,x3):
        x = torch.cat((x1,x2,x3),1)
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg) # 256*3 x H x W
        x1_w = wei[:, 0:self.bs_c,:,:]
        x2_w = wei[:, self.bs_c:self.bs_c*2, :, :]
        x3_w = wei[:, self.bs_c*2:, :, :]
        out = [x1_w.reshape(1,-1), x2_w.reshape(1,-1),x3_w.reshape(1,-1)]
        return out



class FUB(nn.Module):
    def __init__(self, channels, r, node_size,level):
        super(FUB, self).__init__()
        # 직접 해당 클래스 안에서 input_feature를 기반으로 그래프를 구현해야 함
        self.node_num = node_size
        self.level = level
        self.mk_score = MS_CAM(channels*3, r)

    # 입력 받은 feature node  리스트를 기반으로 make_distance로 edge를 계산하고
    def make_edge_matirx(self, node_feats, pixels):
        Node_feats = node_feats
        edge_list = torch.zeros(pixels, 1, self.node_num)
        att_score = self.mk_score(Node_feats[0],Node_feats[1],Node_feats[2])
        for i in range(len(Node_feats)):
            edge_list[:, 0, i] = att_score[i]
        weight = F.softmax(edge_list, dim=2)
        return weight

    def make_node_matrix(self, node_feats,pixels):
        # 여기에 그래프 구성 코드를 집어넣으면 됨
        init_matrix = torch.Tensor(self.node_num, pixels)
        for i, node in enumerate(node_feats):
            init_matrix[i] = node.contiguous().reshape(1, -1)
        node_feat = init_matrix.T.contiguous()
        node_feature_matirx = node_feat.unsqueeze(-1)
        return node_feature_matirx

    def normalize_edge(self, input, t):
        weight = F.softmax(input, dim=2)
        out = torch.where(weight > t, weight, torch.zeros(size=weight.size()))
        return out

    def feat_fusion(self, edge, node):
        h = edge.matmul(node)  # mx1x5 * 5x1
        result = h.squeeze(-1)
        out = result.T.contiguous() # 1xm size
        return out

    def resize_back(self,ori_s, h):
        out = h.reshape(ori_s[0],ori_s[1],ori_s[2],ori_s[3])
        return out

    def forward(self, x):
        node_feats = x  # list form으로 구성
        pixels = total_pixel_size(node_feats[0])
        edge_matrix = self.make_edge_matirx(node_feats,pixels).to(torch.cuda.current_device())
        node_feats_list = self.make_node_matrix(node_feats,pixels)
        node_feats_matrix = node_feats_list.to(torch.cuda.current_device())
        # 노말라이즈가 필요한지 판단하고 필요하다면 아래 모듈 구현해서 추가하기
        #normalized_edge = self.normalize_edge(edge_matrix, 0.2).to(torch.cuda.current_device())
        h = self.feat_fusion(edge_matrix, node_feats_matrix)
        out = self.resize_back(node_feats[0].shape, h)
        return out


class RFC(nn.Module):
    def __init__(self, feat_size):
        super(RFC, self).__init__()
        self.rfc  = nn.Sequential(
            nn.Conv2d(feat_size * 2, feat_size, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Conv2d(feat_size, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
        )
    def forward(self, origin, h):
        result_feat = []
        for origin_feat, updated_feat in zip(origin, h):
            feat = torch.cat([origin_feat, updated_feat], dim=1)
            out = self.rfc(feat)
            result_feat.append(out)
        return result_feat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes