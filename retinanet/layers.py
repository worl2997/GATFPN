import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing
from retinanet.utils import total_pixel_size


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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class conv1x1(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(conv1x1, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv1 = nn.Conv2d(self.in_feat, self.out_feat, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)




# # node_feauter은 forward의 input으로 들어감
# GCN 기반으로 이미지 feature map을 업데이트 하는 부분
# # node_feauter은 forward의 input으로 들어감

class FUB(nn.Module):
    def __init__(self, channels, r, node_size):
        super(FUB, self).__init__()
        # 직접 해당 클래스 안에서 input_feature를 기반으로 그래프를 구현해야 함
        self.node_num = node_size
       # self.make_score = MS_CAM(channels, r)
        self.w = nn.Parameter(torch.Tensor(5, 1))
        self.cl_0 = conv1x1(2 * 256, 256)
        self.cl_1 = conv1x1(2 * 256, 256)
        self.cl_2 = conv1x1(2 * 256, 256)
        self.cl_3 = conv1x1(2 * 256, 256)
        self.cl_4 = conv1x1(2 * 256, 256)

    # 입력 받은 feature node  리스트를 기반으로 make_distance로 edge를 계산하고
    def make_edge_matirx(self, node_feats, pixels):
        Node_feats = node_feats
        edge_list = torch.zeros(pixels, self.node_num, self.node_num).to(torch.cuda.current_device())
        conv_dic = {0:self.cl_0, 1:self.cl_1, 2:self.cl_2, 3:self.cl_3, 4 :self.cl_4}

        for i, node_i in enumerate(Node_feats):
            for j, node_j in enumerate(Node_feats):
                x_add = node_i + node_j  # elementwise add
                c_cat = torch.cat([node_j, x_add], dim=1)  # 이거는 C x H x W 차원일 기준으로 하것
                target = conv_dic[i](c_cat)
                distance = ((node_j - target).abs())
                score  = distance.reshape(1, -1)
                edge_list[:,i,j] = score

        return edge_list

    # graph 와 node feature matrix 반환
    def make_node_matrix(self, node_feats,pixels):
        # 여기에 그래프 구성 코드를 집어넣으면 됨
        init_matrix = torch.Tensor(self.node_num, pixels)

        for i, node in enumerate(node_feats):
            init_matrix[i] = node.reshape(1, -1)

        node_feat = init_matrix.T
        node_feature_matirx = node_feat.unsqueeze(-1)
        return node_feature_matirx

    def normalize_edge(self, input, type, t):
        # pruning -> Normalize (softmax)
        out = torch.where(input > t, input, torch.zeros(size=input.size()))
        out_ = F.softmax(out, dim=2)  #F.normalize(out, p=type, dim=2)

        # Normalize -> pruning
        # out = F.softmax(input, dim=2)
        # out_ = torch.where(out > t, out, torch.zeros(size=input.size()))

        return out_

    def feat_fusion(self, edge, node):
        h = edge.matmul(node)
        result = h.squeeze(-1)
        out = result.T
        return out

    def resize_back(self,ori_s, h):
        out =h.reshape(self.node_num, ori_s[0],ori_s[1],ori_s[2],ori_s[3])
        return out

    def forward(self, x):
        node_feats = x  # list form으로 구성되어있음 [re_c1,.., re_c2] 5개의 피쳐맵들 존재
        pixels = total_pixel_size(node_feats[0])
        edge_matrix = self.make_edge_matirx(node_feats,pixels)
        node_feats_list = self.make_node_matrix(node_feats,pixels)
        node_feats_matrix = node_feats_list.to(torch.cuda.current_device())

        # 노말라이즈가 필요한지 판단하고 필요하다면 아래 모듈 구현해서 추가하기
        normalized_edge = self.normalize_edge(edge_matrix, 2, 0.3).to(torch.cuda.current_device())

        h = self.feat_fusion(normalized_edge, node_feats_matrix)
        out = self.resize_back(node_feats[0].shape,h)
        return out


class MS_CAM(nn.Module):

    def __init__(self, channels=256, r=2):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        refined  = x * wei
        return refined

class RFC(nn.Module):
    def __init__(self, feat_size):
        super(RFC, self).__init__()

        self.rfc_0  = nn.Sequential(
            nn.Conv2d(feat_size * 2, feat_size, kernel_size=1, stride=1, padding=0,bias=False),
            nn.Conv2d(feat_size, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
        )
        self.rfc_1 = nn.Sequential(
            nn.Conv2d(feat_size * 2, feat_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(feat_size, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
        )
        self.rfc_2 = nn.Sequential(
            nn.Conv2d(feat_size * 2, feat_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(feat_size, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
        )
        self.rfc_3 = nn.Sequential(
            nn.Conv2d(feat_size * 2, feat_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(feat_size, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
        )
        self.rfc_4 = nn.Sequential(
            nn.Conv2d(feat_size * 2, feat_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(feat_size, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
        )
        self.rf_0 = MS_CAM(256,2)
        self.rf_1 = MS_CAM(256,2)
        self.rf_2 = MS_CAM(256,2)
        self.rf_3 = MS_CAM(256,2)
        self.rf_4 = MS_CAM(256,2)

    def forward(self, origin, h):
        result_feat = []
        rfc_dic = {0:self.rfc_0, 1:self.rfc_1, 2:self.rfc_2, 3:self.rfc_3, 4 :self.rfc_4}
        rf_dic = {0:self.rf_0, 1:self.rf_1, 2:self.rf_2, 3:self.rf_3, 4 :self.rf_4}
        i = 0
        for origin_feat, updated_feat in zip(origin, h):
            feat = torch.cat([origin_feat, updated_feat], dim=1)
            #rfc = rfc_dic[i]
            out = rfc_dic[i](feat)
            refined = rf_dic[i](out)
            result_feat.append(refined)
            i = i+1
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

