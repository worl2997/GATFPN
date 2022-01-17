import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.layers import BasicBlock, Bottleneck, RFC, FUB, add_conv,upsample
from retinanet.utils import BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import torch.nn.functional as F
import torch.nn as nn



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 256 채널에
# origin feature와 updated feature를 기반으로 prediction head로 넘길 피쳐를 생성하는 부분
# 그래프 노드 -> 5개
# fmap_size 계산 : 256 채널, 42x 25

# 백본으로 부터 추출된 feature map을 기반으로 그래프의 입력으로 들어갈
# node_feature h 와 edge feature를 생성해 주는 부분

# 여기서 차원수를 하나 더 늘려도 됨
class Graph_FPN(nn.Module):
    def __init__(self, c2, c3, c4, c5, feat_size): #  [512, 1024, 2048] 순으로 되어있을거임
        super(Graph_FPN, self).__init__()

        # forward 에서 input을 넣을때는 c6,c5,c4,c3 순으로 넣어주어야함
        self.P6_make = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.P7_make = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        )
        self.FUB_level_0 = graph_fusion(level=0)
        self.FUB_level_1 = graph_fusion(level=1)
        self.FUB_level_2 = graph_fusion(level=2)
        self.FUB_level_3 = graph_fusion(level=3)

    def forward(self, c5, c4, c3, c2):
        updated_level_0_feat = self.FUB_level_0(c5, c4, c3, c2)
        updated_level_1_feat = self.FUB_level_1(c5, c4, c3, c2)
        updated_level_2_feat = self.FUB_level_2(c5, c4, c3, c2)
        updated_level_3_feat = self.FUB_level_3(c5, c4, c3, c2)

        # RFC 할지는 알아서 정하기
        return [updated_level_3_feat, updated_level_2_feat, updated_level_1_feat, updated_level_0_feat]


# 1. 모든 채널 사이즈를 256으로 맞추고 시작
# 2. 각 채널사이즈에 맞게 연산한 뒤 256으로 채널변경
class graph_fusion(nn.Module):
    def __init__(self, level):
        super(graph_fusion, self).__init__()
        self.level = level
        self.dim = [2048, 1024, 512]  #실제 feature => [512, 1024, 2048] -> 채널 사이즈를 다 256으로 맞춰야함
        self.inter_dim =self.dim[self.level]

        # 각 level을 기준으로 reshape
        if level==0:
            self.resize_level_1 = add_conv(1024,self.inter_dim, 3, 2, leaky=False) # 3x3 conv 한번
            self.resize_level_2 = nn.Sequential(    # max-pool ->3x3conv
                nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
                add_conv(512, self.inter_dim, 3, 2, leaky=False),
            )
            self.FUB_0= FUB(2048,r=2,node_size=3)

        elif level==1:
            self.resize_level_0 = nn.Sequential(
                add_conv(2048, self.inter_dim, 1, 1, leaky=False), # stride_level_0 -> 차원수 줄이고 크기 한번 확장
                upsample(scale_factor=2, mode='nearest'),
            )
            self.resize_level_2 = add_conv(512, self.inter_dim, 3, 2, leaky=False) # 3x3conv 한번
            self.FUB_1= FUB(1024,r=4,node_size=4)

        elif level==2: # 512 기준
            self.resize_level_0 = nn.Sequential(
                add_conv(2048, self.inter_dim, 1, 1, leaky=False), # 채널 줄이고 scale factor - 4
                upsample(scale_factor=4, mode='nearest'),
            )
            self.resize_level_1 = nn.Sequential(
                add_conv(1024, self.inter_dim, 1, 1, leaky=False), # 채널 줄이고 크기 1번확장
                upsample(scale_factor=2, mode='nearest'),
            )
            self.FUB_2= FUB(512,r=4,node_size=4)




    def forward(self, p_0, p_1, p_2, p_3):
        if self.level==0:
            level_0_resized = p_0
            level_1_resized = self.resize_level_1(p_1)
            level_2_resized = self.resize_level_2(p_2)
            level_3_resized = self.resize_level_3(p_3)
        elif self.level==1:
            level_0_resized = self.resize_level_0(p_0)
            level_1_resized = p_1
            level_2_resized = self.resize_level_2(p_2)
            level_3_resized = self.resize_level_3(p_3)
        elif self.level==2:
            level_0_resized = self.resize_level_0(p_0)
            level_1_resized = self.resize_level_1(p_1)
            level_2_resized = p_2
            level_3_resized = self.resize_level_3(p_3)
        elif self.level==3:
            level_0_resized = self.resize_level_0(p_0)
            level_1_resized = self.resize_level_1(p_1)
            level_2_resized = self.resize_level_2(p_2)
            level_3_resized = p_3



class RegressionModel(nn.Module): #들어오는 feature 수를 교정해 주어야함
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):
        # layers -> 각각 layer를 몇번 반복사는지 알려줌
        #  ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    def __init__(self, num_classes, block, layers):
        self.node_channel_size = 256 # 일단 임의로 이렇게 지정

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # C1 -> output_size 56x56 (이미지 사이즈에 따라서 다름)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #C2 -> output_size 28x28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #C3 -> 14x14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #C4 -> 7x7

        if block == BasicBlock:
            fpn_channel_sizes = [self.layer1[layers[0] - 1].conv2.out_channels , self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_channel_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.GCN_FPN = Graph_FPN() # 백본으로 부터 나온 feature map들의 채널사이즈를 입력으로 받아서 node_feature를 생성하는 부분

        self.regressionModel = RegressionModel(256) # 256 차원이라..
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # 마지막 블록의 conv2 의 out channel을 따로 뽑아낼 수 있음
        return nn.Sequential(*layers)


    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x) # 256
        x2 = self.layer2(x1) # 512
        x3 = self.layer3(x2) # 1024
        x4 = self.layer4(x3) # 2045

        enhanced_feat = self.GCN_FPN(x1,x2,x3,x4, [i.size() for i in [x1,x2,x3,x4]]) #
        regression = torch.cat([self.regressionModel(feature) for feature in enhanced_feat], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in enhanced_feat], dim=1)
        anchors = self.anchors(img_batch)



        if self.training:
            return self.focalLoss(classification, regression,anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


