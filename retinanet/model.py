import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.layers import BasicBlock, Bottleneck, RFC, FUB, add_conv
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

class Nodefeats_make(nn.Module):
    def __init__(self, fpn_channels):
        super(Nodefeats_make, self).__init__()
        self.fpn_c = fpn_channels # [256, 512, 1024, 2048] # 레벨별 채널 수
        self.num_backbone_feats = len(self.fpn_c)
        self.target_size = 256

        self.make_C5_ = self.make_C5(self.fpn_c[-1],self.target_size) # C4 -> C5
        self.make_C6_ = nn.Conv2d(self.target_size, self.target_size,kernel_size=3, stride=2, padding=1) # C5 -> C6
        self.make_C1 = nn.Conv2d(in_channels=self.fpn_c[0], out_channels=self.target_size, kernel_size=1, stride=1, padding=0)
        self.make_C2 = nn.Conv2d(in_channels=self.fpn_c[1], out_channels=self.target_size, kernel_size=1, stride=1, padding=0)
        self.make_C3 = nn.Conv2d(in_channels=self.fpn_c[2], out_channels=self.target_size, kernel_size=1, stride=1, padding=0)
        self.make_C4 = nn.Conv2d(in_channels=self.fpn_c[3], out_channels=self.target_size, kernel_size=1, stride=1, padding=0)

    def make_C5(self,in_ch, out_ch):
        stage = nn.Sequential()
        stage.add_module('conv1', nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0))# 1x1 channel resize
        stage.add_module('conv2', nn.Conv2d(out_ch, out_ch,kernel_size=3, stride=2 , padding=1))  # 3x3 conv 2 stride
        return stage


    def forward(self, inputs):
        C1, C2, C3, C4 = inputs
        # C1 ~ C6 : original feature
        C5 = self.make_C5_(C4)
        C6 = self.make_C6_(C5)
        #C1 = self.make_C1(C1)
        C2 = self.make_C2(C2)
        C3 = self.make_C3(C3)
        C4 = self.make_C4(C4)

        return [C2,C3,C4,C5,C6]
       #  채널사이즈만 256으로 모두 맞춰줌


'''
1.channel_resize-> 256 
2.resize to intermediate feature scale 
3.revise the FUB 
4.revise MS_cam based RFC 
'''
class GCN_FPN(nn.Module):
    def __init__(self,
                 in_channels,  # 256
                 num_levels,  # 5
                 refine_level=2,
                 r= 4,
                 conv_cfg=None,
                 norm_cfg=None):
        super(GCN_FPN, self).__init__()
        assert 0 <= refine_level < num_levels


        self.refine =  FUB(self.in_channels, self.r, self.num_levels)  # forward input -> resize node list
        self.RFC = RFC(self.in_channels)  # forward input -> updated feature, origin_feature

    def forward(self, inputs):
        # step 1. backbone으로 부터 받은 multi-level feature들의 channel을 resize해서 통합
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]  # (뒤에 hxw 사이즈 반납)
        # 피쳐들의 사이즈를 특정 기준으로 모두 통일
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(inputs[i], output_size=gather_size)
            else:
                #gathered = nn.Upsample(inputs[i], size=tuple(gather_size), mode='nearest')
                gathered = F.interpolate(inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)


        if self.refine is not None:
            updated_feature = self.refine(feats)  # input : Gather feature
            enhanced_feat = self.RFC(feats, updated_feature)

        # step 3 :scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]  # 해당하는 원래 original size

            if i < self.refine_level:
                # 아래 두개의 feats를 enhanced_feat으로 바꾸어주기
                #residual = nn.Upsample(enhanced_feat[i], size=tuple(out_size), mode='nearest')
                resized_back_feats = F.interpolate(enhanced_feat[i], size=out_size, mode='nearest')
            else:
                resized_back_feats = F.adaptive_max_pool2d(enhanced_feat[i], output_size=out_size)
            outs.append(resized_back_feats)


        return updated_feature



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

        self.Node_feats = Nodefeats_make(fpn_channel_sizes)
        self.GCN_FPN = GCN_FPN(in_channels=256,num_levels=5,refine_level=2,r=4) # 백본으로 부터 나온 feature map들의 채널사이즈를 입력으로 받아서 node_feature를 생성하는 부분

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

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # 이부분을 짜야함
        # 구현이 어려운 이유가 backbone, neck, head가 모두 한부분으로 연결되어있음
        enhanced_feat = self.GCN_FPN([x2,x3,x4])

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


