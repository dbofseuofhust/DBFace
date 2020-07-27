import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.models as models
import math
import torch.utils.model_zoo as model_zoo

# refer: https://github.com/d-li14/mobilenetv2.pytorch
_MODEL_URL = "https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth"

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrain(self):
        checkpoint = model_zoo.load_url(_MODEL_URL)
        self.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, in_size):
        super(FPN,self).__init__()
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1)
        self.output4 = conv_bn1X1(in_channels_list[3], out_channels, stride = 1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)
        self.merge3 = conv_bn(out_channels, out_channels)

        self.in_size = in_size

    def forward(self, input):
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        output4 = self.output4(input[3])

        up4 = F.interpolate(output4, size=[i//16 for i in self.in_size], mode="nearest")
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[i//8 for i in self.in_size], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[i//4 for i in self.in_size], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return output1, output2, output3, output4

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class ASFF(nn.Module):
    def __init__(self, in_channel, out_channel, in_size):
        super(ASFF, self).__init__()
        self.compress_level_0 = ConvBNReLU(in_channel, in_channel, 1, 1)
        self.expand = ConvBNReLU(in_channel, out_channel, 3, 1)
        compress_c = 8
        self.weight_level_0 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_level_1 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_level_2 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_level_3 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.in_size = in_size

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized =F.interpolate(level_0_compressed, size=[i//4 for i in self.in_size], mode='nearest')
        level_1_resized =F.interpolate(x_level_1, size=[i//4 for i in self.in_size], mode='nearest')
        level_2_resized =F.interpolate(x_level_2, size=[i//4 for i in self.in_size], mode='nearest')

        level_3_resized =x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:3,:,:]+\
                            level_3_resized * levels_weight[:,3:,:,:]

        out = self.expand(fused_out_reduced)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    else:
        pass

class MDFace(nn.Module):
    """
    Centernet + ttfnet
    Backone：mobilenetv2(0.5，relu替换relu6)
    neck：   4层FPN + ASFF融合 + SSH
    后处理：  softnms
    大小大概：2 Mb
    代码层面优化：
        1、查表法减均值除方差
        2、图像W/H原比例缩放和填充input_ptr合并
        3、尽量减少多余乘法，数组替换vector
        (resize也有点耗时，听说Opencv4.2加速了)
    https://github.com/whoNamedCody/Mask-Face-Detection
    """
    def __init__(self, width_mult=0.5, in_size=(800, 800), has_landmark=False):
        super(MDFace, self).__init__()

        self.has_landmark = has_landmark

        net = MobileNetV2(width_mult=width_mult)
        net.load_pretrain()

        features = net.features
        self.layer1= nn.Sequential(*features[0:4])
        self.layer2 = nn.Sequential(*features[4:7])
        self.layer3 = nn.Sequential(*features[7:14])
        self.layer4 = nn.Sequential(*features[14:18])
        fpn_channels = {0.5: [16, 16, 48, 160], 1:[24, 32, 96, 320]}
        self.fpn = FPN(fpn_channels[width_mult], 32, in_size)
        self.asff = ASFF(32, 32, in_size)
        self.ssh = SSH(32, 24)
        self.conv_bn = conv_bn(24, 24, 1)
        self.head_hm = nn.Conv2d(24, 1, 1)
        self.head_tlrb = nn.Conv2d(24, 4, 1)

        if self.has_landmark: 
            self.head_landmark = nn.Conv2d(24, 10, 1)

        self.fpn.apply(weights_init_kaiming)
        self.asff.apply(weights_init_kaiming)
        self.ssh.apply(weights_init_kaiming)

    def init_weights(self):

        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19

        # Load backbone weights from ImageNet
        self.head_hm.init_normal(0.001, d)
        self.head_tlrb.init_normal(0.001, 0)

        if self.has_landmark:
            self.head_landmark.init_normal(0.001, 0)
    
    def load(self, file):
        checkpoint = torch.load(file, map_location="cpu")
        self.load_state_dict(checkpoint)

    def forward(self, x):
        enc0 = self.layer1(x) # 24
        enc1 = self.layer2(enc0) # 32
        enc2 = self.layer3(enc1) # 96
        enc3 = self.layer4(enc2) # 320
        out1, out2, out3, out4 = self.fpn([enc0, enc1, enc2, enc3])
        out = self.asff(out4, out3, out2, out1)
        out = self.ssh(out)
        out = self.conv_bn(out)
        sigmoid_hm = self.head_hm(out)
        tlrb = self.head_tlrb(out)
        
        if self.has_landmark:
            lankmark = self.head_landmark(out)
            return sigmoid_hm, tlrb, lankmark
        return sigmoid_hm, tlrb

if __name__ == "__main__": 

    model = MDFace(width_mult=0.5)
    x = torch.randn(2, 3, 640, 640)
    out = model(x)
    state = {'model': model.state_dict()}
    torch.save(state, 'model.pth')
    for k, v in out.items():
        print(k, v.shape)