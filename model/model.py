import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchinfo import summary
from thop import profile

# from CrossViewAttention import CMAtt,CMAtt2
# from model.CrossViewAttention import CMAtt,CMAtt2
# from model.fuseDiff import FusionNetwork
# from fuseDiff import FusionNetwork

class Att_Enhance(nn.Module):

    def __init__(self, out_channels):

        super().__init__()
        self.w = nn.Sequential(
            nn.Conv2d(out_channels *2 , out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.cse =  Channel_Attention(out_channels)
        self.sse = Spatial_Attention(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gas, bg):
        weight = self.relu(torch.cat([gas, bg], dim=1))
        weight = self.w(weight)
        out = gas * weight
        ca = self.cse(out)
        out = out * ca
        sa = self.sse(out)
        out = out * sa
        return out

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gmp_pool(x).permute(0, 2, 3, 1)
        x = self.down(x)
        max_out = self.up(self.act(x)).permute(0, 3, 1, 2)
        out = self.sigmoid(max_out)
        return out

class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv1(x)
        out = self.softmax(x1)
        return out

class Att_avg_pool(nn.Module):
    def __init__(self, dim, reduction):
        super(Att_avg_pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5, force_dropout: bool = False):
        super().__init__()
        self.force_dropout = force_dropout
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training or self.force_dropout)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Global Contextual module

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1_1 = BasicConv2d(
            out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_2 = BasicConv2d(
            out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))

        self.branch2 = BasicConv2d(in_channel, out_channel, 1)
        self.branch2_1 = BasicConv2d(
            out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2))
        self.branch2_2 = BasicConv2d(
            out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0))

        self.branch3 = BasicConv2d(in_channel, out_channel, 1)
        self.branch3_1 = BasicConv2d(
            out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_2 = BasicConv2d(
            out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0))

        # self.se = Att_avg_pool(out_channel, 4)
        # self.catt_list =nn.ModuleList()
        # for i in range(4):
        #     self.catt_list.append(Channel_Attention(out_channel))
        # self.attncat = AttentionConcat()
        self.catt = Channel_Attention(4*out_channel,4)
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        # x0 = self.se(x0)
        x1 = self.branch1_2(self.branch1_1(self.branch1(x)))
        # x1 = self.se(x1)
        x2 = self.branch2_2(self.branch2_1(self.branch2(x)))
        # x2 = self.se(x2)
        x3 = self.branch3_2(self.branch3_1(self.branch3(x)))
        # x3 = self.se(x3)

        # x_list = [x0,x1,x2,x3]

        # for i in range(4):
        #     catt = self.catt_list[i]
        #     x_list[i] = catt(x_list[i])
        
        # x_cat =self.attncat(x_list)
        # x_cat = self.conv_cat(x_cat)
        # x_add = x0 + x1 + x2 + x3

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = x_cat * self.catt(x_cat)
        x_cat = self.conv_cat(x_cat)
        
        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation_init(nn.Module):
    def __init__(self, channel,n_class=2, mode='None'):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        #########################################################################################################
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        # self.se = Att_avg_pool(3 * channel, 4)
        if mode == 'out':
            self.conv = nn.Conv2d(3*channel, n_class, 1)
        if mode == 'splat':
            self.conv = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        ##########################################################################################################
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(
            x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        ##########################################################################################################
        x2_2 = self.conv_concat2(
            torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(
            torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        x = self.conv(self.conv4(x3_2))
        return x

# Refinement flow

class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, attention, x1, x2, x3):
        x1 = x1 + torch.mul(x1, self.upsample2(attention))
        x2 = x2 + torch.mul(x2, self.upsample2(attention))
        x3 = x3 + torch.mul(x3, attention)
        return x1, x2, x3



class mini_Aspp(nn.Module):
    def __init__(self, channel):
        super(mini_Aspp, self).__init__()
        self.conv_6 = nn.Conv2d(
            channel, channel, kernel_size=3,  stride=1, padding=6,  dilation=6)
        self.conv_12 = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv_18 = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.bn(self.conv_6(x))
        x2 = self.bn(self.conv_12(x))
        x3 = self.bn(self.conv_18(x))
        feature_map = x1 + x2 + x3
        return feature_map
####################################################################################################

class FA_encoder(nn.Module):
    def __init__(self, num_resnet_layers,dropout_rate: float = .09):
        super(FA_encoder, self).__init__()
        self.num_resnet_layers = num_resnet_layers
        if self.num_resnet_layers == 50:
            resnet_raw_model = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model = models.resnet152(pretrained=True)
            self.inplanes = 2048
        ########  Thermal ENCODER  ########
        self.encoder_conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_conv1.weight.data = torch.unsqueeze(
            torch.mean(resnet_raw_model.conv1.weight.data, dim=1), dim=1)
        self.encode_dropout = MCDropout(dropout_rate)

        self.encoder_bn1 = resnet_raw_model.bn1
        self.encoder_relu = resnet_raw_model.relu
        self.encoder_maxpool = resnet_raw_model.maxpool
        self.encoder_layer1 = resnet_raw_model.layer1
        self.encoder_layer2 = resnet_raw_model.layer2
        self.encoder_layer3 = resnet_raw_model.layer3
        self.encoder_layer4 = resnet_raw_model.layer4

        self.ae0 = Att_Enhance(64)
        self.ae1 = Att_Enhance(256)
        self.ae2 = Att_Enhance(512)
        self.ae3 = Att_Enhance(1024)
        self.ae4 = Att_Enhance(2048)

    def forward(self, bg, gas):
        ######################################################################
        # layer0
        ######################################################################

        combined = torch.cat((bg, gas), dim=0)  # 拼接后的形状 (batch_size_bg + batch_size_gas, channels, height, width)
        # 获取 bg 和 gas 的批次大小
        batch_size_bg = bg.size(0)
        batch_size_gas = gas.size(0)

        combined = self.encoder_conv1(combined)

        combined = self.encoder_bn1(combined)

        combined = self.encoder_relu(combined)

        combined = self.encoder_maxpool(combined)

        ######################################################################
        bg_out = combined[:batch_size_bg]  # 分离出 bg 的部分
        gas_out = combined[batch_size_bg:batch_size_bg + batch_size_gas]  # 分离出 gas 的部分
        
        fuse0 = self.ae0(bg_out,gas_out)

        ######################################################################
        # layer1
        ######################################################################
        combined = self.encoder_layer1(combined)
        bg1 = combined[:batch_size_bg]  # 分离出 bg 的部分
        gas1 = combined[batch_size_bg:batch_size_bg + batch_size_gas]  # 分离出 gas 的部分
        fuse1 = self.ae1(gas1,bg1)
        ######################################################################
        # layer2
        ######################################################################
        combined = self.encoder_layer2(combined)
        bg2 = combined[:batch_size_bg]  # 分离出 bg 的部分
        gas2 = combined[batch_size_bg:batch_size_bg + batch_size_gas]  # 分离出 gas 的部分
        fuse2 = self.ae2(gas2,bg2)
        ######################################################################
        # layer3
        ######################################################################
        combined = self.encoder_layer3(combined)
        bg3 = combined[:batch_size_bg]  # 分离出 bg 的部分
        gas3 = combined[batch_size_bg:batch_size_bg + batch_size_gas]  # 分离出 gas 的部分
        fuse3 = self.ae3(gas3,bg3)
        ######################################################################
        # layer4
        ######################################################################
        combined = self.encoder_layer4(combined)
        bg4 = combined[:batch_size_bg]  # 分离出 bg 的部分
        gas4 = combined[batch_size_bg:batch_size_bg + batch_size_gas]  # 分离出 gas 的部分
        fuse4 = self.ae4(gas4,bg4)
        ######################################################################
        fuse = [fuse0, fuse1, fuse2, fuse3, fuse4]
        gas = [gas, gas1, gas2, gas3, gas4]
        bg = [bg, bg1, bg2, bg3, bg4]
        # weight = [weight0,weight1,weight2,weight3,weight4]
        return fuse, gas, bg

class TransBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(
                planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out

class Cascaded_decoder(nn.Module):
    def __init__(self, n_class=2, channel=64):
        super(Cascaded_decoder, self).__init__()
        ########  DECODER  ########
        self.rfb2_1 = GCM(512, channel)
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(256, channel)
        self.rfb2_2 = GCM(512, channel)
        self.agg1 = aggregation_init(channel,n_class, mode='out')
        self.agg1_splat = aggregation_init(channel, mode='splat')
        self.agg2 = aggregation_final(channel)
        self.miniaspp = mini_Aspp(channel)
        self.HA = Refine()
        ######################################################################
        # upsample function
        self.upsample = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        ######################################################################
        self.inplanes = channel
        self.agant1 = self._make_agant_layer(channel*3, channel)
        self.deconv1 = self._make_transpose_layer(
            TransBottleneck, channel, 3, stride=2)
        # self.inplanes = channel/2
        self.agant2 = self._make_agant_layer(channel, channel)
        self.deconv2 = self._make_transpose_layer(
            TransBottleneck, channel, 3, stride=2)
        self.out2_conv = nn.Conv2d(channel, n_class, kernel_size=1)
        ######################################################################

    def _make_transpose_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes))
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))

        self.inplanes = planes
        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x):
        ######################################################################
        bg, bg1, bg2, bg3, bg4 = x[0], x[1], x[2], x[3], x[4]
        ######################################################################
        # produce initial saliency map by decoder1
        ######################################################################
        x2_1 = self.rfb2_1(bg2)
        x3_1 = self.rfb3_1(bg3)
        x4_1 = self.rfb4_1(bg4)
        attention_gate = torch.sigmoid(self.agg1_splat(x4_1, x3_1, x2_1))
        ##############################################################################
        x, x1, x2 = self.HA(attention_gate, bg, bg1, bg2)
        x0_2 = self.rfb0_2(x)
        x1_2 = self.rfb1_2(x1)
        x2_2 = self.rfb2_2(x2)
        # ux5_2 = self.upsample2(x5_2)
        ##############################################################################
        # feature_map = ux5_2 + ux1_2 + ux2_1 + ux3_1 + ux0_2 + ux4_1
        # feature_map = self.miniaspp(feature_map)
        ##############################################################################
        hight_output = self.upsample(self.agg1(x4_1, x3_1, x2_1))
        ##############################################################################
        # Refine low-layer features by initial map
        ##############################################################################
        # PTM module
        ##############################################################################
        # y = feature_map
        y = self.agg2(x2_2,x1_2,x0_2)
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        ######################################################################
        return hight_output, y
        

class GasSegNet(nn.Module):
    def __init__(self, n_class,num_resnet_layers=50):
        super(GasSegNet, self).__init__()
        self.FA_encoder = FA_encoder(num_resnet_layers)
        self.CD = Cascaded_decoder(n_class)


    def forward(self, bg, gas):
        fuse,gas,bg = self.FA_encoder(bg,gas)
        diff = fuse

        out, out_1 = self.CD(diff)
        return out, out_1, fuse, gas,bg


def unit_test():
    num_minibatch = 2
    bg = torch.randn(num_minibatch, 1, 512, 640).cuda(0)
    gas = torch.randn(num_minibatch, 1, 512, 640).cuda(0)
    RTCAN_Net = GasSegNet(2).cuda(0)
    # input = torch.cat((bg, gas), dim=1)
    out = RTCAN_Net(bg, gas)
    print(out)

def summarize():
    model = GasSegNet(2).cuda(0)
    summary(model, input_size=(4,4,512,640))

def model_stat():
    model = GasSegNet(2).cuda(0)
    gas = torch.randn(2,1,512,640).cuda()
    bg = torch.randn(2,1,512,640).cuda()
    from thop import profile

    flops, params = profile(model, inputs=(gas,bg, ))
    # flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))


if __name__ == '__main__':
    # summarize()
    # unit_test()
    model_stat()