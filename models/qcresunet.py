import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import generate_model

def init_subpixel(weight):
    co, ci, h, w = weight.shape
    co2 = co // 4
    # initialize sub kernel
    k = torch.empty([co2, ci, h, w])
    nn.init.kaiming_uniform_(k)
    # repeat 4 times
    k = k.repeat_interleave(4, dim=0)
    weight.data.copy_(k)

def init_linear(m, relu=True):
    if relu: nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    else: nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)


class ECA3d(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x).squeeze(-1)
        
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act=True):
        padding = (kernel_size - 1) // 2
        layers = [
          nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
          nn.InstanceNorm3d(out_channels)
        ]
        if act: layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
    
    def reset_parameters(self):
        init_linear(self[0])
        self[1].reset_parameters()


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

class UpsampleShuffle(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        
    def reset_parameters(self):
        init_subpixel(self[0].weight)
        nn.init.zeros_(self[0].bias)


class UpsampleBilinear(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )
    
    def reset_parameters(self):
        init_linear(self[0])


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_t = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv_t(x)
    
    def reset_parameters(self):
        init_linear(self.conv_t)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upsample, skip=True):
        super().__init__()
        if skip:
            self.up = upsample(in_channels, in_channels // 2)
        else:
            self.up = upsample(in_channels, in_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        else:
            return self.conv(x1)


class QCResUNet(nn.Module):
    def __init__(self, 
                 network_depth, 
                 num_input_channels, 
                 n_classes,
                 out_channels, 
                 upsample=UpsampleBilinear, 
                 skip=True,
                 drop_path=0.0,
                 dropout=0.0,
                 do_reg=True,
                 do_ds=True):
        super().__init__()
        self.do_reg = do_reg
        self.do_ds = do_ds
        self.skip = skip

        self.input_channel_attn  = ECA3d()

        self.resnet_encoder = generate_model(network_depth, 
                                             num_input_channels=num_input_channels, 
                                             n_classes=n_classes, 
                                             conv1_t_stride=2, 
                                             feature_extarctor=True,
                                             drop_path=drop_path)
        
        block_inplanes = self.resnet_encoder.block_inplanes # [32, 64, 128, 256]
        
        if skip:
            self.up1 = Up(block_inplanes[2], block_inplanes[1], upsample)
            self.up2 = Up(block_inplanes[1], block_inplanes[1], upsample)
            self.up3 = Up(block_inplanes[1], block_inplanes[0], upsample)
        else:
            self.up1 = nn.Conv3d(block_inplanes[3], block_inplanes[1], 3, stride=1, padding=12, dilation=12, bias=True)
            self.up2 = nn.Conv3d(block_inplanes[1], 2, 3, stride=1, padding=12, dilation=12, bias=True)
            # self.up3 = Up(128, 64, upsample, False)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(block_inplanes[3], n_classes))
        
        if self.do_ds:
            self.upsacle = nn.Upsample(scale_factor=2, mode='trilinear')
            self.ds1 = nn.Conv3d(block_inplanes[1], out_channels, 1, bias=False)
            self.ds2 = nn.Conv3d(block_inplanes[1], out_channels, 1, bias=False)
            self.ds3 = nn.Conv3d(block_inplanes[0], out_channels, 1, bias=False) 

            self.chan_attn1 = ECA3d()
            self.chan_attn2 = ECA3d()
            self.chan_attn3 = ECA3d()
        
        self.seg_out = nn.Conv3d(block_inplanes[0], out_channels, 1, bias=False)
        self.chan_attn_out = ECA3d()
        self.seg_out_final = nn.Conv3d(out_channels*2, out_channels, 1, bias=False)

    def forward(self, x):
        input_size = x.size()
        query_mask = x[:, 4:, ...]

        x = self.input_channel_attn(x)

        if self.skip:
            x_reg, cahce_dict = self.resnet_encoder(x, intermediate=True)
            # print(x_reg.shape)

            x = self.up1(cahce_dict[-1], cahce_dict[-2])
            # print(x.shape)
            x_seg1 = self.upsacle(self.ds1(x))
            x_seg1 = torch.cat([x_seg1, F.interpolate(query_mask, scale_factor=0.25, mode='trilinear')], 1)
            x_seg1 = self.chan_attn1(x_seg1)
            # print(x_seg1.shape)

            x = self.up2(x, cahce_dict[-3])
            x_seg2 = self.upsacle(self.ds2(x))
            x_seg2 = torch.cat([x_seg2, F.interpolate(query_mask, scale_factor=0.5, mode='trilinear')], 1)
            x_seg2 = self.chan_attn2(x_seg2)

            x = self.up3(x, cahce_dict[-4]) 
            x_seg3 = self.upsacle(self.ds3(x))
            x_seg3 = torch.cat([x_seg3, query_mask], 1)
            x_seg3 = self.chan_attn3(x_seg3)
          
            x = self.upsacle(x)
            x = self.seg_out(x)
            x = torch.cat([x, query_mask], 1)
            x = self.chan_attn_out(x)
          
            x_seg = self.upsacle(self.upsacle(x_seg1) + x_seg2) + x_seg3 + x
            x_seg = self.seg_out_final(x_seg)
        else:
            x_reg = self.resnet_encoder(x, intermediate=False)
            x = self.up1(x_reg)
            x = self.up2(x)
            x_seg = F.interpolate(x, tuple(list(input_size[-3:])), mode='trilinear')

        if self.do_reg:
            x_reg = F.adaptive_avg_pool3d(x_reg, 1).flatten(1)
            x_reg = self.fc(x_reg)

            return x_reg, x_seg
        else:
            return x_seg

def qcresunet34(num_input_channels=5, n_classes=1, out_channels=1, **kwargs):
    return QCResUNet(34, num_input_channels, n_classes, out_channels, **kwargs)

def qcresunet50(num_input_channels=5, n_classes=1, out_channels=1, **kwargs):
    return QCResUNet(50, num_input_channels, n_classes, out_channels, **kwargs)

if __name__ == "__main__":
    # from thop import profile
    model = qcresunet34()
    input = torch.randn(1, 5, 160, 192, 160)
    model(input)
