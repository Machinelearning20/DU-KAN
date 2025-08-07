import torch
from torch import nn
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from kanlayer import MoKLayer
from utils import *
__all__ = ['DUKAN']
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        layer_hp = [['TaylorKAN', 4], ['TaylorKAN', 4], ['KANLinear', 4], ['KANLinear', 4]]
        if not no_kan:
            self.fc1 = MoKLayer(in_features, hidden_features, layer_hp)
            self.fc2 = MoKLayer(hidden_features, out_features, layer_hp)
            self.fc3 = MoKLayer(hidden_features, out_features, layer_hp)

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)
        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class Cross_Att(nn.Module):
    def __init__(self, dim_s, dim_l):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.att = nn.MultiheadAttention(embed_dim=dim_s, num_heads=4, kdim=dim_l, vdim=dim_l, batch_first=True)
        self.proj_s = nn.Linear(dim_s, dim_s)
        self.proj_l = nn.Linear(dim_l, dim_s)

    def forward(self, s, l, H_s, W_s, H_l, W_l):
        B, N_s, C_s = s.shape
        B, N_l, C_l = l.shape


        s_norm = self.norm_s(s)
        l_norm = self.norm_l(l)
        attn_output, _ = self.att(query=s_norm, key=l_norm, value=l_norm)
        s = s + self.proj_s(attn_output)


        l_proj = self.proj_l(l_norm)

        l_up = F.interpolate(l_proj.permute(0, 2, 1).view(B, -1, H_l, W_l),
                             size=(H_s, W_s), mode='bilinear').flatten(2).transpose(1, 2)
        return s + l_up



class DUKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[256, 320, 512], no_kan=False, drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        kan_input_dim = embed_dims[0]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # -------------------------- （patch_size=4）--------------------------
        self.encoder1_4 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2_4 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3_4 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.patch_embed3_4 = PatchEmbed(img_size=img_size // 4, patch_size=4, stride=2,
                                         in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4_4 = PatchEmbed(img_size=img_size // 8, patch_size=4, stride=2,
                                         in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.block1_4 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan
        )])
        self.block2_4 = nn.ModuleList([KANBlock(
            dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan
        )])
        self.norm3_4 = norm_layer(embed_dims[1])
        self.norm4_4 = norm_layer(embed_dims[2])

        # -------------------------- （patch_size=8）--------------------------
        self.encoder1_8 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2_8 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3_8 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.patch_embed3_8 = PatchEmbed(img_size=img_size // 4, patch_size=8, stride=2,
                                         in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4_8 = PatchEmbed(img_size=img_size // 8, patch_size=8, stride=2,
                                         in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.block1_8 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan
        )])
        self.block2_8 = nn.ModuleList([KANBlock(
            dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan
        )])
        self.norm3_8 = norm_layer(embed_dims[1])
        self.norm4_8 = norm_layer(embed_dims[2])

        # -------------------------- TIF --------------------------
        self.cross_att1 = Cross_Att(dim_s=kan_input_dim//8, dim_l=kan_input_dim//8)
        self.cross_att2 = Cross_Att(dim_s=kan_input_dim//4, dim_l=kan_input_dim//4)
        self.cross_att3 = Cross_Att(dim_s=embed_dims[0], dim_l=embed_dims[0])
        self.cross_att4 = Cross_Att(dim_s=embed_dims[1], dim_l=embed_dims[1])


        self.change1 = nn.Conv2d(kan_input_dim//8, kan_input_dim//8, kernel_size=1)
        self.change2 = nn.Conv2d(kan_input_dim//4, kan_input_dim//4, kernel_size=1)
        self.change3 = nn.Conv2d(kan_input_dim//1, kan_input_dim//1, kernel_size=1)
        self.change4 = nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=1)

        # -------------------------- decoder--------------------------
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan
        )])
        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan
        )])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])
        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)


        self.final_s2 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.final_s3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        img_size = x.shape[2]

        out4 = F.relu(F.max_pool2d(self.encoder1_4(x), 2, 2))
        t1_4 = out4
        out4 = F.relu(F.max_pool2d(self.encoder2_4(out4), 2, 2))
        t2_4 = out4
        out4 = F.relu(F.max_pool2d(self.encoder3_4(out4), 2, 2))
        t3_4 = out4

        out4, H4, W4 = self.patch_embed3_4(out4)
        for blk in self.block1_4:
            out4 = blk(out4, H4, W4)
        out4 = self.norm3_4(out4)
        out4 = out4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        t4_4 = out4

        out4, H4, W4 = self.patch_embed4_4(out4)
        for blk in self.block2_4:
            out4 = blk(out4, H4, W4)
        out4 = self.norm4_4(out4)
        out4 = out4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()


        out8 = F.relu(F.max_pool2d(self.encoder1_8(x), 2, 2))
        t1_8 = out8
        out8 = F.relu(F.max_pool2d(self.encoder2_8(out8), 2, 2))
        t2_8 = out8
        out8 = F.relu(F.max_pool2d(self.encoder3_8(out8), 2, 2))
        t3_8 = out8

        # KAN
        out8, H8, W8 = self.patch_embed3_8(out8)
        for blk in self.block1_8:
            out8 = blk(out8, H8, W8)
        out8 = self.norm3_8(out8)
        out8 = out8.reshape(B, H8, W8, -1).permute(0, 3, 1, 2).contiguous()
        t4_8 = out8

        out8, H8, W8 = self.patch_embed4_8(out8)
        for blk in self.block2_8:
            out8 = blk(out8, H8, W8)
        out8 = self.norm4_8(out8)
        out8 = out8.reshape(B, H8, W8, -1).permute(0, 3, 1, 2).contiguous()


        # Fusion 1
        H1, W1 = t1_4.shape[2], t1_4.shape[3]
        H1_8, W1_8 = t1_8.shape[2], t1_8.shape[3]
        t1_flat4 = t1_4.flatten(2).transpose(1, 2)  # (B, N, C)
        t1_flat8 = t1_8.flatten(2).transpose(1, 2)
        t1_fuse = self.cross_att1(t1_flat4, t1_flat8, H1, W1, H1_8, W1_8)
        t1 = t1_fuse.transpose(1, 2).view(B, -1, H1, W1)
        t1 = self.change1(t1)

        # Fusion 2
        H2, W2 = t2_4.shape[2], t2_4.shape[3]
        H2_8, W2_8 = t2_8.shape[2], t2_8.shape[3]
        t2_flat4 = t2_4.flatten(2).transpose(1, 2)
        t2_flat8 = t2_8.flatten(2).transpose(1, 2)
        t2_fuse = self.cross_att2(t2_flat4, t2_flat8, H2, W2, H2_8, W2_8)
        t2 = t2_fuse.transpose(1, 2).view(B, -1, H2, W2)
        t2 = self.change2(t2)

        # Fusion 3
        H3, W3 = t3_4.shape[2], t3_4.shape[3]
        H3_8, W3_8 = t3_8.shape[2], t3_8.shape[3]
        t3_flat4 = t3_4.flatten(2).transpose(1, 2)
        t3_flat8 = t3_8.flatten(2).transpose(1, 2)
        t3_fuse = self.cross_att3(t3_flat4, t3_flat8, H3, W3, H3_8, W3_8)
        t3 = t3_fuse.transpose(1, 2).view(B, -1, H3, W3)
        t3 = self.change3(t3)

        # Fusion 4
        H4, W4 = t4_4.shape[2], t4_4.shape[3]
        H4_8, W4_8 = t4_8.shape[2], t4_8.shape[3]
        t4_flat4 = t4_4.flatten(2).transpose(1, 2)
        t4_flat8 = t4_8.flatten(2).transpose(1, 2)
        t4_fuse = self.cross_att4(t4_flat4, t4_flat8, H4, W4, H4_8, W4_8)
        t4 = t4_fuse.transpose(1, 2).view(B, -1, H4, W4)
        t4 = self.change4(t4)
        s2 = t4

        out = (out4 + out8) / 2


        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear'))
        s3 = out
        t4 = F.interpolate(t4, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True) 
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)


        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear'))
        t3 = F.interpolate(t3, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True) 
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)


        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear'))
        t2 = F.interpolate(t2, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True) 
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear'))
        t1 = F.interpolate(t1, size=(out.shape[2], out.shape[3]), mode='bilinear', align_corners=True) 
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))


        s2 = torch.nn.functional.interpolate(s2, size=out.shape[2:], mode='bilinear', align_corners=False)
        s3 = torch.nn.functional.interpolate(s3, size=out.shape[2:], mode='bilinear', align_corners=False)

        return self.final(out), self.final_s2(s2), self.final_s3(s3)
