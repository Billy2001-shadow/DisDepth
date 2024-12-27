import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .tiny_vit import TinyViT
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels=[128, 160, 320, 320], 
        out_channels=[48, 96, 192, 384], 
        aligned_channels=64,
        spatial_sizes=[(60, 80), (30, 40), (15, 20), (15, 20)],
        upsample_factors=[4, 4, 4, 2],
        input_size=(480, 640),
        use_bn=False, 
    ):
        super(DPTHead, self).__init__()
        
        self.embed_dim = in_channels
        self.out_channels = out_channels
        self.spatial_sizes = spatial_sizes  # 四层征图的尺寸
        self.input_size = input_size


        # 将embed_dim通道数的特征图映射到out_channels通道数
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0)
            for in_c, out_c in zip(self.embed_dim, self.out_channels)
        ])
        
        # 对四层特征图分别进行 四倍、四倍、四倍、两倍 上采样 因为最后两层特征图的尺寸相同，所以只对最后一层进行两倍上采样，这样四层特征图的size分别就是两倍的关系
        self.upsample_factors = upsample_factors # 每层的上采样倍数
        self.resize_layers = nn.ModuleList([
            # 输入输出通道相同，该层的作用是上采样(保持通道不变)
            nn.ConvTranspose2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=factor,  # kernel_size与上采样倍数相等
                stride=factor,
                padding=0
            )
            for out_c, factor in zip(out_channels, self.upsample_factors)
        ])
        
       
        # scratch主要有四个模块：layerx_rn(通道对齐)、refinenetx(逐层向上特征融合)、output_conv1(减少一半通道数)、output_conv2(1.通道数减少到32、2.再由32到1、3.最后sigmoid到[0,1])
        self.scratch = _make_scratch(
            out_channels,
            aligned_channels,
            groups=1,
            expand=False,
        )
                
        self.scratch.refinenet1 = _make_fusion_block(aligned_channels, use_bn) 
        self.scratch.refinenet2 = _make_fusion_block(aligned_channels, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(aligned_channels, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(aligned_channels, use_bn)
        
        head_features_1 = aligned_channels
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features):
        out = []
       
        for i, x in enumerate(out_features):

            spatial_h, spatial_w = self.spatial_sizes[i]  # 获取各特征层对应的尺寸
            
            # 将特征的序列重塑成(batch_size, embed_dim, height, width)形式
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], spatial_h, spatial_w))
            
            x = self.projects[i](x) # torch.Size([16, 128, 28, 28])
            x = self.resize_layers[i](x) # 112*112 56*56 28*28 14*14
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1) # 统一调整到256通道
        layer_2_rn = self.scratch.layer2_rn(layer_2) # 112*112 56*56 28*28 14*14 均为256通道
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, self.input_size, mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class TinyVitDpt(nn.Module):
    def __init__(
        self, 
        config,
        embed_dims=[64, 128, 160, 320], 
        features=64, 
        in_channels=[128, 160, 320, 320],
        out_channels=[48, 96, 192, 384], 
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
        use_bn=False, 
    ):
        super(TinyVitDpt, self).__init__()
        
        
        self.input_size = (config.input_height, config.input_width)
        # 修改一下img_size=224 
        self.pretrained = TinyViT(img_size=self.input_size,  # 480 640
            embed_dims=embed_dims, # 5M
            num_heads=num_heads,
            window_sizes=window_sizes,
            drop_path_rate=drop_path_rate,
        )

        w,h = self.input_size
        spatial_sizes=[(h//8, w//8), (h//16, w//16), (h//32, w//32), (h//32, w//32)] # 480 640 -> [(60, 80), (30, 40), (15, 20), (15, 20)],
        upsample_factors=[4, 4, 4, 2]                                        #            [(240, 320), (120, 160), (60, 80), (30, 40)]
        self.depth_head = DPTHead(in_channels,out_channels,features,spatial_sizes,upsample_factors,self.input_size,use_bn)
        
      
    def forward(self, x):
        features_extraction = self.pretrained(x)
        depth = self.depth_head(features_extraction) 
        depth = F.relu(depth)
        return depth
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=224):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        # depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=224):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=False, # 保持宽高比 True
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)