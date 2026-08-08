import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.sdfs import sdfExtrusion, transform_points
from utils.utils import add_latent


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.global_cross_attention = nn.MultiheadAttention(embed_dim=1536, num_heads=8, batch_first=True)
        self.local_cross_attention = nn.MultiheadAttention(embed_dim=1536, num_heads=8, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        self.projection = nn.Linear(1536, 256, bias=True)
        self.local_projection = nn.Linear(1536, 256, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in (self.projection, self.local_projection):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, dino_global_feat, dino_local_feat, depth_global_feat, depth_local_feat, normal_global_feat, normal_local_feat):
        # vits14: 384, vitb14: 768, vitl14: 1024; vitg14: 1536
        dino_global_feat = dino_global_feat.unsqueeze(1)
        depth_global_feat = depth_global_feat.unsqueeze(1)
        normal_global_feat = normal_global_feat.unsqueeze(1)
        global_attn_output, _ = self.global_cross_attention(query=dino_global_feat, key=depth_global_feat, value=normal_global_feat)
        global_feat = global_attn_output.squeeze(1)

        local_attn_output, _ = self.local_cross_attention(query=dino_local_feat, key=depth_local_feat, value=normal_local_feat)

        global_feat = self.projection(global_feat)  # [B, 256]
        global_feat = global_feat.unsqueeze(1)  # [B, 1, 256]
        local_feat = self.local_projection(local_attn_output)  # [B, num_patches, 256]

        combined_feat, _ = self.cross_attention(query=global_feat, key=local_feat, value=local_feat)
        combined_feat = combined_feat.squeeze(1)  # [B, 256]

        return combined_feat

class Decoder(nn.Module):  
    def __init__(self, ef_dim=32, num_primitives=4):
        super(Decoder, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = ef_dim * 8
        self.num_primitive_parameters_aggregated = 4+3+1

        self.param_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.feature_dim * 2, self.feature_dim * 4, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.feature_dim * 4, int(self.num_primitives*self.num_primitive_parameters_aggregated), bias=True)
        )
        self._initialize_weight()
    
    def _initialize_weight(self):
        for module in self.param_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, feat):
        shapes = self.param_predictor(feat)
        para_3d = shapes[...,:self.num_primitives*(4+3+1)].view(-1, (4+3+1), int(self.num_primitives)) # B,C,P
        return para_3d


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform distribution [-1/n, 1/n]
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layers: uniform distribution [-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SketchHead(nn.Module):
    def __init__(self, d_in, dims, omega_0=30.0, outermost_linear=False):
        super().__init__()
        self.omega_0 = omega_0
        self.outermost_linear = outermost_linear
        dims = [d_in] + dims + [1]
        self.num_layers = len(dims) - 1
        
        self.first_layer = SIRENLayer(dims[0], dims[1], is_first=True, omega_0=omega_0)
        self.hidden_layers = nn.ModuleList()
        for i in range(1, self.num_layers - 1):
            self.hidden_layers.append(SIRENLayer(dims[i], dims[i + 1], is_first=False, omega_0=omega_0))
        
        self.final_layer = nn.Linear(dims[-2], dims[-1])
        
        with torch.no_grad():
            bound = math.sqrt(6 / dims[-2]) / omega_0
            self.final_layer.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        x = self.first_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        if self.outermost_linear:
            x = self.final_layer(x)
        else:
            x = torch.sin(self.omega_0 * self.final_layer(x))
        
        return x

class Generator(nn.Module):
    def __init__(self, num_primitives=4, test=False):
        super(Generator, self).__init__()
        self.num_primitives = num_primitives
        self.test=test
        D_IN = 2
        LATENT_SIZE = 256
        
        for i in range(num_primitives):
            setattr(self, 'sketch_head_'+str(i),
                SketchHead(d_in=D_IN+LATENT_SIZE, dims = [ 512, 512, 512 ]))
    
    def forward(self, sample_point_coordinates, primitive_parameters, code):
        B, N = sample_point_coordinates.shape[:2]  # Batch size, number of testing points
        primitive_parameters = primitive_parameters.transpose(2, 1)  # [B, K, 8]
        B, K, param_dim = primitive_parameters.shape
        
        boxes = primitive_parameters[..., :8]
        transformed_points = transform_points(boxes[..., :4], boxes[..., 4:7], sample_point_coordinates)  # [B, N, K, 3]
        
        sdfs_2d_list = []
        for i in range(self.num_primitives):
            points_2d = transformed_points[..., i, :2]  # [B, N, 2]
            global_feat = code.unsqueeze(1).repeat(1, N, 1)  # [B, N, 256]
            
            points_with_code = add_latent(points_2d, global_feat)  # [B, N, 2+256]
            
            sdf_2d = getattr(self, f'sketch_head_{i}')(points_with_code)
            sdfs_2d_list.append(sdf_2d.reshape(B, N, -1).float())
        
                    
        sdfs_2d = torch.cat(sdfs_2d_list, dim=-1) # [B, N, K]
        box_ext = sdfExtrusion(sdfs_2d, boxes[..., 7], transformed_points).squeeze(-1)
        primitive_sdf = box_ext

        with torch.no_grad():
            weights = F.softmax(-20*primitive_sdf, dim=-1)
        union_sdf = torch.sum(weights*primitive_sdf, dim=-1)

        return union_sdf, sdfs_2d, transformed_points