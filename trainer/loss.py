import torch
import torch.nn as nn
import torch.nn.functional as F

class reconLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, outputs, gt_3d_sdf):
        output_3d_sdf = outputs["output_3d_sdf"]
        
        loss_recon = nn.L1Loss()(output_3d_sdf, gt_3d_sdf)
        loss_recon = self.weights["recon_weight"] * loss_recon
                     
        res = {"L_recon": loss_recon}
        return res