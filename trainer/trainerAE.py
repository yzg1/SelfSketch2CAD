import torch
from .base import BaseTrainer
from .loss import reconLoss
from model import Encoder, Decoder, Generator
from collections import OrderedDict


class TrainerAE(BaseTrainer):
    def build_net(self):
        self.encoder = Encoder().cuda()
        self.decoder = Decoder(num_primitives=self.specs["NumPrimitives"]).cuda()
        self.generator = Generator(num_primitives=self.specs["NumPrimitives"]).cuda()
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.decoder = torch.nn.DataParallel(self.decoder)
            self.generator = torch.nn.DataParallel(self.generator)

    def set_optimizer(self, lr, betas):
        trainable_params = []
        if hasattr(self.encoder, 'module'):
            encoder_module = self.encoder.module
        else:
            encoder_module = self.encoder
            
        for name, param in encoder_module.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        trainable_params.extend(self.decoder.parameters())
        trainable_params.extend(self.generator.parameters())
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr = lr,
            betas = (betas[0], betas[1])
        )
    
    def set_loss_function(self):
        self.loss_func = reconLoss(self.specs["LossWeightTrain"]).cuda()


    def forward(self, data):
        dino_global_feat = data['dino_global_features'].cuda()
        dino_local_feat = data['dino_local_features'].cuda()
        depth_global_feat = data['depth_global_features'].cuda()
        depth_local_feat = data['depth_local_features'].cuda()
        normal_global_feat = data['normal_global_features'].cuda()
        normal_local_feat = data['normal_local_features'].cuda()
        sdf_data = data['sdf_data'].cuda()
        load_point_batch_size = sdf_data.shape[1]
        point_batch_size = 16*16*16*2
        point_batch_num = int(load_point_batch_size/point_batch_size)
        which_batch = torch.randint(point_batch_num+1, (1,))
        
        if which_batch == point_batch_num:
            xyz = sdf_data[:,-point_batch_size:, :3]
            gt_3d_sdf = sdf_data[:,-point_batch_size:, 3]
        else:
            xyz = sdf_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
            gt_3d_sdf = sdf_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]

        shape_code = self.encoder(dino_global_feat, dino_local_feat, depth_global_feat, depth_local_feat, normal_global_feat, normal_local_feat)
        shape_3d = self.decoder(shape_code)
        output_3d_sdf, sdfs_2d, transformed_points = self.generator(xyz, shape_3d, shape_code)
        h = shape_3d[:, 7, :].unsqueeze(1)
        
        outputs = {
            "output_3d_sdf": output_3d_sdf,
            "sdfs_2d": sdfs_2d,
            "transformed_points": transformed_points,
            "h": h
        }

        loss_dict = self.loss_func(outputs, gt_3d_sdf)
        
        del shape_code, shape_3d, transformed_points, h, xyz, gt_3d_sdf
        torch.cuda.empty_cache()

        return outputs, loss_dict
    
    def train_func(self, data):
        """one step of training"""
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        
        self.optimizer.zero_grad()
        outputs, losses = self.forward(data)
        total_loss = sum(losses.values())
        total_loss.backward()
        
        trainable_params = []
        if hasattr(self.encoder, 'module'):
            encoder_module = self.encoder.module
        else:
            encoder_module = self.encoder
            
        for param in encoder_module.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        trainable_params.extend(self.decoder.parameters())
        trainable_params.extend(self.generator.parameters())
        
        self.optimizer.step()
        
        if self.clock.step % 50 == 0:
            torch.cuda.empty_cache()
        
        self.update_epoch_info(losses)
        if self.clock.step % 10 == 0:
            self.record_to_tb(losses)
        
        loss_info = OrderedDict({k: "{:.6f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_loss.items()})
        out_info = loss_info.copy()
        
        return outputs, out_info