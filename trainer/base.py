import os
import torch
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.workspace import get_model_params_dir,get_model_params_dir_shapename


class BaseTrainer(object):
    def __init__(self, specs):
        self.specs = specs
        self.experiment_directory = specs['experiment_directory']
        self.log_dir = os.path.join(specs['experiment_directory'], 'log/')
        
        self.best_loss = float('inf')
        self.epoch_loss = None
        self.epoch_acc = None
        self.clock = TrainClock()

        self.build_net()
        self.set_loss_function()
        self.set_optimizer(lr=specs["LearningRate"],betas=specs["betas"])
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))

    @abstractmethod
    def build_net(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_optimizer(self):
        raise NotImplementedError
    
    def load_shape_code(self):
        """load shape code for finetuning"""
        pass
    
    def set_loss_function(self):
        """set loss function used in training"""
        pass

    @abstractmethod
    def forward(self, data):
        """forward logic for your network"""
        """should return network outputs, losses(dict)"""
        raise NotImplementedError
    
    @abstractmethod
    def train_func(self, data):
        """one step of training"""
        raise NotImplementedError
    
    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

    def record_to_tb(self, loss_dict):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        tb = self.train_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)
            
    def update_epoch_info(self, loss_dict):
        if self.clock.minibatch == 0:
            self.epoch_loss = None
            
        self.epoch_loss = {key: self.epoch_loss.get(key, 0) + loss_dict[key] for key in loss_dict} \
                                                                if self.epoch_loss else loss_dict
                                                                
    def save_model_parameters(self, filename):
        model_params_dir = get_model_params_dir(self.experiment_directory)
        torch.save(
            {"epoch": self.clock.epoch,
            "encoder_state_dict": self.encoder.module.state_dict() if isinstance(self.encoder, torch.nn.DataParallel) else self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.module.state_dict() if isinstance(self.decoder, torch.nn.DataParallel) else self.decoder.state_dict(),
            "generator_state_dict": self.generator.module.state_dict() if isinstance(self.generator, torch.nn.DataParallel) else self.generator.state_dict(),
            "opt_state_dict": self.optimizer.state_dict()},
            os.path.join(model_params_dir, filename),
        )
        
    def save_model_if_best(self):
        epoch_loss_value = sum(self.epoch_loss.values()).item()/(self.clock.minibatch+1)
        if epoch_loss_value < self.best_loss:
            model_params_dir = get_model_params_dir(self.experiment_directory)
            torch.save(
                {"epoch": self.clock.epoch,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "opt_state_dict": self.optimizer.state_dict()},
                os.path.join(model_params_dir, 'best.pth'),
            )
            self.best_loss = epoch_loss_value
            
    def save_model_parameters_per_shape(self, shapename, filename):

        model_params_dir = get_model_params_dir(self.experiment_directory)
        model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename)

        torch.save(
            {"epoch": self.clock.epoch,
            "shape_code_state_dict": self.shape_code,
            "decoder_state_dict": self.decoder.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "opt_state_dict": self.optimizer.state_dict()}, 
            os.path.join(model_params_dir, filename)
        )

    def load_model_parameters(self, checkpoint, opt=False):

        filename = os.path.join(
            self.experiment_directory, "ModelParameters", checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)
        
        def strip_module_prefix(state_dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                new_state_dict[new_key] = value
            return new_state_dict

        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder.module.load_state_dict(data["encoder_state_dict"])
        else:
            self.encoder.load_state_dict(strip_module_prefix(data["encoder_state_dict"]))
            
        if isinstance(self.decoder, torch.nn.DataParallel):
            self.decoder.module.load_state_dict(data["decoder_state_dict"])
        else:
            self.decoder.load_state_dict(strip_module_prefix(data["decoder_state_dict"]))
        
        if isinstance(self.generator, torch.nn.DataParallel):
            self.generator.module.load_state_dict(data["generator_state_dict"])
        else:
            self.generator.load_state_dict(strip_module_prefix(data["generator_state_dict"]))
        
        if opt:
            self.optimizer.load_state_dict(data["opt_state_dict"])
        return data["epoch"]
    
    def load_model_parameters_per_shape(self, shapename, checkpoint):

        filename = os.path.join(
            self.experiment_directory, "ModelParameters", shapename, checkpoint + ".pth"
        )

        if not os.path.isfile(filename):
            raise Exception('model state dict "{}" does not exist'.format(filename))

        data = torch.load(filename)
        
        def strip_module_prefix(state_dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                new_state_dict[new_key] = value
            return new_state_dict

        self.decoder.load_state_dict(strip_module_prefix(data["decoder_state_dict"]))
        self.generator.load_state_dict(strip_module_prefix(data["generator_state_dict"]))
        self.optimizer.load_state_dict(data["opt_state_dict"])

        shape_code = data["shape_code_state_dict"]
        
        return data["epoch"], shape_code


class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']
