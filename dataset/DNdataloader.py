import os
import glob
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class GTSamples(Dataset):
    """Dataset for training and testing with sketch images, depth maps and normal maps"""
    def __init__(self, data_source, partition="train", img_size=256):
        super().__init__()
        self.data_source = data_source
        self.partition = partition
        self.img_size = img_size
        
        self.sketch_types = ["edgemap", "contours", "PhotoSketch"]
        self.sketch_roots = {
            sketch_type: os.path.join(self.data_source, f"renderingimg/{sketch_type}")
            for sketch_type in self.sketch_types
        }
        self.depthmap_root = os.path.join(self.data_source, "renderingimg/depthmap")
        self.normalmap_root = os.path.join(self.data_source, "renderingimg/normalmap")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        assert self.partition in ["train", "test"], "Partition must be 'train' or 'test'"

        # Load model IDs from train.txt or test.txt
        txt_file = os.path.join(self.data_source, f"{partition}.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Partition file {txt_file} does not exist")
        
        with open(txt_file, 'r') as f:
            self.model_ids = [line.strip() for line in f.readlines() if line.strip()]

        self._get_model_data()
        self._generate_sample_pairs()

    def _generate_sample_pairs(self):
        """Generate all combinations of models and sketch types"""
        self.sample_pairs = []
        for model_id in self.available_model_ids:
            for sketch_type in self.sketch_types:
                self.sample_pairs.append((model_id, sketch_type))

    def _get_model_data(self):
        """Get available model IDs and their corresponding views"""
        edgemap_root = self.sketch_roots['edgemap_bold']
        if not os.path.exists(edgemap_root):
            raise FileNotFoundError(f"Edge map root {edgemap_root} does not exist")
        
        self.model_data = {}
        
        for model_id in self.model_ids:
            model_path = os.path.join(edgemap_root, model_id)
            if not os.path.isdir(model_path):
                print(f"Warning: Model directory {model_path} does not exist, skipping.")
                continue
            
            png_files = glob.glob(os.path.join(model_path, "*.png"))
            if not png_files:
                continue
            
            available_views = []
            for png_file in png_files:
                view_name = os.path.basename(png_file)
                view_id = view_name.split('_view_')[-1].split('.')[0]
                
                depth_path = os.path.join(self.depthmap_root, model_id, 
                                         f"{model_id}_view_{view_id}_depth.png")
                normal_path = os.path.join(self.normalmap_root, model_id, 
                                          f"{model_id}_view_{view_id}_normal.png")
                
                if all(os.path.exists(p) for p in [png_file, depth_path, normal_path]):
                    available_views.append(view_id)
            
            if available_views:
                self.model_data[model_id] = available_views
        
        self.available_model_ids = list(self.model_data.keys())

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        """Get a training sample: sketch image, depth map, and normal map"""
        model_id, sketch_type = self.sample_pairs[idx]
        available_views = self.model_data[model_id]
        
        # Randomly select a view for this model
        view_id = random.choice(available_views)
        
        sketch_root = self.sketch_roots[sketch_type]
        sketch_path = os.path.join(sketch_root, model_id, f"{model_id}_view_{view_id}.png")
        depth_path = os.path.join(self.depthmap_root, model_id, 
                                 f"{model_id}_view_{view_id}_depth.png")
        normal_path = os.path.join(self.normalmap_root, model_id, 
                                  f"{model_id}_view_{view_id}_normal.png")
    
        sketch_img = Image.open(Path(sketch_path)).convert("RGB")
        depth_img = Image.open(Path(depth_path)).convert("RGB")
        normal_img = Image.open(Path(normal_path)).convert("RGB")
    
        # Apply transforms
        sketch_image = self.transform(sketch_img)
        depth_image = self.transform(depth_img)
        normal_image = self.transform(normal_img)
    
        return {
            'sketch_image': sketch_image,
            'depth_image': depth_image,
            'normal_image': normal_image,
            'model_id': model_id,
            'view_id': view_id,
        }


if __name__ == "__main__":
    train_dataset = GTSamples(data_source='****', 
                             partition='train')
    
    print(f"Dataset length: {len(train_dataset)}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    for batch in train_loader:
        print(f"Sketch shape: {batch['sketch_image'].shape}")
        print(f"Depth shape: {batch['depth_image'].shape}")
        print(f"Normal shape: {batch['normal_image'].shape}")
        break