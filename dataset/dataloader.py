import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import h5py


class GTSamples(Dataset):
    def __init__(self, data_source, partition="train", img_size=256,
                 use_precomputed_dino=False, dino_features_dir=None):
        super().__init__()
        self.data_source = Path(data_source)
        self.partition = partition
        self.img_size = img_size
        self.use_precomputed_dino = use_precomputed_dino
        self.dino_features_dir = Path(dino_features_dir) if dino_features_dir else None
        
        self.sketch_types = ["edgemap", "contours", "PhotoSketch"]
        self.depth_type = "depthmap"
        self.normal_type = "normalmap"
        self.sketch_roots = {st: self.data_source / "renderingimg" / st for st in self.sketch_types}
        self.depth_root = self.data_source / "renderingimg" / self.depth_type
        self.normal_root = self.data_source / "renderingimg" / self.normal_type
        self._load_hdf5_data()
        
        # Only need transform when NOT using precomputed features
        if not self.use_precomputed_dino:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        assert partition in ["train", "test"]

        # If using precomputed .npy features, pre-build lookup table
        if self.use_precomputed_dino:
            if self.dino_features_dir is None:
                raise ValueError("dino_features_dir must be provided when use_precomputed_dino=True")
            self._build_npy_feature_lookup()

        self._get_model_data()
        self._generate_sample_pairs()

    def _load_hdf5_data(self):
        if self.partition == "test":
            hdf5_file = self.data_source / 'sdf_test.hdf5'
            name_file = self.data_source / 'test_names.npz'
        else:
            hdf5_file = self.data_source / 'sdf_train.hdf5'
            name_file = self.data_source / 'train_names.npz'

        npz_data = np.load(name_file)
        self.hdf5_names = npz_data['train_names' if self.partition == 'train' else 'test_names']
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.hdf5_points = torch.from_numpy(self.hdf5_file['points'][:]).float()

        self.id_to_hdf5_idx = {}
        for idx, full_name in enumerate(self.hdf5_names):
            name_str = full_name.decode('utf-8') if isinstance(full_name, bytes) else str(full_name)
            simplified_id = name_str[:8]
            self.id_to_hdf5_idx.setdefault(simplified_id, []).append(idx)

        print(f"Loaded {len(self.hdf5_names)} models from HDF5 ({self.partition})")

    def _build_npy_feature_lookup(self):
        self.npy_lookup = {}
        
        for sketch_type in self.sketch_types:
            feat_dir = self.dino_features_dir / sketch_type
            if not feat_dir.exists():
                print(f"Warning: {feat_dir} not exists, skip {sketch_type}")
                continue
            
            lookup = {}
            for model_dir in feat_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_id = model_dir.name
                
                for global_file in model_dir.glob("global_*.npy"):
                    view_id = global_file.stem.split("_", 1)[1]
                    local_file = model_dir / f"local_{view_id}.npy"
                    if local_file.exists():
                        lookup[(model_id, view_id)] = {
                            'global': str(global_file),
                            'local': str(local_file)
                        }
            
            self.npy_lookup[sketch_type] = lookup
            print(f"  {sketch_type}: {len(lookup)} views loaded from .npy")
        
        depth_feat_dir = self.dino_features_dir / self.depth_type
        if depth_feat_dir.exists():
            depth_lookup = {}
            for model_dir in depth_feat_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_id = model_dir.name
                
                for global_file in model_dir.glob("global_*_depth.npy"):
                    view_id = global_file.stem.split("_")[1]
                    local_file = model_dir / f"local_{view_id}_depth.npy"
                    if local_file.exists():
                        depth_lookup[(model_id, view_id)] = {
                            'global': str(global_file),
                            'local': str(local_file)
                        }
            
            self.npy_lookup[self.depth_type] = depth_lookup
            print(f"  {self.depth_type}: {len(depth_lookup)} views loaded from .npy")
        else:
            print(f"Warning: {depth_feat_dir} not exists, skip depth features")
            self.npy_lookup[self.depth_type] = {}
        
        normal_feat_dir = self.dino_features_dir / self.normal_type
        if normal_feat_dir.exists():
            normal_lookup = {}
            for model_dir in normal_feat_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_id = model_dir.name
                
                for global_file in model_dir.glob("global_*_normal.npy"):
                    view_id = global_file.stem.split("_")[1]
                    local_file = model_dir / f"local_{view_id}_normal.npy"
                    if local_file.exists():
                        normal_lookup[(model_id, view_id)] = {
                            'global': str(global_file),
                            'local': str(local_file)
                        }
            
            self.npy_lookup[self.normal_type] = normal_lookup
            print(f"  {self.normal_type}: {len(normal_lookup)} views loaded from .npy")
        else:
            print(f"Warning: {normal_feat_dir} not exists, skip normal features")
            self.npy_lookup[self.normal_type] = {}

    def _get_model_data(self):
        ref_root = self.sketch_roots['edgemap']
        self.model_data = {}

        for model_dir in sorted(ref_root.iterdir()):
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            if model_id not in self.id_to_hdf5_idx:
                continue

            available_views = set()
            
            if self.use_precomputed_dino:
                # Use .npy files as source of truth
                for sketch_type in self.sketch_types:
                    if sketch_type in self.npy_lookup:
                        for (mid, vid) in self.npy_lookup[sketch_type]:
                            if mid == model_id:
                                available_views.add(vid)
            else:
                # Use original PNG files
                for png_file in model_dir.glob("*.png"):
                    fname = png_file.name
                    if "_view_" in fname:
                        vid = fname.split("_view_")[-1].rsplit(".", 1)[0]
                        available_views.add(vid)

            if available_views:
                self.model_data[model_id] = sorted(list(available_views))

        self.available_model_ids = list(self.model_data.keys())
        total_views = sum(len(vs) for vs in self.model_data.values())
        print(f"Found {len(self.available_model_ids)} models, {total_views} views for {self.partition}")

    def _generate_sample_pairs(self):
        self.sample_pairs = []
        for model_id in self.available_model_ids:
            for sketch_type in self.sketch_types:
                # If using precomputed features, skip if no features for this type
                if self.use_precomputed_dino and sketch_type not in self.npy_lookup:
                    continue
                if self.use_precomputed_dino:
                    available_type_views = {
                        vid for (mid, vid) in self.npy_lookup[sketch_type]
                        if mid == model_id
                    }
                    if not available_type_views:
                        continue
                self.sample_pairs.append((model_id, sketch_type))
        
        print(f"Generated {len(self.sample_pairs)} sample pairs (model × sketch_type)")

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        model_id, sketch_type = self.sample_pairs[idx]
        available_views = self.model_data[model_id]
        
        # Randomly pick a view that actually exists for this sketch_type
        if self.use_precomputed_dino:
            valid_views = [
                vid for vid in available_views
                if (model_id, vid) in self.npy_lookup[sketch_type]
            ]
        else:
            valid_views = available_views
        
        view_id = random.choice(valid_views)
        hdf5_idx = random.choice(self.id_to_hdf5_idx[model_id])

        result = {
            'model_id': model_id,
            'view_id': view_id,
            'sketch_type': sketch_type,
            'hdf5_idx': hdf5_idx,
            'full_model_name': str(self.hdf5_names[hdf5_idx]),
        }

        if self.use_precomputed_dino:
            paths = self.npy_lookup[sketch_type][(model_id, view_id)]
            global_feat = np.load(paths['global'])   # (1536,)
            local_feat  = np.load(paths['local'])    # (256, 1536)
            
            result['dino_global_features'] = torch.from_numpy(global_feat.astype(np.float32))
            result['dino_local_features']  = torch.from_numpy(local_feat.astype(np.float32))

            depth_paths = self.npy_lookup[self.depth_type][(model_id, view_id)]
            depth_global = np.load(depth_paths['global'])
            depth_local = np.load(depth_paths['local'])
            result['depth_global_features'] = torch.from_numpy(depth_global.astype(np.float32))
            result['depth_local_features'] = torch.from_numpy(depth_local.astype(np.float32))

            normal_paths = self.npy_lookup[self.normal_type][(model_id, view_id)]
            normal_global = np.load(normal_paths['global'])
            normal_local = np.load(normal_paths['local'])
            result['normal_global_features'] = torch.from_numpy(normal_global.astype(np.float32))
            result['normal_local_features'] = torch.from_numpy(normal_local.astype(np.float32))
        else:
            sketch_path = self.sketch_roots[sketch_type] / model_id / f"{model_id}_view_{view_id}.png"
            img = np.array(Image.open(sketch_path).convert("RGB"))
            result['sketch_image'] = self.transform(img)

        result['sdf_data'] = self.hdf5_points[hdf5_idx]
        return result

    def __del__(self):
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()