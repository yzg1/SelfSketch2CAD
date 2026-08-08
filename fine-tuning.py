import os
import torch
import random
import argparse
from tqdm import tqdm
from PIL import Image
import glob
import torchvision.transforms as transforms

from utils import init_seeds
from utils.workspace import load_experiment_specifications
from trainer import FineTunerAE
from dataset import dataloader
import numpy as np


def extract_model_id_from_image_path(image_path):
    """Extract model_id from image path"""
    filename = os.path.basename(image_path)
    model_id = os.path.splitext(filename)[0][:8]
    return model_id


def get_model_ids_and_paths(input_path):
    """Get model IDs and corresponding image paths from input (file or directory)"""
    model_ids = []
    image_paths = []
    
    if os.path.isfile(input_path):
        model_id = extract_model_id_from_image_path(input_path)
        if model_id:
            model_ids.append(model_id)
            image_paths.append(input_path)
    elif os.path.isdir(input_path):
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in image_extensions:
            for image_file in glob.glob(os.path.join(input_path, ext)):
                model_id = extract_model_id_from_image_path(image_file)
                if model_id and model_id not in model_ids:
                    model_ids.append(model_id)
                    image_paths.append(image_file)
    else:
        print(f"Error: Input path {input_path} is neither a file nor a directory")
        return [], []
    
    return model_ids, image_paths


def find_model_indices_in_dataset(dataset, target_model_ids):
    """Find indices of target model IDs in the dataset"""
    indices = []
    available_model_ids = dataset.available_model_ids
    
    for target_id in target_model_ids:
        if target_id in available_model_ids:
            model_idx = available_model_ids.index(target_id)
            indices.append(model_idx)
        else:
            print(f"Warning: Model ID {target_id} not found in dataset")
    
    return indices

def main(args):
    init_seeds()
    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    specs = load_experiment_specifications(experiment_directory)
    sdf_dataset = dataloader.GTSamples(specs["DataSource"], partition='test', use_precomputed_dino=True, dino_features_dir=specs.get("DinoFeaturesDir"))
    
    target_model_ids, image_paths = get_model_ids_and_paths(args.input_path)
    if not target_model_ids:
        print("No valid model IDs found from input. Exiting.")
        return
    
    print(f'Model IDs to fine-tune: {target_model_ids}')
    
    epoches_ft = int(args.epochs)
    specs["experiment_directory"] = experiment_directory
    
    for model_id in target_model_ids:
        print(f'Fine-tuning model: {model_id}')
        
        if model_id not in sdf_dataset.available_model_ids:
            print(f"Error: Model {model_id} not found in dataset. Skipping.")
            continue
            
        data_sample = None
        for idx, (mid, sketch_type) in enumerate(sdf_dataset.sample_pairs):
            if mid == model_id:
                data_sample = sdf_dataset[idx]
                break

        if data_sample is None:
            print(f"Error: Model {model_id} not found in dataset sample_pairs. Skipping.")
            continue

        dino_global_feat = data_sample['dino_global_features'].unsqueeze(0).cuda()
        dino_local_feat = data_sample['dino_local_features'].unsqueeze(0).cuda()
        depth_global_feat = data_sample['depth_global_features'].unsqueeze(0).cuda()
        depth_local_feat = data_sample['depth_local_features'].unsqueeze(0).cuda()
        normal_global_feat = data_sample['normal_global_features'].unsqueeze(0).cuda()
        normal_local_feat = data_sample['normal_local_features'].unsqueeze(0).cuda()
        sdf_data = data_sample['sdf_data'].unsqueeze(0).cuda()
        
        ft_agent = FineTunerAE(specs)
        start_epoch = ft_agent.load_shape_code(dino_global_feat, dino_local_feat, depth_global_feat, depth_local_feat, normal_global_feat, normal_local_feat, args.checkpoint)
        
        # Start fine-tuning
        clock = ft_agent.clock
        pbar = tqdm(range(start_epoch, start_epoch + epoches_ft))
        
        for e in pbar:
            for i in range(40):
                batch_data = {
                    'dino_global_features': dino_global_feat,
                    'dino_local_features': dino_local_feat,
                    'depth_global_feat': depth_global_feat,
                    'depth_local_feat': depth_local_feat,
                    'normal_global_feat': normal_global_feat,
                    'normal_local_feat': normal_local_feat,
                    'sdf_data': sdf_data
                }
                outputs, out_info = ft_agent.train_func(batch_data)
                pbar.set_description(f"EPOCH[{e}][{epoches_ft}] Model: {model_id}")
                clock.tick()
            
            pbar.set_postfix(out_info)
            ft_agent.save_model_if_best_per_shape(model_id)
            clock.tock()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True, help="Directory containing experiment specifications")
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="best", help="Checkpoint to load (default: best)")
    arg_parser.add_argument("--input", dest="input_path", required=True, help="Path to image file or directory containing images")
    arg_parser.add_argument("--epochs", dest="epochs", default=50, help="Number of epochs for fine-tuning")
    arg_parser.add_argument("--gpu", "-g", dest="gpu", default=0, help="GPU device ID to use")
        
    args = arg_parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
    
    main(args)