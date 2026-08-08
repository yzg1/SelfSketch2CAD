import os
import torch
import utils
import argparse
from utils.workspace import load_experiment_specifications
from torchvision.utils import save_image
from torchvision import transforms
from trainer import FineTunerAE
from dataset import dataloader
import numpy as np
import glob
from PIL import Image
from pathlib import Path


def extract_model_info_from_image_path(image_path):
    """Extract model_id from image path"""
    filename = os.path.basename(image_path)
    model_id = os.path.splitext(filename)[0][:8]
    return model_id


def get_image_paths_from_input(input_path):
    """Get image paths and corresponding model IDs from input parameter"""
    image_data = []
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            model_id = extract_model_info_from_image_path(input_path)
            if model_id:
                image_data.append((input_path, model_id))
        else:
            try:
                with open(input_path, 'r') as f:
                    model_ids = [line.strip() for line in f if line.strip()]
                    for model_id in model_ids:
                        image_data.append((None, model_id))
            except:
                print(f"Error: Could not read input file {input_path}")
                return []
                
    elif os.path.isdir(input_path):
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in image_extensions:
            image_files = glob.glob(os.path.join(input_path, ext))
            for image_file in image_files:
                model_id = extract_model_info_from_image_path(image_file)
                if model_id:
                    image_data.append((image_file, model_id))
    else:
        print(f"Error: Input path {input_path} does not exist")
        return []
    
    return image_data


def main(args):
    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    specs = load_experiment_specifications(experiment_directory)
    sdf_dataset = dataloader.GTSamples(specs["DataSource"], partition="test", use_precomputed_dino=True, dino_features_dir=specs.get("DinoFeaturesDir"))	
    
    reconstruction_dir = os.path.join(experiment_directory, "Reconstructions")
    MC_dir = os.path.join(reconstruction_dir, 'MC/')
    CAD_dir = os.path.join(reconstruction_dir, 'CAD/')
    sk_dir = os.path.join(reconstruction_dir, 'sk/')


    for directory in [reconstruction_dir, CAD_dir, sk_dir, MC_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    
    test_items = []
    
    if args.input_path is not None:
        print(f'Running test on models specified by input: {args.input_path}')
        image_data = get_image_paths_from_input(args.input_path)
        
        if not image_data:
            print("No valid image data found from input. Exiting.")
            return
        
        for image_path, model_id in image_data:
            if model_id in sdf_dataset.available_model_ids:
                data = None
                for idx, (mid, sketch_type) in enumerate(sdf_dataset.sample_pairs):
                    if mid == model_id:
                        data = sdf_dataset[idx]
                        break
                
                if data is not None:
                    test_items.append((model_id, data))
                    print(f"Loaded features for {model_id}")
                else:
                    print(f"Warning: Could not load data for model ID {model_id}")
            else:
                print(f"Warning: Model ID {model_id} not found in dataset")
        
    else:
        print("No input path specified. Exiting.")
        return
    
    if not test_items:
        print("No valid test items found. Exiting.")
        return
        
    print(f'Number of test items: {len(test_items)}')
    
    specs["experiment_directory"] = experiment_directory
    ft_agent = FineTunerAE(specs)
    
    ft_agent.load_model_parameters("best")
    
    for model_id, data_sample in test_items:
        print(f"Testing model: {model_id}")

        # Load shape-specific fine-tuned model and optimized shape code
        try:
            epoch, shape_code = ft_agent.load_model_parameters_per_shape(model_id, "best")
            shape_code = shape_code.cuda()
            print(f"Loaded fine-tuned model for {model_id} from epoch {epoch}")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned model for {model_id}: {e}")
            print("Skipping this model...")
            continue
        
        with torch.no_grad():
            shape_3d = ft_agent.decoder(shape_code)
        
        filename_prefix = model_id
            
        mesh_filename = os.path.join(MC_dir, filename_prefix)
        CAD_mesh_filepath = os.path.join(CAD_dir, filename_prefix)
        sk_filepath = os.path.join(sk_dir, filename_prefix)
        
        # Create CAD mesh
        utils.create_CAD_mesh(ft_agent.generator, shape_code.cuda(), shape_3d.cuda(), CAD_mesh_filepath)
        
        # Create mesh using marching cubes
        utils.create_mesh_mc(ft_agent.generator, shape_3d.cuda(), shape_code.cuda(), mesh_filename, N=int(args.grid_sample), threshold=float(args.mc_threshold))
        
        # Draw 2D sketch image
        utils.draw_2d_im_sketch(shape_code.cuda(), ft_agent.generator, sk_filepath)
        
        print(f"Results saved for {filename_prefix}")


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description="test trained model")
	arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True)
	arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="best")
	arg_parser.add_argument("--input", dest="input_path", default=None, help="Path to image file, directory containing images, or text file with model IDs")
	arg_parser.add_argument("--mc_threshold", dest="mc_threshold", default=0, help="marching cube threshold")
	arg_parser.add_argument("--gpu", "-g", dest="gpu", default=0, help="gpu id")
	arg_parser.add_argument("--grid_sample", dest="grid_sample", default=256, help="sample points resolution option")
	args = arg_parser.parse_args()

	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
 
	main(args)