import os
import torch
import argparse
import torch.utils.data as data_utils
from tqdm import tqdm

from utils import init_seeds
from utils.workspace import load_experiment_specifications
from dataset import dataloader
from trainer import TrainerAE


def main(args):
    init_seeds()
    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    specs = load_experiment_specifications(experiment_directory)
    
    sdf_dataset = dataloader.GTSamples(specs["DataSource"], partition="train", use_precomputed_dino=True, dino_features_dir=specs.get("DinoFeaturesDir"))
    data_loader = data_utils.DataLoader(sdf_dataset, batch_size=specs["BatchSize"], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
    specs["experiment_directory"] = experiment_directory
    tr_agent = TrainerAE(specs)
    
    clock = tr_agent.clock
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        try:
            # Load model parameters and get the epoch number
            start_epoch = tr_agent.load_model_parameters(args.resume_checkpoint, opt=True)
            clock.epoch = start_epoch + 1  # Start from next epoch
            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
            print(f"Resuming training from epoch {clock.epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
            clock.epoch = 1
    
    # Calculate target epochs
    target_epochs = args.target_epochs if args.target_epochs else specs["NumEpochs"]
    epochs_to_train = target_epochs - start_epoch
    
    print(f"Training from epoch {clock.epoch} to epoch {target_epochs}")
    print(f"Total epochs to train: {epochs_to_train}")
    
    # Start training
    for epoch in range(clock.epoch, target_epochs + 1):
        # Begin iteration
        pbar = tqdm(data_loader)
        for b, data in enumerate(pbar):
            # Train step
            outputs, out_info = tr_agent.train_func(data)
            pbar.set_description("EPOCH[{}][{}]".format(epoch, b))
            pbar.set_postfix(out_info)
            clock.tick()
        
        total_epoch_loss = sum(tr_agent.epoch_loss.values()).item() / len(data_loader)
        print(f"Epoch {epoch} average total loss: {total_epoch_loss:.6f}")
        
        # Save model at specified frequency
        if epoch % specs["SaveFrequency"] == 0:
            tr_agent.save_model_parameters(f"{epoch}.pth")
            print(f"Saved checkpoint: {epoch}.pth")
        
        # Save best model
        tr_agent.save_model_if_best()
        clock.tock()
    
    # Save final model
    if target_epochs % specs["SaveFrequency"] != 0:
        tr_agent.save_model_parameters(f"{target_epochs}.pth")
        print(f"Saved final checkpoint: {target_epochs}.pth")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True, help="Name of the experiment directory")
    arg_parser.add_argument("--gpu", "-g", dest="gpu", default="0", help="GPU device to use")
    arg_parser.add_argument("--resume", "-r", dest="resume_checkpoint", default=None, help="Checkpoint filename to resume from (e.g., '500')")
    arg_parser.add_argument("--target-epochs", "-t", dest="target_epochs", type=int, default=None, help="Target number of epochs to train to")
    args = arg_parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    main(args)