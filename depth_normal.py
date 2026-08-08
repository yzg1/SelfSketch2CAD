import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import argparse
from dataset.DNdataloader import GTSamples
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import glob
import numpy as np

class UNetPredictor(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Encoder
        self.e1 = self._conv_block(input_channels, 64, normalize=False)  # 112x112x64
        self.e2 = self._conv_block(64, 128)   # 56x56x128
        self.e3 = self._conv_block(128, 256)  # 28x28x256
        self.e4 = self._conv_block(256, 512)  # 14x14x512
        self.e5 = self._conv_block(512, 512)  # 7x7x512
        
        # Decoder for normal map
        self.d5_n = self._deconv_block(512, 512)  # from e5
        self.d4_n = self._deconv_block(512 + 512, 512)  # cat d5 + e4
        self.d3_n = self._deconv_block(512 + 256, 256)  # cat d4 + e3
        self.d2_n = self._deconv_block(256 + 128, 128)  # cat d3 + e2
        self.d1_n = self._deconv_block(128 + 64, 64)    # cat d2 + e1
        self.normal_final = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)  # 224→224x3, no upsample
        
        # Decoder for depth map
        self.d5_d = self._deconv_block(512, 512)
        self.d4_d = self._deconv_block(512 + 512, 512)
        self.d3_d = self._deconv_block(512 + 256, 256)
        self.d2_d = self._deconv_block(256 + 128, 128)
        self.d1_d = self._deconv_block(128 + 64, 64)
        self.depth_final = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights()
    
    def _conv_block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.003, eps=1e-5))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.003, eps=1e-5),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Shared encoder
        e1 = self.e1(x)      # 112x112x64
        e2 = self.e2(e1)     # 56x56x128
        e3 = self.e3(e2)     # 28x28x256
        e4 = self.e4(e3)     # 14x14x512
        e5 = self.e5(e4)     # 7x7x512
        
        # Normal map decoder path
        d5_n = self.d5_n(e5)                          # 7→14x512
        d4_n = self.d4_n(torch.cat([d5_n, e4], 1))    # 14→28x512
        d3_n = self.d3_n(torch.cat([d4_n, e3], 1))    # 28→56x256
        d2_n = self.d2_n(torch.cat([d3_n, e2], 1))    # 56→112x128
        d1_n = self.d1_n(torch.cat([d2_n, e1], 1))    # 112→224x64
        normal_map = self.normal_final(d1_n)          # 224x64 →224x3
        normal_map = torch.sigmoid(normal_map)
        
        # Depth map decoder path
        d5_d = self.d5_d(e5)                          # 7→14x512
        d4_d = self.d4_d(torch.cat([d5_d, e4], 1))    # 14→28x512
        d3_d = self.d3_d(torch.cat([d4_d, e3], 1))    # 28→56x256
        d2_d = self.d2_d(torch.cat([d3_d, e2], 1))    # 56→112x128
        d1_d = self.d1_d(torch.cat([d2_d, e1], 1))    # 112→224x64
        depth_map = self.depth_final(d1_d)            # 224x64 →224x3
        depth_map = torch.sigmoid(depth_map)
        
        return depth_map, normal_map

def compute_normal_loss(pred_normal, gt_normal):
    """Combined cosine similarity and L1 loss for normal prediction"""
    # Cosine similarity loss
    cosine_sim = F.cosine_similarity(pred_normal, gt_normal, dim=1)
    cosine_loss = torch.mean(1 - cosine_sim)
    
    # L1 loss
    l1_loss = F.l1_loss(pred_normal, gt_normal)
    
    return cosine_loss + 1.5 * l1_loss

def compute_loss(pred_depth, gt_depth, pred_normal, gt_normal, depth_weight=1.0, normal_weight=1.0):
    loss_depth = F.l1_loss(pred_depth, gt_depth)
    loss_normal = compute_normal_loss(pred_normal, gt_normal)
    total_loss = depth_weight * loss_depth + normal_weight * loss_normal
    return total_loss, loss_depth, loss_normal

def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_depth_loss = 0
    total_normal_loss = 0
    
    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validation")
        for batch in val_pbar:
            sketch_image = batch['sketch_image'].to(device)
            gt_depth = batch['depth_image'].to(device)
            gt_normal = batch['normal_image'].to(device)

            pred_depth, pred_normal = model(sketch_image)
            loss, depth_loss, normal_loss = compute_loss(pred_depth, gt_depth, pred_normal, gt_normal)
            
            total_loss += loss.item()
            total_depth_loss += depth_loss.item()
            total_normal_loss += normal_loss.item()
            
            val_pbar.set_postfix({
                'val_loss': f"{loss.item():.6f}",
                'val_depth': f"{depth_loss.item():.6f}",
                'val_normal': f"{normal_loss.item():.6f}"
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    avg_normal_loss = total_normal_loss / len(dataloader)
    
    return avg_loss, avg_depth_loss, avg_normal_loss

def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_depth_loss = 0
        total_train_normal_loss = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            sketch_image = batch['sketch_image'].to(device)
            gt_depth = batch['depth_image'].to(device)
            gt_normal = batch['normal_image'].to(device)

            pred_depth, pred_normal = model(sketch_image)
            total_loss, loss_depth, loss_normal = compute_loss(pred_depth, gt_depth, pred_normal, gt_normal)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_train_depth_loss += loss_depth.item()
            total_train_normal_loss += loss_normal.item()

            pbar.set_postfix({
                'total_loss': f"{total_loss.item():.6f}",
                'depth_loss': f"{loss_depth.item():.6f}",
                'normal_loss': f"{loss_normal.item():.6f}"
            })
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_depth_loss = total_train_depth_loss / len(train_dataloader)
        avg_train_normal_loss = total_train_normal_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Train Depth: {avg_train_depth_loss:.6f}, Train Normal: {avg_train_normal_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            avg_val_loss, avg_val_depth_loss, avg_val_normal_loss = validate_model(model, val_dataloader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.6f}, Val Depth: {avg_val_depth_loss:.6f}, Val Normal: {avg_val_normal_loss:.6f}")
            
            writer.add_scalar('Loss/Val_Total', avg_val_loss, epoch)
            writer.add_scalar('Loss/Val_Depth', avg_val_depth_loss, epoch)
            writer.add_scalar('Loss/Val_Normal', avg_val_normal_loss, epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(save_dir, "best.pth")
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), best_checkpoint_path)
                else:
                    torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Saved best model with validation loss: {best_val_loss:.6f}")

        writer.add_scalar('Loss/Train_Total', avg_train_loss, epoch)
        writer.add_scalar('Loss/Train_Depth', avg_train_depth_loss, epoch)
        writer.add_scalar('Loss/Train_Normal', avg_train_normal_loss, epoch)

        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(save_dir, f"{epoch+1}.pth")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    writer.close()

def get_image_paths(input_path):
    """Get image paths from input (single file or directory)"""
    supported_extensions = {'.png', '.jpg', '.jpeg'}
    image_paths = []
    
    if os.path.isfile(input_path):
        if os.path.splitext(input_path.lower())[1] in supported_extensions:
            image_paths.append(input_path)
        else:
            print(f"Warning: {input_path} is not a supported image format")
    elif os.path.isdir(input_path):
        for ext in supported_extensions:
            pattern = os.path.join(input_path, f"*{ext}")
            image_paths.extend(glob.glob(pattern))
            pattern = os.path.join(input_path, f"*{ext.upper()}")
            image_paths.extend(glob.glob(pattern))
        image_paths = sorted(list(set(image_paths)))
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist")
    
    return image_paths

def load_and_preprocess_image(image_path, img_size=256):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor

def test_model_with_input(model, input_path, output_dir, device, img_size=256):
    """Test model with flexible input (single image or directory)"""
    model.eval()
    
    image_paths = get_image_paths(input_path)
    
    if not image_paths:
        print(f"No valid images found in {input_path}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    depth_output_dir = os.path.join(output_dir, "depth")
    normal_output_dir = os.path.join(output_dir, "normal")
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(normal_output_dir, exist_ok=True)
    
    with torch.no_grad():
        test_pbar = tqdm(image_paths, desc="Processing images")
        for image_path in test_pbar:
            sketch_tensor = load_and_preprocess_image(image_path, img_size)
            if sketch_tensor is None:
                continue
            
            sketch_image = sketch_tensor.unsqueeze(0).to(device)
            
            pred_depth, pred_normal = model(sketch_image)
            
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            depth_filename = f"{image_name}_depth_pred.png"
            normal_filename = f"{image_name}_normal_pred.png"
            
            save_image(pred_depth, os.path.join(depth_output_dir, depth_filename))
            
            save_image(pred_normal, os.path.join(normal_output_dir, normal_filename))
            
            test_pbar.set_postfix({'processing': image_name})
    
    print(f"Processing complete! Depth maps saved to {depth_output_dir}, Normal maps saved to {normal_output_dir}")

def test_model_with_dataset(model, dataset, output_dir, device):
    model.eval()
    with torch.no_grad():
        test_pbar = tqdm(range(len(dataset)), desc="Testing")
        for i in test_pbar:
            data = dataset[i]
            sketch_image = data['sketch_image'].unsqueeze(0).to(device)
            model_id = data['model_id']
            view_id = data['view_id']

            pred_depth, pred_normal = model(sketch_image)

            depth_dir = os.path.join(output_dir, "depth", model_id)
            normal_dir = os.path.join(output_dir, "normal", model_id)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(normal_dir, exist_ok=True)

            depth_filename = f"{model_id}_{view_id}_depth_pred.png"
            save_image(pred_depth, os.path.join(depth_dir, depth_filename))

            normal_filename = f"{model_id}_{view_id}_normal_pred.png"
            save_image(pred_normal, os.path.join(normal_dir, normal_filename))

            test_pbar.set_postfix({'model': model_id, 'view': view_id})

def main():
    parser = argparse.ArgumentParser(description="Train or test a depth and normal prediction model.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Mode: 'train' or 'test'")
    parser.add_argument("--checkpoint", type=str, default="best.pth",
                        help="Path to checkpoint file for testing (e.g., '200.pth')")
    parser.add_argument("--input", type=str, default=None,
                        help="Input for testing: single image file or directory containing images")
    parser.add_argument("--output", type=str, default="./DN_pre",
                        help="Output directory for predictions")
    parser.add_argument("--gpu", "-g", type=str, default="0",
                        help="GPU ID(s) to use (e.g., '0' or '0,1' for multiple GPUs)")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU(s): {args.gpu}")

    data_source = "****"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetPredictor(input_channels=3)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    if args.mode == "train":
        train_dataset = GTSamples(data_source, partition="train")
        val_dataset = GTSamples(data_source, partition="test")
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

        train_model(model, train_dataloader, val_dataloader, num_epochs=300, optimizer=optimizer, save_dir="DN_Checkpoints", device=device)

    elif args.mode == "test":
        if not args.checkpoint:
            print("Error: --checkpoint is required for testing mode.")
            return

        checkpoint_path = os.path.join("DN_Checkpoints", args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint {checkpoint_path} does not exist.")
            return
        
        checkpoint = torch.load(checkpoint_path)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")

        if args.input:
            test_model_with_input(model, args.input, args.output, device)
        else:
            dataset = GTSamples(data_source, partition="test")
            test_model_with_dataset(model, dataset, args.output, device)

if __name__ == "__main__":
    main()