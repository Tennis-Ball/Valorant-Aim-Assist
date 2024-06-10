import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model on a custom dataset.')
    parser.add_argument('--data', type=str, default='../../static/data/data.yaml', help='Path to the data config file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--img-size', type=int, default=416, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--weights', type=str, default='', help='Path to weights file')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0 or 0,1,2,3 or cpu)')
    args = parser.parse_args()
    return args

def train_yolov8(data, epochs, img_size, batch_size, weights, device):
    model = YOLO('yolov8m.pt')  # Initialize the YOLOv8 model with the pretrained weights
    
    # Check if custom weights are provided
    if weights:
        model.load(weights)

    # Train the model
    model.train(data=data, epochs=epochs, imgsz=img_size, batch=batch_size, device=device)

def main():
    args = parse_args()

    # Print current working directory
    print("Current working directory:", os.getcwd())

    # Ensure data config file exists
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"Data config file not found: {args.data}")

    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"Starting training with the following configuration:\n{data_config}")

    train_yolov8(data=args.data, epochs=args.epochs, img_size=args.img_size, batch_size=args.batch_size, weights=args.weights, device=args.device)

if __name__ == '__main__':
    main()
