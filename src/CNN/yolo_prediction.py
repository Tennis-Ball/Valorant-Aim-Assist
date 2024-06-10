import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run YOLOv9 model on an input image and output the result with bounding boxes.')
    parser.add_argument('--image', type=str, default='../../static/images/test.png', help='Path to the input image')
    parser.add_argument('--weights', type=str, default='./runs/detect/train20/weights/best.pt', help='Path to the trained model weights')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save the output image with bounding boxes')
    args = parser.parse_args()
    return args

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return img

def plot_boxes(img, boxes, output_path):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    plt.show()

def main():
    args = parse_args()
    
    # Load YOLO model
    model = YOLO(args.weights)
    
    # Load image
    img = load_image(args.image)
    
    # Convert image to tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    
    # Perform inference
    results = model.predict(source=args.image)
    
    # Get bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Plot boxes on the image
    plot_boxes(img, boxes, args.output)

if __name__ == '__main__':
    main()
