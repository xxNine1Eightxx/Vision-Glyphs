#!/usr/bin/env python3
"""
Demo script for R-CNN Suite
"""
import json
import os
from pathlib import Path
from rcnn_suite import *

def demo_inference():
    """Demo inference on sample image"""
    print("=== R-CNN Suite Demo ===")
    
    # Create sample class names (VOC classes)
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
                   "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
                   "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"]
    
    # Use Faster R-CNN by default (pretrained on COCO)
    cfg = FasterConfig(num_classes=91)  # COCO has 91 classes
    detector = FasterDetector(cfg)
    
    # Try to find a sample image
    sample_images = ["sample.jpg", "demo.jpg", "test.jpg"]
    sample_image = None
    for img in sample_images:
        if os.path.exists(img):
            sample_image = img
            break
    
    if not sample_image:
        print("No sample image found. Please provide an image with --image argument")
        print("You can also run: python rcnn_suite.py --mode faster --image your_image.jpg --out output.jpg")
        return
    
    print(f"Using sample image: {sample_image}")
    
    # Run inference
    results = detector.predict_image(sample_image, class_names)
    
    # Print results
    print("\nDetection results:")
    for box, score, label in zip(results["boxes"], results["scores"], results["names"]):
        if score > 0.5:
            print(f"  {label}: {score:.3f} at {[int(x) for x in box]}")
    
    # Visualize
    output_image = "demo_output.jpg"
    x, im = load_rgb(sample_image)
    vis = draw_boxes(im, np.array(results["boxes"]), results["names"], 
                   np.array(results["scores"]), thr=0.5)
    cv2.imwrite(output_image, vis[:, :, ::-1])
    print(f"\nVisualization saved to {output_image}")

def demo_rcnn():
    """Demo R-CNN with selective search"""
    print("\n=== R-CNN Demo ===")
    
    class_names = ["object"] * 20  # Simple class names
    
    cfg = RCNNConfig(num_classes=20)
    model = RCNNDetector(cfg).to(device_auto()).eval()
    
    sample_images = ["sample.jpg", "demo.jpg", "test.jpg"]
    sample_image = None
    for img in sample_images:
        if os.path.exists(img):
            sample_image = img
            break
    
    if sample_image:
        print(f"Testing R-CNN on: {sample_image}")
        out = model.predict_image(sample_image, class_names)
        
        print("R-CNN Results:")
        for box, score, label in zip(out["boxes"], out["scores"], out["names"]):
            if score > 0.5:
                print(f"  {label}: {score:.3f} at {[int(x) for x in box]}")
        
        # Save visualization
        output_image = "rcnn_output.jpg"
        x, im = load_rgb(sample_image)
        vis = draw_boxes(im, np.array(out["boxes"]), out["names"], 
                       np.array(out["scores"]), thr=0.5)
        cv2.imwrite(output_image, vis[:, :, ::-1])
        print(f"R-CNN output saved to {output_image}")

def create_sample_image():
    """Create a simple sample image if none exists"""
    sample_path = "sample.jpg"
    if not os.path.exists(sample_path):
        print("Creating sample image...")
        # Create a simple image with colored rectangles
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Add some colored rectangles
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # Red
        cv2.rectangle(img, (200, 100), (300, 200), (0, 255, 0), -1)  # Green  
        cv2.rectangle(img, (350, 150), (450, 250), (0, 0, 255), -1)  # Blue
        cv2.rectangle(img, (100, 250), (250, 350), (255, 255, 0), -1)  # Cyan
        
        cv2.imwrite(sample_path, img)
        print(f"Sample image created: {sample_path}")
    return sample_path

if __name__ == "__main__":
    # Create sample image if none exists
    create_sample_image()
    
    # Run demos
    demo_inference()
    demo_rcnn()
    
    print("\n=== Demo Complete ===")
    print("You can also run the full CLI:")
    print("  python rcnn_suite.py --mode faster --image sample.jpg --out result.jpg")
    print("  python rcnn_suite.py --mode rcnn --image sample.jpg --out rcnn_result.jpg")
