#!/usr/bin/env python3
# rcnn_suite.py  —  ⧈⊕⊗⊟⊢⊚⊡⊣⊤  |  ⧈⊗⊙≡⊟⊢⊚⊡⊣⊤
# PyTorch ≥ 2.2, torchvision ≥ 0.18, scikit-image, opencv-python
import argparse, json, os, sys, time, math, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, ops, transforms, datasets

import numpy as np
import cv2
from PIL import Image

# ===================== Utility =====================
def device_auto() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_rgb(path: str, size_limit: int = 1333) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess image for detection"""
    im_bgr = cv2.imread(path)
    if im_bgr is None: 
        raise FileNotFoundError(f"Could not load image: {path}")
    im = im_bgr[:, :, ::-1]  # BGR to RGB
    h, w = im.shape[:2]
    s = min(1.0, size_limit / max(h, w))
    if s < 1.0:
        im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    return tform(im).unsqueeze(0), im  # [1,3,H,W], RGB uint8

@torch.no_grad()
def draw_boxes(im_rgb: np.ndarray, boxes: np.ndarray, labels: List[str], scores: np.ndarray, 
               thr=0.5, colors: Optional[Dict[str, Tuple[int,int,int]]] = None) -> np.ndarray:
    """Draw bounding boxes with labels and scores"""
    out = im_rgb.copy()
    
    # Generate colors for classes if not provided
    if colors is None:
        unique_labels = list(set(labels))
        color_map = {}
        for i, lbl in enumerate(unique_labels):
            hue = int(180 * i / max(1, len(unique_labels)))
            color_map[lbl] = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0,0])
    else:
        color_map = colors
    
    for b, s, lbl in zip(boxes, scores, labels):
        if s < thr: continue
        x1,y1,x2,y2 = map(int, b)
        color = color_map.get(lbl, (0, 255, 0))
        
        # Draw box
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        
        # Draw label background
        label_text = f"{lbl}:{s:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(out, (x1, y1-text_height-baseline), 
                     (x1+text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(out, label_text, (x1, y1-baseline), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
    return out

def apply_deltas_xyxy(boxes: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply bounding box regression deltas to proposals"""
    # boxes: [N,4] xyxy, deltas: [N,4] (dx,dy,dw,dh) in log-space width/height
    x1,y1,x2,y2 = [boxes[:,i] for i in range(4)]
    w = (x2 - x1).clamp(min=1.0); h = (y2 - y1).clamp(min=1.0)
    cx = x1 + 0.5*w; cy = y1 + 0.5*h
    dx,dy,dw,dh = deltas.unbind(1)
    cx_ = cx + dx*w
    cy_ = cy + dy*h
    w_  = w * torch.exp(dw).clamp(max=1e4)
    h_  = h * torch.exp(dh).clamp(max=1e4)
    x1_ = cx_ - 0.5*w_; y1_ = cy_ - 0.5*h_
    x2_ = cx_ + 0.5*w_; y2_ = cy_ + 0.5*h_
    return torch.stack([x1_,y1_,x2_,y2_], 1)

def nms_per_class(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                  iou_thr=0.5, score_thr=0.05, topk: int = 300) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply NMS independently for each class"""
    keep_boxes, keep_scores, keep_labels = [], [], []
    classes = labels.unique()
    for c in classes.tolist():
        m = labels == c
        b = boxes[m]; s = scores[m]
        if b.numel() == 0: continue
        keep = ops.nms(b, s, iou_thr)
        b = b[keep]; s = s[keep]
        m2 = s >= score_thr
        b, s = b[m2], s[m2]
        l = torch.full((b.shape[0],), c, dtype=torch.long, device=b.device)
        keep_boxes.append(b); keep_scores.append(s); keep_labels.append(l)
    
    if not keep_boxes:
        return (boxes.new_zeros((0,4)), scores.new_zeros((0,)), labels.new_zeros((0,), dtype=torch.long))
    
    B = torch.cat(keep_boxes); S = torch.cat(keep_scores); L = torch.cat(keep_labels)
    
    # Apply top-k filtering
    if B.shape[0] > topk:
        idx = torch.topk(S, k=min(topk, S.shape[0])).indices
        B, S, L = B[idx], S[idx], L[idx]
    
    return B, S, L

# ===================== ⊕ Selective Search (skimage) =====================
def selective_search_boxes(im_rgb: np.ndarray, min_area: int = 400) -> List[List[int]]:
    """Generate region proposals using selective search"""
    try:
        from skimage import segmentation, future, color, measure
    except ImportError:
        warnings.warn("scikit-image not available, using simple grid proposals")
        return grid_proposals(im_rgb, min_area)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = segmentation.slic(color.rgb2lab(im_rgb), n_segments=200, compactness=10, start_label=1)
        rag = future.graph.rag_mean_color(im_rgb, labels)
        merged = future.graph.merge_hierarchical(labels, rag, thresh=40, rag_copy=False)
    
    props = []
    for r in measure.regionprops(merged):
        y0,x0,y1,x1 = r.bbox
        area = (y1-y0)*(x1-x0)
        if area >= min_area and area < im_rgb.shape[0] * im_rgb.shape[1] * 0.8:  # Avoid huge boxes
            props.append([x0,y0,x1,y1])
    
    return props

def grid_proposals(im_rgb: np.ndarray, min_area: int = 400) -> List[List[int]]:
    """Fallback: generate simple grid-based proposals"""
    h, w = im_rgb.shape[:2]
    proposals = []
    
    # Multi-scale grid
    for scale in [0.25, 0.5, 0.75]:
        cell_h = int(h * scale)
        cell_w = int(w * scale)
        
        for i in range(0, h - cell_h + 1, max(1, cell_h // 2)):
            for j in range(0, w - cell_w + 1, max(1, cell_w // 2)):
                x1, y1 = j, i
                x2, y2 = j + cell_w, i + cell_h
                area = (x2 - x1) * (y2 - y1)
                if area >= min_area:
                    proposals.append([x1, y1, x2, y2])
    
    return proposals

# ===================== ⊗ Feature Extractor =====================
class Backbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        if pretrained:
            base = getattr(models, name)(weights="IMAGENET1K_V1")
        else:
            base = getattr(models, name)(weights=None)
        self.out_channels = 2048 if "50" in name else 512
        self.stem = nn.Sequential(*list(base.children())[:-2])  # CxHxW
    
    def forward(self, x): 
        return self.stem(x)

# ===================== ⊢⊚⊡ Detection Head =====================
class RCNNHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 7):
        super().__init__()
        self.pool_size = pool_size
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls = nn.Linear(1024, num_classes + 1)    # + background
        self.box = nn.Linear(1024, (num_classes + 1) * 4)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_feats):
        x = torch.flatten(roi_feats, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.cls(x)
        deltas = self.box(x)
        return logits, deltas

# ===================== ⊟ ROI Align Wrapper =====================
def roi_align_on_feature(fmap: torch.Tensor, boxes_xyxy: torch.Tensor, img_shape_hw: Tuple[int,int],
                         out_size: int = 7) -> torch.Tensor:
    """Extract ROI features using ROI Align"""
    Hf, Wf = fmap.shape[-2:]
    Hi, Wi = img_shape_hw
    scale_y = Hf / float(Hi)
    scale_x = Wf / float(Wi)
    boxes = boxes_xyxy.clone().to(fmap.device)
    boxes[:,[0,2]] *= scale_x
    boxes[:,[1,3]] *= scale_y
    batch_ids = torch.zeros((boxes.shape[0],), dtype=torch.int32, device=fmap.device)
    rois = torch.cat([batch_ids.unsqueeze(1), boxes], dim=1)
    return ops.roi_align(fmap, rois, output_size=out_size, spatial_scale=1.0, aligned=True)

# ===================== R-CNN Pipeline =====================
@dataclass
class RCNNConfig:
    num_classes: int = 20
    backbone: str = "resnet50"
    score_thr: float = 0.05
    nms_iou: float = 0.5
    topk: int = 300
    amp: bool = True
    pretrained: bool = True

class RCNNDetector(nn.Module):
    def __init__(self, cfg: RCNNConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg.backbone, cfg.pretrained)
        self.head = RCNNHead(self.backbone.out_channels, cfg.num_classes)
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.cfg,
        }, path)
        print(f"Model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

    @torch.no_grad()
    def predict_image(self, image_path: str, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        dev = next(self.parameters()).device
        x, im = load_rgb(image_path)
        H, W = im.shape[:2]
        x = x.to(dev)
        
        # ⊕ proposals
        props = selective_search_boxes(im)
        if len(props) == 0:
            print("No proposals generated, using fallback grid")
            props = grid_proposals(im)
            if len(props) == 0:
                return {"boxes": [], "scores": [], "labels": [], "names": []}
        
        props_t = torch.tensor(props, dtype=torch.float32, device=dev)
        
        # ⊗ feature map
        with torch.autocast("cuda" if dev.type=="cuda" else "cpu", enabled=self.cfg.amp):
            fmap = self.backbone(x)
            
            # ⊟ roi align
            rois = roi_align_on_feature(fmap, props_t, (H, W), out_size=7)
            
            # ⊢⊚⊡ head
            logits, deltas_all = self.head(rois)
            probs = F.softmax(logits, dim=1)
            conf, lab = probs.max(1)  # [N]
            
            # Filter out background (class 0)
            fg_mask = lab > 0
            if not fg_mask.any():
                return {"boxes": [], "scores": [], "labels": [], "names": []}
                
            props_t = props_t[fg_mask]
            conf = conf[fg_mask]
            lab = lab[fg_mask]
            deltas_all = deltas_all[fg_mask]
            
            # Pick class-specific deltas
            K = self.cfg.num_classes + 1
            idx = lab * 4 + torch.arange(4, device=dev).unsqueeze(0)
            deltas = torch.gather(deltas_all, 1, idx).view(-1, 4)
            
            pred_boxes = apply_deltas_xyxy(props_t, deltas)
            
            # Clip to image
            pred_boxes[:,0::2] = pred_boxes[:,0::2].clamp(0, W-1)
            pred_boxes[:,1::2] = pred_boxes[:,1::2].clamp(0, H-1)
            
            # ⊣ per-class NMS
            B, S, L = nms_per_class(pred_boxes, conf, lab, self.cfg.nms_iou, 
                                  self.cfg.score_thr, self.cfg.topk)
        
        names = [str(int(c.item())) if not class_names else class_names[int(c)-1] for c in L]
        return {
            "boxes": B.cpu().numpy().tolist(),
            "scores": S.cpu().numpy().tolist(),
            "labels": L.cpu().numpy().tolist(),
            "names": names
        }

# ===================== Faster R-CNN Path (TorchVision) =====================
@dataclass
class FasterConfig:
    num_classes: int = 91
    score_thr: float = 0.05
    amp: bool = True

class FasterDetector:
    def __init__(self, cfg: FasterConfig):
        self.cfg = cfg
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
        
        # Replace head for custom classes
        if cfg.num_classes != 91:
            in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                in_feats, cfg.num_classes)
        
        self.model.eval()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.cfg,
        }, path)
        print(f"Model saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

    @torch.no_grad()
    def predict_image(self, image_path: str, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        dev = device_auto()
        self.model.to(dev)
        x, im = load_rgb(image_path)
        
        with torch.autocast("cuda" if dev.type=="cuda" else "cpu", enabled=self.cfg.amp):
            out = self.model([x.squeeze(0).to(dev)])[0]
        
        scores = out["scores"].detach().cpu().numpy()
        m = scores >= self.cfg.score_thr
        boxes = out["boxes"].detach().cpu().numpy()[m]
        labels = out["labels"].detach().cpu().numpy()[m]
        scores = scores[m]
        
        names = [str(int(c)) if not class_names else class_names[int(c)-1] for c in labels]
        return {
            "boxes": boxes.tolist(), 
            "scores": scores.tolist(), 
            "labels": labels.tolist(), 
            "names": names
        }

    def train_voc(self, voc_root: str, year="2007", image_set="trainval", epochs=6, 
                  batch_size=2, lr=5e-4, workers=2, output_dir="checkpoints"):
        """Train on VOC dataset"""
        dev = device_auto()
        self.model.to(dev).train()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # VOC dataset with proper transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        ds = datasets.VOCDetection(
            voc_root, year=year, image_set=image_set, download=False, transforms=transform)
        
        def collate(batch):
            images, targets = [], []
            for img, ann in batch:
                images.append(img)
                objs = ann["annotation"]["object"]
                if isinstance(objs, dict): 
                    objs = [objs]
                
                boxes, labels = [], []
                for ob in objs:
                    b = ob["bndbox"]
                    # Convert to float and ensure proper order
                    x1 = float(b["xmin"]); y1 = float(b["ymin"])
                    x2 = float(b["xmax"]); y2 = float(b["ymax"])
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)  # Single class for simplicity
                
                if boxes:  # Only add if there are valid boxes
                    t = {
                        "boxes": torch.tensor(boxes, dtype=torch.float32),
                        "labels": torch.tensor(labels, dtype=torch.int64)
                    }
                    targets.append(t)
                else:
                    # Add dummy target for images without objects
                    t = {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.int64)
                    }
                    targets.append(t)
            
            return images, targets

        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                       num_workers=workers, collate_fn=collate, pin_memory=True)
        
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
        
        scaler = torch.cuda.amp.GradScaler(enabled=(dev.type=="cuda" and self.cfg.amp))
        
        print(f"Training Faster R-CNN on VOC for {epochs} epochs...")
        
        for ep in range(1, epochs+1):
            t0 = time.time()
            losses = []
            
            for i, (images, targets) in enumerate(dl):
                images = [im.to(dev) for im in images]
                targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
                
                opt.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=(dev.type=="cuda" and self.cfg.amp)):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())
                
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                losses.append(loss.item())
                
                if i % 50 == 0:
                    print(f"Epoch {ep}, Batch {i}, Loss: {loss.item():.4f}")
            
            scheduler.step()
            
            avg_loss = sum(losses) / len(losses)
            epoch_time = time.time() - t0
            
            print(f"Epoch {ep}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if ep % 2 == 0:
                checkpoint_path = os.path.join(output_dir, f"faster_rcnn_epoch_{ep}.pth")
                self.save_checkpoint(checkpoint_path)
        
        self.model.eval()
        print("Training completed!")

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser(description="R-CNN / Faster R-CNN detection suite")
    ap.add_argument("--mode", choices=["rcnn", "faster"], default="faster")
    ap.add_argument("--image", type=str, help="Image path for inference")
    ap.add_argument("--out", type=str, help="Output image with detection boxes")
    ap.add_argument("--voc_train", type=str, help="VOC root path to train Faster R-CNN")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--classes", type=str, help="JSON file with class names list")
    ap.add_argument("--checkpoint", type=str, help="Model checkpoint to load")
    ap.add_argument("--score_thr", type=float, default=0.5, help="Score threshold for visualization")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device to use")
    
    args = ap.parse_args()

    # Set device
    if args.device == "auto":
        dev = device_auto()
    else:
        dev = torch.device(args.device)
    print(f"Using device: {dev}")

    # Load class names
    class_names = None
    if args.classes:
        with open(args.classes) as f: 
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes")

    if args.mode == "rcnn":
        cfg = RCNNConfig()
        model = RCNNDetector(cfg).to(dev).eval()
        
        # Load checkpoint if provided
        if args.checkpoint:
            model.load_checkpoint(args.checkpoint)
        
        if not args.image:
            print("Provide --image for inference")
            return
        
        print(f"Processing image: {args.image}")
        out = model.predict_image(args.image, class_names)
        print(json.dumps(out, indent=2))
        
        if args.out:
            x, im = load_rgb(args.image)
            vis = draw_boxes(im, np.array(out["boxes"]), out["names"], 
                           np.array(out["scores"]), thr=args.score_thr)
            cv2.imwrite(args.out, vis[:, :, ::-1])
            print(f"Output saved to: {args.out}")
    
    else:  # Faster R-CNN
        cfg = FasterConfig()
        fd = FasterDetector(cfg)
        
        # Load checkpoint if provided
        if args.checkpoint:
            fd.load_checkpoint(args.checkpoint)
        
        if args.voc_train:
            print(f"Training on VOC dataset: {args.voc_train}")
            fd.train_voc(args.voc_train, epochs=args.epochs)
        
        if args.image:
            print(f"Processing image: {args.image}")
            out = fd.predict_image(args.image, class_names)
            print(json.dumps(out, indent=2))
            
            if args.out:
                x, im = load_rgb(args.image)
                vis = draw_boxes(im, np.array(out["boxes"]), out["names"], 
                               np.array(out["scores"]), thr=args.score_thr)
                cv2.imwrite(args.out, vis[:, :, ::-1])
                print(f"Output saved to: {args.out}")

if __name__ == "__main__":
    main()
