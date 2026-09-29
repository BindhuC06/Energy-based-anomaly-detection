import os
import argparse
import re
from PIL import Image, ImageDraw, ImageFont
import torch

from ebm_ablation import GraphAblator
from SGG import SceneGraphBuilder

def get_color(is_anomalous: bool):
    return (255, 50, 50) if is_anomalous else (50, 220, 50) # Red for anomalous, Green for normal

IMAGE_DIR = "test"

def process_single_image(image_path, ablator, sgg_builder, out_dir):
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        return
        
    base_name = os.path.basename(image_path)
    name_no_ext, ext = os.path.splitext(base_name)
        
    print(f"\n[Visualization] Processing: {image_path}")
    results = ablator.run_ablation(image_path)
    
    if results is None:
        print("[Visualization] Ablation returned None (pipeline error, missing model, or empty graph). Skipping.")
        return
    
    if not isinstance(results, dict):
        print("[Visualization] Ablation returned unexpected type. Skipping.")
        return
        
    scene_json = results["scene_json"]
    objects = scene_json.get("objects", [])
    relationships = scene_json.get("relationships", [])
    
    if not objects:
        print("[Visualization] No objects detected in scene graph. Nothing to draw.")
        return
    
    # 2. Determine anomalous nodes and relationships
    anomalous_node_indices = set()
    anomalous_edge_indices = set()
    
    if results.get("is_anomalous", False):
        node_results = results.get("node_results", [])
        edge_results = results.get("edge_results", [])
        
        # Node threshold: delta_e > 0.3. If none exceed this, default to the top-1 highest anomaly
        anomalous_node_indices = {res["index"] for res in node_results if res["delta_e"] > 0.3}
        if not anomalous_node_indices and node_results:
            anomalous_node_indices = {node_results[0]["index"]}
            
        anomalous_edge_indices = {res["index"] for res in edge_results if res["delta_e"] > 0.3}
        if not anomalous_edge_indices and edge_results:
            anomalous_edge_indices = {edge_results[0]["index"]}
            
    print(f"[Visualization] Anomalous nodes (indices): {anomalous_node_indices}")
    print(f"[Visualization] Anomalous edges (indices): {anomalous_edge_indices}")
    
    # 3. Load the image for drawing
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Try to load a clean TrueType font or fallback
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            font_size = 12
            
    # We need to map object ID in JSON to the actual bounding box coordinates & center point
    # so we can draw lines representing relationships.
    obj_info = {}
    id_to_index = {str(obj.get("id", str(idx))): idx for idx, obj in enumerate(objects)}
    
    # 4. Draw Node Bounding Boxes
    for idx, obj in enumerate(objects):
        name = obj.get("name", "object")
        attrs = obj.get("attributes", [])
        
        # Format label to segregate object name and attributes cleanly
        if attrs:
            label = f"{name} ({', '.join(attrs)})"
        else:
            label = name
        
        # Parse bounding box
        img_w, img_h = img.size
        bbox = sgg_builder._parse_bbox(obj.get("bounding_box", [0, 0, 0, 0]))
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        
        xmin_pix = (xmin / 1000.0) * img_w
        ymin_pix = (ymin / 1000.0) * img_h
        xmax_pix = (xmax / 1000.0) * img_w
        ymax_pix = (ymax / 1000.0) * img_h
        
        # Determine coloring: RED = anomalous, GREEN = normal
        is_node_anomalous = idx in anomalous_node_indices
        color = get_color(is_node_anomalous)
        
        # Draw bounding box (PIL takes [xmin, ymin, xmax, ymax])
        draw.rectangle([xmin_pix, ymin_pix, xmax_pix, ymax_pix], outline=color, width=3)
        
        # Draw background label for text readability
        if hasattr(draw, "textbbox"):
            t_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = t_bbox[2] - t_bbox[0]
            text_h = t_bbox[3] - t_bbox[1]
        else:
            text_w = len(label) * (font_size * 0.6)
            text_h = font_size
        
        # Place label above the bounding box
        label_y = max(0, ymin_pix - text_h - 4)
        draw.rectangle([xmin_pix, label_y, xmin_pix + text_w + 6, label_y + text_h + 4], fill=(0, 0, 0, 180))
        draw.text((xmin_pix + 3, label_y + 2), label, fill=color, font=font)
        
        # Calculate center point for drawing relationship lines
        center_x = (xmin_pix + xmax_pix) / 2
        center_y = (ymin_pix + ymax_pix) / 2
        
        # Stringify object IDs to prevent int/str lookup mismatches
        obj_info[str(obj.get("id", idx))] = {
            "center": (center_x, center_y),
            "label": name
        }
        
    # 5. Draw Relationship Lines
    for idx, rel in enumerate(relationships):
        sub_id = str(rel.get("subject_id", ""))
        obj_id = str(rel.get("object_id", ""))
        pred = rel.get("predicate", "rel")
        
        if sub_id in obj_info and obj_id in obj_info:
            sub_pt = obj_info[sub_id]["center"]
            obj_pt = obj_info[obj_id]["center"]
            
            # Determine color: RED = anomalous, GREEN = normal
            is_edge_anomalous = idx in anomalous_edge_indices
            color = get_color(is_edge_anomalous)
            
            # Draw line between objects
            draw.line([sub_pt, obj_pt], fill=color, width=2)
            
            # Draw midpoint label with background box
            mid_x = (sub_pt[0] + obj_pt[0]) / 2
            mid_y = (sub_pt[1] + obj_pt[1]) / 2
            
            pred_label = f"[{pred}]"
            if hasattr(draw, "textbbox"):
                t_bbox = draw.textbbox((0, 0), pred_label, font=font)
                label_w = t_bbox[2] - t_bbox[0]
                label_h = t_bbox[3] - t_bbox[1]
            else:
                label_w = len(pred_label) * (font_size * 0.6)
                label_h = font_size
            
            draw.rectangle([mid_x - label_w/2 - 2, mid_y - label_h/2 - 2, mid_x + label_w/2 + 2, mid_y + label_h/2 + 2], fill=(0, 0, 0, 180))
            draw.text((mid_x - label_w/2, mid_y - label_h/2), pred_label, fill=color, font=font)

    # 6. Save Visualized Output
    out_file = os.path.join(out_dir, f"{name_no_ext}_visualized{ext}")
    img.save(out_file)
    print(f"[Visualization] Successfully saved visual scene graph output to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize EBM Anomaly Localization on an image or directory.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to visualize.")
    parser.add_argument("--dir", type=str, default=None, help="Path to a directory of images.")
    parser.add_argument("--threshold", type=float, default=2.0, help="Energy threshold for anomaly detection.")
    parser.add_argument("--out_dir", type=str, default="visualized_output", help="Directory to save the visualizations.")
    args = parser.parse_args()
    
    if args.image is None and args.dir is None:
        print("Error: You must provide either --image or --dir.")
        print("  Single image:  python visualize_anomaly.py --image 1000-test/frg_lc_157.jpg")
        print("  Full directory: python visualize_anomaly.py --dir 1000-test")
        return
    
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print("[Visualization] Initializing Ablation pipeline...")
    ablator = GraphAblator(energy_threshold=args.threshold)
    sgg_builder = SceneGraphBuilder()
    
    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        process_single_image(args.image, ablator, sgg_builder, out_dir=out_dir)
    else:
        # Directory mode
        target_dir = args.dir
        if not os.path.exists(target_dir):
            print(f"Error: Directory not found: {target_dir}")
            return
            
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = []
        for f in os.listdir(target_dir):
            if os.path.splitext(f.lower())[1] in valid_exts and "_visualized" not in f:
                image_files.append(os.path.join(target_dir, f))
                
        if not image_files:
            print(f"No valid images found in {target_dir}.")
            return
            
        print(f"[Visualization] Found {len(image_files)} images in {target_dir}.")
        
        for img_path in image_files:
            try:
                process_single_image(img_path, ablator, sgg_builder, out_dir=out_dir)
            except Exception as e:
                print(f"[Visualization] Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
