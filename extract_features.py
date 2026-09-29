import os
import json
import torch
import gc
import time
import argparse
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description="Extract visual features from GQA images using Qwen2.5-VL.")
    parser.add_argument("--json_path", type=str, default="./data/train_sceneGraphs.json", help="Path to GQA JSON")
    parser.add_argument("--image_dir", type=str, default="./gqa/images", help="Directory containing GQA images")
    parser.add_argument("--out_dir", type=str, default="./gqa/features", help="Output directory for .pt tensors")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="LVLM model ID")
    parser.add_argument("--limit", type=int, default=70000, help="Maximum number of images to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Index to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading GQA JSON from {args.json_path}...")
    with open(args.json_path, 'r', encoding='utf-8') as f:
        gqa_data = json.load(f)
        
    image_ids = list(gqa_data.keys())
    total_images = len(image_ids)
    print(f"Found {total_images} total images in JSON.")
    
    if args.limit > 0:
        image_ids = image_ids[args.start_idx:args.start_idx + args.limit]
        print(f"Limiting processing to {len(image_ids)} images (starting from index {args.start_idx}).")
        
    print(f"Loading Model: {args.model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    image_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    
    # We use a very basic prompt to simply force the model to process the image tokens 
    # through the visual encoder and projection layers.
    prompt_text = "Describe the image."
    
    print("Starting Feature Extraction...")
    start_time = time.time()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, img_id in enumerate(image_ids, 1):
        out_path = os.path.join(args.out_dir, f"{img_id}.pt")
        
        # Skip if already extracted (useful for resuming)
        if os.path.exists(out_path):
            skip_count += 1
            if idx % 100 == 0:
                print(f"[{idx}/{len(image_ids)}] Skipped {img_id} (Already exists)")
            continue
            
        img_path = os.path.join(args.image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            error_count += 1
            print(f"[{idx}/{len(image_ids)}] Error: Image not found at {img_path}")
            continue
            
        try:
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            
            # Forward pass (prefill only, no generation needed!)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
            final_layer_hidden = outputs.hidden_states[-1]
            image_mask = inputs.input_ids == image_pad_token_id
            image_embeddings = final_layer_hidden[image_mask]
            
            # Detach and move to CPU to save disk space and RAM, convert to float32/16
            image_embeddings = image_embeddings.detach().cpu().to(torch.float32)
            
            torch.save(image_embeddings, out_path)
            success_count += 1
            
            # Clear memory immediately
            del inputs, outputs, final_layer_hidden, image_mask, image_embeddings
            
        except Exception as e:
            error_count += 1
            print(f"[{idx}/{len(image_ids)}] Error processing {img_id}: {e}")
            
        # Aggressive memory cleanup every 10 images
        if idx % 10 == 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
        # Progress reporting
        if idx % 10 == 0 or idx == len(image_ids):
            elapsed = time.time() - start_time
            # Only calculate ETA based on actually processed images to avoid skewed ETAs from skipped ones
            processed = success_count + error_count
            if processed > 0:
                avg_time = elapsed / processed
                remaining = len(image_ids) - idx
                eta = avg_time * remaining
                eta_str = f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m" if eta > 3600 else f"{int(eta // 60)}m {int(eta % 60)}s"
            else:
                eta_str = "Calculating..."
                
            bar_len = 30
            filled_len = int(round(bar_len * idx / len(image_ids)))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            
            print(f"[{bar}] {idx}/{len(image_ids)} ({(idx/len(image_ids))*100:.1f}%) | "
                  f"Success: {success_count} | Skip: {skip_count} | Err: {error_count} | ETA: {eta_str}")

    print("\nExtraction Complete!")
    print(f"Successfully Extracted: {success_count}")
    print(f"Skipped (Already Existed): {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Total Time: {(time.time() - start_time) / 3600:.2f} hours")

if __name__ == "__main__":
    main()
