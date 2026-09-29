import os
import json
import torch
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class PerceptionLayer:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda"):
        self.device = device
        self.model_id = model_id
        
        print(f"[PerceptionLayer] Loading model {self.model_id}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Get the ID for the <|image_pad|> token, used to mask the visual embeddings
        self.image_pad_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        print("[PerceptionLayer] Model loaded successfully.")

    def parse_json_output(self, text: str) -> dict:
        """Attempts to safely parse JSON from the model's raw text output, including auto-repairing truncated JSON."""
        # Remove any surrounding markdown block code characters
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[PerceptionLayer] JSON Decoding Error: {e}")
            print("[PerceptionLayer] Attempting auto-repair of truncated JSON...")
            
            # Auto-repair strategy for truncated strings/JSON
            repaired = cleaned
            # 1. Close unterminated double quote if needed
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            
            # 2. Balance brackets and braces
            open_brackets = repaired.count('[') - repaired.count(']')
            open_braces = repaired.count('{') - repaired.count('}')
            
            repaired += ']' * max(0, open_brackets)
            repaired += '}' * max(0, open_braces)
            
            try:
                parsed = json.loads(repaired)
                print("[PerceptionLayer] Auto-repair successful!")
                return parsed
            except Exception:
                pass
                
            # If basic repair fails, attempt to extract valid objects via regex or partial load
            print(f"Raw Output: {text}")
            return {"objects": [], "relationships": [], "error": "Failed to parse JSON", "raw_text": text}

    def process_image(self, image_path: str):
        """
        Processes an image to extract a structured JSON scene graph and dense visual embeddings.
        Returns:
            dict: {
                "scene_graph": dict (Parsed JSON),
                "image_embeddings": torch.Tensor (Shape: [num_image_tokens, hidden_dim])
            }
        """
        print(f"[PerceptionLayer] Processing {image_path}...")
        
        prompt = (
            "You are a specialized visual perception model. Analyze the provided image in detail. "
            "Extract a comprehensive Scene Graph containing the identified objects and their relationships. "
            "You MUST output valid JSON only. Do not include any conversational text or markdown formatting outside of the JSON block.\n\n"
            "JSON Structure:\n"
            "{\n"
            '  "objects": [\n'
            '    {\n'
            '      "id": "<unique_string_id>",\n'
            '      "name": "<single_word_noun>", // CRITICAL: This MUST be a single word (e.g. "woman", "table", "man", "microphone"). Do NOT use phrases like "woman speaking".\n'
            '      "attributes": ["<attr1>", "<attr2>"],\n'
            '      "bounding_box": "[ymin, xmin, ymax, xmax]", // Coordinates MUST be normalized between 0 and 1000\n'
            '      "centrality_score": <float between 0.0 and 1.0 indicating importance in the scene>\n'
            '    }\n'
            '  ],\n'
            '  "relationships": [\n'
            '    {\n'
            '      "subject_id": "<id_from_objects>",\n'
            '      "predicate": "<simple_spatial_or_action_verb>", // CRITICAL: Use simple vocabulary (e.g. "at", "on", "holding", "behind").\n'
            '      "object_id": "<id_from_objects>"\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "CRITICAL: relationships must ONLY link subject_id and object_id that exist in the objects list. Do not link to attributes or terms not in the objects list."
        )
        
        # Load and resize image to prevent CUDA OOM on large images
        img = Image.open(image_path).convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)
        
        print("[PerceptionLayer] Running inference...")
        
        # Use output_hidden_states=True to get final layer representations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=8192,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
        generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        parsed_json = self.parse_json_output(output_text)
        
        # In return_dict_in_generate=True for Qwen2.5-VL generation, 
        # hidden_states is returned as a tuple of tuples.
        # But we need the hidden states from the forward pass of the prompt itself (the prefill phase),
        # not the auto-regressive generation steps, to get the image mask embeddings.
        
        print("[PerceptionLayer] Extracting visual embeddings from prefill...")
        # Extract prefill hidden states directly from the generate output
        # outputs.hidden_states[0] is the prefill tuple of layers, [-1] is the final layer
        final_layer_hidden = outputs.hidden_states[0][-1] # [batch, seq_len, hidden_dim]
        
        # Isolate the image tokens
        image_mask = inputs.input_ids == self.image_pad_token_id
        image_embeddings = final_layer_hidden[image_mask] # [num_image_tokens, hidden_dim]
        
        return {
            "scene_graph": parsed_json,
            "image_embeddings": image_embeddings
        }

if __name__ == "__main__":
    # Test script if executed directly
    perception = PerceptionLayer()
    
    test_img = "dummy.jpg"
    # if not os.path.exists(test_img):
    #     # Create a red dummy image just for testing if one doesn't exist
    #     img = Image.new('RGB', (224, 224), color = 'red')
    #     img.save(test_img)
        
    print(f"\n--- Testing Perception Layer on {test_img} ---")
    result = perception.process_image(test_img)
    
    print("\n[Output JSON Structure]")
    print(json.dumps(result["scene_graph"], indent=2))
    
    print(f"\n[Image Embeddings Shape]: {result['image_embeddings'].shape}")
