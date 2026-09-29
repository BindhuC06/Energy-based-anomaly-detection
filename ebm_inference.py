import os
import json
import torch
import warnings
warnings.filterwarnings("ignore")

from Perception import PerceptionLayer
from SGG import SceneGraphBuilder
from train_ebm import EBM_GAT
from torch_geometric.data import Batch

class EBMInferencePipeline:
    def __init__(self, model_path="./models/ebm_gat.pth", vocab_path="./models/rel_vocab.json", node_vocab_path="./models/node_vocab.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Inference] Initializing pipeline on {self.device}...")
        
        # 1. Load Vocabularies
        if not os.path.exists(vocab_path) or not os.path.exists(node_vocab_path):
            raise FileNotFoundError("Vocabularies not found. Did you run training?")
            
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.rel_vocab = json.load(f)
            
        with open(node_vocab_path, 'r', encoding='utf-8') as f:
            self.node_vocab = json.load(f)
        
        # 2. Load EBM GAT Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}.")
            
        self.ebm_model = EBM_GAT(visual_dim=2048, num_relations=len(self.rel_vocab), num_nodes_vocab=len(self.node_vocab)).to(self.device)
        self.ebm_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.ebm_model.eval()
        
        # 3. Load Perception & SGG
        self.perception = PerceptionLayer(device=self.device)
        self.sgg_builder = SceneGraphBuilder()
        print("[Inference] Pipeline Ready.")

    def run(self, image_path: str):
        print(f"\n--- Processing {image_path} ---")
        perception_output = self.perception.process_image(image_path)
        scene_json = perception_output["scene_graph"]
        image_embeddings = perception_output["image_embeddings"]
        
        # Get image dimensions to normalize bounding boxes
        from PIL import Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size
            
        global_visual = image_embeddings.detach().cpu().to(torch.float32).mean(dim=0)
        
        graph = self.sgg_builder.build_graph(scene_json, img_w=img_w, img_h=img_h)
        num_nodes = graph.x.size(0)
        
        if num_nodes > 0:
            bboxes = graph.x[:, 0:4]
            global_visual_tiled = global_visual.unsqueeze(0).repeat(num_nodes, 1)
            final_node_features = torch.cat([bboxes, global_visual_tiled], dim=1)
        else:
            final_node_features = torch.empty((0, 2048 + 4))
            
        # Map predicate strings to vocabulary IDs
        edge_attrs = []
        for text in getattr(graph, 'edge_texts', []):
            edge_attrs.append([self.rel_vocab.get(text, 0)]) 
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.long) if edge_attrs else torch.empty((0, 1), dtype=torch.long)
        
        # Map object names to vocabulary IDs
        node_attrs = []
        for text in getattr(graph, 'node_texts', []):
            # The SGG builds node_texts as "attribute name" so we extract the last word (name) to match our vocab
            name = text.split()[-1] if text else ""
            node_attrs.append(self.node_vocab.get(name, 0))
        node_attr_tensor = torch.tensor(node_attrs, dtype=torch.long) if node_attrs else torch.empty((0,), dtype=torch.long)
        
        graph.x = final_node_features
        graph.edge_attr = edge_attr_tensor
        graph.node_attr = node_attr_tensor
        
        if num_nodes == 0:
            print("[Inference] Warning: Scene graph contains 0 objects (parsing failed or empty scene). Assigning default anomaly energy score.")
            return {
                "energy": 10.0,
                "scene_json": scene_json,
                "graph_data": graph
            }

        batch = Batch.from_data_list([graph]).to(self.device)
        
        print("[Inference] Calculating Energy Score...")
        with torch.no_grad():
            energy = self.ebm_model(batch.x, batch.edge_index, batch.edge_attr, batch.node_attr, batch.batch)
            
        final_energy = energy.item()
        print(f"=====================================")
        print(f"FINAL ENERGY SCORE: {final_energy:.4f}")
        print(f"=====================================")
        
        return {
            "energy": final_energy,
            "scene_json": scene_json,
            "graph_data": graph
        }

if __name__ == "__main__":
    pass
