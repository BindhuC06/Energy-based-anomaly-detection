import os
import torch
import warnings
warnings.filterwarnings("ignore")

from ebm_inference import EBMInferencePipeline
from torch_geometric.data import Data, Batch

class GraphAblator:
    def __init__(self, energy_threshold=2.0):
        self.pipeline = EBMInferencePipeline()
        self.ebm_model = self.pipeline.ebm_model
        self.device = self.pipeline.device
        self.energy_threshold = energy_threshold
        
    def _calculate_energy(self, sub_graph: Data) -> float:
        """Helper to run the EBM on a specific PyG Data object."""
        batch = Batch.from_data_list([sub_graph]).to(self.device)
        with torch.no_grad():
            energy = self.ebm_model(batch.x, batch.edge_index, batch.edge_attr, batch.node_attr, batch.batch)
        return energy.item()

    def run_ablation(self, image_path: str):
        """
        Runs the full pipeline. If the scene is anomalous, systematically ablates 
        nodes and edges to localize the source of the anomaly.
        """
        print(f"\n[Ablation Module] Initializing for {image_path}...")
        
        # 1. Run baseline inference
        inference_result = self.pipeline.run(image_path)
        base_energy = inference_result["energy"]
        scene_json = inference_result["scene_json"]
        base_graph = inference_result["graph_data"]
        
        print(f"\n[Ablation] Baseline Energy: {base_energy:.4f}")
        
        if base_energy < self.energy_threshold:
            print("[Ablation] Scene energy is BELOW the anomaly threshold. No ablation needed. Scene is normal.")
            return {
                "is_anomalous": False,
                "base_energy": base_energy,
                "node_results": [],
                "edge_results": [],
                "scene_json": scene_json
            }
            
        print("[Ablation] High Energy Detected! Starting Anomaly Localization (Graph Ablation)...")
        
        num_nodes = base_graph.x.size(0)
        if num_nodes == 0:
            print("[Ablation] No nodes in graph to ablate.")
            return

        num_edges = base_graph.edge_index.size(1)
        
        node_results = []
        edge_results = []
        
        # 2. Node Ablation (Find the out-of-place object)
        print(f"[Ablation] Testing {num_nodes} nodes...")
        objects_list = scene_json.get("objects", [])
        
        for node_idx in range(num_nodes):
            # Create a mask to remove this node
            node_mask = torch.ones(num_nodes, dtype=torch.bool)
            node_mask[node_idx] = False
            
            # Create a mask to remove any edges connected to this node
            edge_mask = (base_graph.edge_index[0] != node_idx) & (base_graph.edge_index[1] != node_idx)
            
            # Construct the ablated subgraph
            ablated_graph = Data(
                x = base_graph.x[node_mask],
                edge_index = base_graph.edge_index[:, edge_mask],
                edge_attr = base_graph.edge_attr[edge_mask] if base_graph.edge_attr.numel() > 0 else base_graph.edge_attr,
                node_attr = base_graph.node_attr[node_mask] if getattr(base_graph, 'node_attr', None) is not None else None
            )
            
            # Remap edge indices because we deleted a node
            # (PyG requires contiguous indices from 0 to N-1). 
            # We create a mapping tensor: mapping[old_idx] = new_idx
            mapping = torch.zeros(num_nodes, dtype=torch.long)
            mapping[node_mask] = torch.arange(node_mask.sum().item())
            ablated_graph.edge_index = mapping[ablated_graph.edge_index]
            
            ablated_energy = self._calculate_energy(ablated_graph)
            delta_e = base_energy - ablated_energy
            
            # Get semantic name for reporting
            obj_name = "Unknown"
            if node_idx < len(objects_list):
                obj_data = objects_list[node_idx]
                obj_name = f"{' '.join(obj_data.get('attributes', []))} {obj_data.get('name', '')}".strip()
                
            node_results.append({
                "type": "Node",
                "index": node_idx,
                "description": obj_name,
                "delta_e": delta_e
            })
            
        # 3. Edge Ablation (Find the broken relationship)
        print(f"[Ablation] Testing {num_edges} relationships...")
        relationships_list = scene_json.get("relationships", [])
        
        for edge_idx in range(num_edges):
            edge_mask = torch.ones(num_edges, dtype=torch.bool)
            edge_mask[edge_idx] = False
            
            ablated_graph = Data(
                x = base_graph.x, # Nodes remain untouched
                edge_index = base_graph.edge_index[:, edge_mask],
                edge_attr = base_graph.edge_attr[edge_mask],
                node_attr = getattr(base_graph, 'node_attr', None)
            )
            
            ablated_energy = self._calculate_energy(ablated_graph)
            delta_e = base_energy - ablated_energy
            
            # Get semantic name for reporting
            rel_desc = "Unknown Edge"
            if edge_idx < len(relationships_list):
                rel = relationships_list[edge_idx]
                # Try to map back IDs to names for readability
                subj_name = rel.get('subject_id', 'Subj')
                obj_name = rel.get('object_id', 'Obj')
                # A robust mapping would use the id_to_index mapping from SGG, but this is a quick summary
                rel_desc = f"[{subj_name}] --({rel.get('predicate', 'related')})--> [{obj_name}]"
                
            edge_results.append({
                "type": "Edge",
                "index": edge_idx,
                "description": rel_desc,
                "delta_e": delta_e
            })
            
        # 4. Compile Report
        # Sort both lists by Delta E descending (highest drop in energy = most anomalous)
        node_results.sort(key=lambda x: x["delta_e"], reverse=True)
        edge_results.sort(key=lambda x: x["delta_e"], reverse=True)
        
        print("\n=====================================")
        print("     ANOMALY LOCALIZATION REPORT     ")
        print("=====================================")
        print(f"Original Scene Energy: {base_energy:.4f}\n")
        
        print("--- Top Anomalous Objects (Nodes) ---")
        for i, res in enumerate(node_results[:3]):
            print(f"{i+1}. {res['description']} | Energy Drop (ΔE): {res['delta_e']:.4f}")
            
        print("\n--- Top Anomalous Relationships (Edges) ---")
        for i, res in enumerate(edge_results[:3]):
            print(f"{i+1}. {res['description']} | Energy Drop (ΔE): {res['delta_e']:.4f}")
        print("=====================================")

        return {
            "is_anomalous": True,
            "base_energy": base_energy,
            "node_results": node_results,
            "edge_results": edge_results,
            "scene_json": scene_json
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EBM Anomaly Localization on an image.")
    parser.add_argument("image_path", type=str, help="Path to the image to analyze.")
    parser.add_argument("--threshold", type=float, default=2.0, help="Energy threshold for anomaly detection.")
    args = parser.parse_args()
    
    if os.path.exists("./models/ebm_gat.pth") and os.path.exists("./models/rel_vocab.json"):
        ablator = GraphAblator(energy_threshold=args.threshold) 
        ablator.run_ablation(args.image_path)
    else:
        print("Error: Trained models/vocabulary not found in ./models/")
