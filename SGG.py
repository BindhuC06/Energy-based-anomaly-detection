import torch
try:
    from torch_geometric.data import Data
except ImportError:
    print("[SGG] torch_geometric is not installed. Please install it using: pip install torch_geometric")
    # Define a dummy class so the file can still be parsed before installation
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class SceneGraphBuilder:
    def __init__(self):
        """
        Initializes the Scene Graph Builder. 
        This module converts structured JSON from the Perception layer into PyTorch Geometric Data objects.
        """
        pass

    def _parse_bbox(self, bbox_data):
        """Safely parses bounding box data into a list of 4 floats."""
        if isinstance(bbox_data, str):
            # Try to strip brackets and split if it came as a raw string "[y, x, y, x]"
            bbox_data = bbox_data.strip("[]")
            try:
                return [float(x.strip()) for x in bbox_data.split(',')]
            except ValueError:
                return [0.0, 0.0, 0.0, 0.0]
        elif isinstance(bbox_data, list):
            return [float(x) for x in bbox_data]
        return [0.0, 0.0, 0.0, 0.0]

    def build_graph(self, scene_json: dict, image_embeddings: torch.Tensor = None, img_w: float = 1.0, img_h: float = 1.0):
        """
        Builds a PyTorch Geometric Data object from the perception JSON.
        
        Args:
            scene_json (dict): The parsed JSON containing 'objects' and 'relationships'.
            image_embeddings (torch.Tensor, optional): The dense visual masks extracted from LVLM.
            img_w (float): Image width for bbox normalization.
            img_h (float): Image height for bbox normalization.
        """
        objects = scene_json.get("objects", [])
        relationships = scene_json.get("relationships", [])
        
        # 1. Map string IDs to integer indices for PyG
        id_to_index = {}
        for idx, obj in enumerate(objects):
            obj_id = obj.get("id", str(idx))
            id_to_index[obj_id] = idx
            
        num_nodes = len(objects)
        
        # 2. Construct Node Features (x)
        # We must match the EBM training format: [y_norm, x_norm, h_norm, w_norm, centrality]
        node_features = []
        node_texts = [] 
        
        for obj in objects:
            bbox = self._parse_bbox(obj.get("bounding_box", [0,0,0,0]))
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Normalize to [0, 1] for EBM features
            y_norm = ymin / 1000.0
            x_norm = xmin / 1000.0
            h_norm = (ymax - ymin) / 1000.0
            w_norm = (xmax - xmin) / 1000.0
            
            centrality = float(obj.get("centrality_score", 0.5))
            
            # Combine into a tensor
            feature_vec = [y_norm, x_norm, h_norm, w_norm, centrality]
            node_features.append(feature_vec)
            
            # Combine name and attributes for semantic embedding
            name = obj.get("name", "unknown")
            attrs = " ".join(obj.get("attributes", []))
            node_texts.append(f"{attrs} {name}".strip())
            
        if num_nodes > 0:
            x = torch.tensor(node_features, dtype=torch.float)
        else:
            x = torch.empty((0, 5), dtype=torch.float)

        # 3. Construct Edges (edge_index) and Edge Attributes
        edge_sources = []
        edge_targets = []
        edge_texts = [] # Store predicates to be converted to edge_attr embeddings in GAT
        
        for rel in relationships:
            subj_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            predicate = rel.get("predicate", "related to")
            
            if subj_id in id_to_index and obj_id in id_to_index:
                edge_sources.append(id_to_index[subj_id])
                edge_targets.append(id_to_index[obj_id])
                edge_texts.append(predicate)
                
        if len(edge_sources) > 0:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 4. Assemble the PyG Data object
        # PyG Data can hold custom attributes. We attach texts and embeddings for downstream usage.
        graph_data = Data(
            x=x, 
            edge_index=edge_index
        )
        
        # Attach semantic text fields so the EBM-GAT can run them through an nn.Embedding 
        # or CLIP encoder during its forward pass for maximum accuracy!
        graph_data.node_texts = node_texts
        graph_data.edge_texts = edge_texts
        
        # Attach the raw global image embeddings from the LVLM (the semantic mask)
        if image_embeddings is not None:
            graph_data.image_mask = image_embeddings
            
        return graph_data

if __name__ == "__main__":
    # Quick Test
    test_json = {
        "objects": [
            {
                "id": "obj1",
                "name": "dog",
                "attributes": ["brown", "furry"],
                "bounding_box": [0.1, 0.2, 0.5, 0.6],
                "centrality_score": 0.9
            },
            {
                "id": "obj2",
                "name": "frisbee",
                "attributes": ["red", "plastic"],
                "bounding_box": [0.5, 0.5, 0.6, 0.7],
                "centrality_score": 0.7
            }
        ],
        "relationships": [
            {
                "subject_id": "obj1",
                "predicate": "catching",
                "object_id": "obj2"
            }
        ]
    }
    
    # Dummy image embeddings (64 tokens, 2048 hidden dim)
    dummy_embeddings = torch.randn((64, 2048))
    
    builder = SceneGraphBuilder()
    graph = builder.build_graph(test_json, dummy_embeddings)
    
    print("--- SGG Graph Built Successfully ---")
    print(f"Nodes (x): {graph.x.shape} (bbox + centrality)")
    print(f"Edge Index: {graph.edge_index.shape}")
    print(f"Node Texts (Semantic): {graph.node_texts}")
    print(f"Edge Texts (Predicate): {graph.edge_texts}")
    print(f"Image Mask Context: {graph.image_mask.shape}")
