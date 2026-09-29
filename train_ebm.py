import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import time

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:
    print("Please install PyTorch Geometric (torch_geometric) to run this script.")
    import sys
    sys.exit(1)

#Load Dataset
class GQADataset(Dataset):
    def __init__(self, json_path, feature_dir, limit=None):
        self.feature_dir = feature_dir
        
        print(f"[Dataset] Loading JSON from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.gqa_data = json.load(f)
            
        all_ids = list(self.gqa_data.keys())
        self.valid_ids = []
        
        print("[Dataset] Verifying extracted features...")
        for img_id in all_ids:
            if os.path.exists(os.path.join(feature_dir, f"{img_id}.pt")):
                self.valid_ids.append(img_id)
            if limit and len(self.valid_ids) >= limit:
                break
                
        print(f"[Dataset] Found {len(self.valid_ids)} valid images with extracted features.")
        
        # Build Vocabularies
        self.rel_vocab = {"<UNK>": 0}
        self.node_vocab = {"<UNK>": 0}
        
        print("[Dataset] Building vocabularies...")
        for img_id in self.valid_ids:
            objs = self.gqa_data[img_id].get("objects", {})
            for obj_id, obj_data in objs.items():
                # Add objects
                name = obj_data.get("name", "").strip()
                if name and name not in self.node_vocab:
                    self.node_vocab[name] = len(self.node_vocab)
                    
                # Add relationships
                for rel in obj_data.get("relations", []):
                    pred = rel.get("name")
                    if pred and pred not in self.rel_vocab:
                        self.rel_vocab[pred] = len(self.rel_vocab)
                        
        self.num_relations = len(self.rel_vocab)
        self.num_nodes_vocab = len(self.node_vocab)
        print(f"[Dataset] Vocabulary size: {self.num_nodes_vocab} nodes, {self.num_relations} relationships.")
        
    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        scene_data = self.gqa_data[img_id]
        
        visual_features = torch.load(os.path.join(self.feature_dir, f"{img_id}.pt"), map_location='cpu')
        global_visual = visual_features.mean(dim=0) # [2048]
        
        img_w = scene_data.get("width", 1.0)
        img_h = scene_data.get("height", 1.0)
        
        objects = scene_data.get("objects", {})
        id_to_idx = {obj_id: i for i, obj_id in enumerate(objects.keys())}
        
        node_features = []
        node_attrs = []
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        
        for obj_id, obj_data in objects.items():
            # Normalized Bounding Box [y, x, h, w]
            y = obj_data.get("y", 0) / img_h
            x = obj_data.get("x", 0) / img_w
            h = obj_data.get("h", 0) / img_h
            w = obj_data.get("w", 0) / img_w
            bbox = torch.tensor([y, x, h, w], dtype=torch.float32)
            
            node_feat = torch.cat([bbox, global_visual])
            node_features.append(node_feat)
            
            name = obj_data.get("name", "").strip()
            node_attrs.append(self.node_vocab.get(name, 0))
            
            # Edges
            for rel in obj_data.get("relations", []):
                target_id = rel.get("object")
                if target_id in id_to_idx:
                    edge_sources.append(id_to_idx[obj_id])
                    edge_targets.append(id_to_idx[target_id])
                    pred_id = self.rel_vocab.get(rel.get("name"), 0)
                    edge_attrs.append([pred_id])
                    
        x = torch.stack(node_features) if node_features else torch.empty((0, 2048 + 4))
        node_attr = torch.tensor(node_attrs, dtype=torch.long) if node_attrs else torch.empty((0,), dtype=torch.long)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long) if edge_sources else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long) if edge_attrs else torch.empty((0, 1), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr)

#contrastive learning
def generate_hard_negatives(batch_data, num_relations, num_nodes_vocab):
    neg_data = batch_data.clone()
    
    # 1. Edge Reversal
    num_edges = neg_data.edge_index.size(1)
    if num_edges > 0:
        mask = torch.rand(num_edges) < 0.3
        temp = neg_data.edge_index[0, mask].clone()
        neg_data.edge_index[0, mask] = neg_data.edge_index[1, mask]
        neg_data.edge_index[1, mask] = temp
        
    # 2. Predicate Shifting
    if num_edges > 0:
        mask = torch.rand(num_edges) < 0.3
        random_preds = torch.randint(0, num_relations, (mask.sum().item(), 1), device=neg_data.edge_attr.device)
        neg_data.edge_attr[mask] = random_preds
        
    # 3. Node Context Swapping
    num_nodes = neg_data.x.size(0)
    if num_nodes > 0:
        mask = torch.rand(num_nodes) < 0.2
        num_swap = mask.sum().item()
        if num_swap > 0:
            # We swap the categorical identity of the node with a completely random object from the global vocabulary
            random_nodes = torch.randint(0, num_nodes_vocab, (num_swap,), device=neg_data.node_attr.device)
            neg_data.node_attr[mask] = random_nodes
            
    return neg_data

#GAT-EBM model
class EBM_GAT(nn.Module):
    def __init__(self, visual_dim, num_relations, num_nodes_vocab, hidden_dim=256):
        super(EBM_GAT, self).__init__()
        
        # Embed node categorical identities
        self.node_emb = nn.Embedding(num_nodes_vocab, hidden_dim)
        
        # Project visual + bbox + embedded node identity into hidden dimension
        self.node_proj = nn.Linear(visual_dim + 4 + hidden_dim, hidden_dim)
        
        # Embed edge predicates
        self.edge_emb = nn.Embedding(num_relations, hidden_dim)
        
        # GAT Layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, add_self_loops=False)
        
        # Energy Readout (Scalar)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus() # Ensures energy is always positive preventing shortcut learning
        )
    #Forward pass
    def forward(self, x, edge_index, edge_attr, node_attr, batch=64):
        # 1. Embeddings
        node_id_feats = self.node_emb(node_attr)
        x_concat = torch.cat([x, node_id_feats], dim=-1)
        x = torch.relu(self.node_proj(x_concat))
        
        if edge_attr.numel() > 0:
            edge_attr = self.edge_emb(edge_attr.squeeze(-1))
        else:
            edge_attr = None
            
        # 2. Message Passing
        x = torch.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        
        # 3. Global Pooling
        graph_emb = global_mean_pool(x, batch)
        
        # 4. Energy Readout
        energy = self.energy_head(graph_emb)
        return energy.squeeze(-1)

#training loop
def main():
    json_path = "./data/train_sceneGraphs.json"
    feature_dir = "./gqa/features" 
    
    dataset = GQADataset(json_path, feature_dir)
    if len(dataset) == 0:
        print("No extracted features found!")
        return
        
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = EBM_GAT(visual_dim=2048, num_relations=dataset.num_relations, num_nodes_vocab=dataset.num_nodes_vocab).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    margin = 5.0
    epochs = 10
    
    print("\nStarting Contrastive EBM Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pos_energy_sum = 0.0
        neg_energy_sum = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pos_energy = model(batch.x, batch.edge_index, batch.edge_attr, batch.node_attr, batch.batch)
            
            neg_batch = generate_hard_negatives(batch, dataset.num_relations, dataset.num_nodes_vocab).to(device)
            neg_energy = model(neg_batch.x, neg_batch.edge_index, neg_batch.edge_attr, neg_batch.node_attr, neg_batch.batch)
            
            loss = pos_energy.mean() + torch.relu(margin - neg_energy).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pos_energy_sum += pos_energy.mean().item()
            neg_energy_sum += neg_energy.mean().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | Pos Energy: {pos_energy.mean().item():.4f} | Neg Energy: {neg_energy.mean().item():.4f}")
                
        avg_loss = total_loss / len(loader)
        print(f"========== Epoch {epoch+1} Summary ==========")
        print(f"Time: {time.time() - start_time:.2f}s | Avg Loss: {avg_loss:.4f} | "
              f"Avg Pos Energy: {pos_energy_sum / len(loader):.4f} | Avg Neg Energy: {neg_energy_sum / len(loader):.4f}\n")
              
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/ebm_gat.pth")
    
    with open("./models/rel_vocab.json", "w", encoding="utf-8") as f:
        json.dump(dataset.rel_vocab, f, indent=4)
        
    with open("./models/node_vocab.json", "w", encoding="utf-8") as f:
        json.dump(dataset.node_vocab, f, indent=4)
        
    print("Training Complete! Model saved to ./models/ebm_gat.pth and vocabularies to ./models/*.json")

if __name__ == "__main__":
    main()
