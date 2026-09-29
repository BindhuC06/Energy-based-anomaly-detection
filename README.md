# Energy-Based Graph Attention Networks for Visual Anomaly Localization

This package contains the complete pipeline for extracting features, training the Energy-Based Model (EBM), and running Anomaly Localization inference using Qwen-2.5-VL.

## Environment Setup
First, ensure you have a Python environment with CUDA support. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Step 1: Download GQA Dataset & Extract Visual Features
Before training, the EBM requires dense visual embeddings extracted from the Large Vision-Language Model.
1. Download the GQA dataset images into a folder called gqa/images
2. Run the extraction script. This script passes all images through Qwen-2.5-VL and saves the 2048-dimensional visual feature embeddings as `.pt` files.
```bash
python extract_features.py
```

## Step 2: Train the EBM-GAT
Once the visual features are fully extracted and saved to disk, you can train the Energy-Based Model using contrastive learning.
1. Make sure `train_ebm.py` points to your extracted `features/` directory and your `train_sceneGraphs.json`.
2. Run the training script:
```bash
python train_ebm.py
```
3. This will train the GAT-EBM for 10 epochs. 
4. Once finished, it will output three files into a `models/` directory:
   - `ebm_gat.pth` (The trained model weights)
   - `node_vocab.json` (The categorical vocabulary mapping for objects)
   - `rel_vocab.json` (The categorical vocabulary mapping for relationship predicates)

## Step 3: Anomaly Localization (Inference)
Now that the model is trained, you can run the full end-to-end inference pipeline on raw images.
The pipeline extracts the Scene Graph via Qwen-VL, builds the PyTorch Geometric structure, evaluates the Baseline Energy, and performs Graph Ablation to isolate the anomalies (highlighted in red).

1. Place a few `.jpg` or `.png` test images into the `test/` directory.
2. Run the visualization script:
```bash
python visualize_anomaly.py --dir test
```
*(Alternatively, test a single image: `python visualize_anomaly.py --image test/my_image.jpg`)*
3. The script will output the annotated images into the `visualized_output/` folder. Normal bounding boxes are drawn in Green, and Anomalous sub-components are drawn in Red based on the EBM's energy differential ($\Delta E$).

## Script Architecture Overview
- `extract_features.py`: Batches images through Qwen to generate base visual embeddings for training.
- `train_ebm.py`: The main EBM trainer using Graph Attention Networks and stochastic hard-negative graph generation.
- `Perception.py`: Houses the Qwen-2.5-VL inference logic for end-to-end processing. It extracts visual embeddings directly from the initial prefill hidden states to save VRAM.
- `SGG.py`: The Scene Graph Generator. Transforms the JSON and visual tensors into PyTorch Geometric `Data` graphs.
- `ebm_inference.py` & `ebm_ablation.py`: Executes the forward passes and calculates the differential energy drops ($\Delta E$).
- `visualize_anomaly.py`: Maps the JSON bounding boxes and ablative findings onto the original image.
