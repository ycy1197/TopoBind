# TopoBind: Multi-Modal Prediction of Antibody–Antigen Binding Free Energy

A deep learning framework for predicting antibody–antigen binding free energy (ΔG) by integrating **ESM-2 embeddings** with **structural topological descriptors**.  
TopoBind leverages cross-attention, adaptive gating, and Lasso regression to achieve state-of-the-art regression and classification performance on antibody–antigen complexes.

---

## 🚀 Features

**Multi-Modal Integration**
- **Sequence embeddings** from ESM-2 (3B model, mean-pooled representations).
- **Topological descriptors** including:
  - Contact statistics  
  - Interface geometry  
  - Distance matrices  
  - Persistent homology features

**Advanced Architecture**
- Cross-attention between sequence and topological embeddings  
- Adaptive gated fusion of multiple structural sub-representations  
- Lasso regression head for ΔG prediction  

**Training & Evaluation Pipeline**
- Custom PyTorch Dataset & DataLoader with normalization  
- Early stopping and learning rate scheduling  
- Lasso regression for downstream analysis  
- ROC and regression metrics evaluation  
- Visualization of loss curves, scatter plots, error distributions, and ROC curves  

---

## 📁 Project Structure

TopoBind/
├── download_pdb.py # Download raw PDB files from RCSB
├── prepare_alphafold.py # Prepare antigen/antibody FASTA for AlphaFold/ColabFold
├── extract_topo_features.py # Extract scalar topological descriptors (contact, interface, distance, PH)
├── extract_esm_features.py # Extract ESM-2 embeddings (3B) for antibody–antigen concatenated sequences
├── train.py # Dataset & dataloader utilities (ProteinBindingDataset, collate function)
├── model.py # Model architectures (CrossAttention, EnhancedCrossAttentionModel)
├── evaluate.py # Evaluation metrics & feature extraction helpers
├── visualization.py # Training utilities, plotting, and ROC curve generation
├── main.py # Main entry point: training, feature extraction, regression & evaluation
├── data/ # (Optional) Dataset files and cached CSVs
├── esm_features/ # Cached ESM embeddings (.npy, .pkl)
├── processed_topo_features/ # Cached topological descriptors (.npz, .pkl)
├── results/ # Plots, metrics, and ROC data
└── README.md # This file
