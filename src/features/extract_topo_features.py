#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract scalar topological features from antibody–antigen complexes.

- Input : precomputed topo feature files (*.npz.gz), one per PDB
- Output:
    processed_topo_features/<PDB>.npz     # per-PDB scalar feature arrays
    processed_topo_features/topo_features.pkl   # dict {PDB -> feature values + names}
    processed_topo_features/topo_feature_names.txt  # list of feature names
    processed_topo_features/processed_topo_pdbs.txt # resume support
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gzip
import io

def extract_topo_scalar_features(topo_file):
    """Extract scalar features from a single topological feature file."""
    features = {}
    try:
        with gzip.open(topo_file, 'rb') as f:
            file_content = f.read()
        topo_data = np.load(io.BytesIO(file_content), allow_pickle=True)

        # 1. Contact map features
        for obj_type in ['antibody', 'antigen']:
            contact_key = f'{obj_type}_contact_map'
            if contact_key in topo_data:
                contact_map = topo_data[contact_key]
                if contact_map.shape[0] > 0:
                    features[f'{obj_type}_contact_density'] = contact_map.sum() / max(1, contact_map.size)
                    contact_degrees = contact_map.sum(axis=1)
                    features[f'{obj_type}_contact_degree_mean'] = contact_degrees.mean()
                    features[f'{obj_type}_contact_degree_std'] = contact_degrees.std()
                    features[f'{obj_type}_contact_degree_max'] = contact_degrees.max()
                    if contact_map.shape[0] > 2:
                        features[f'{obj_type}_clustering'] = (
                            np.trace(np.matmul(contact_map, np.matmul(contact_map, contact_map))) /
                            max(1, np.sum(contact_degrees * (contact_degrees - 1)))
                        )

        # 2. Interface features
        if 'interface_contact' in topo_data:
            interface = topo_data['interface_contact']
            features['interface_size'] = interface.sum()
            features['interface_density'] = interface.sum() / max(1, interface.size)
            if interface.shape[0] > 0:
                features['ab_interface_coverage'] = np.sum(np.any(interface, axis=1)) / interface.shape[0]
                features['ag_interface_coverage'] = np.sum(np.any(interface, axis=0)) / interface.shape[1]
                ab_interface_res = np.sum(interface, axis=1)
                ag_interface_res = np.sum(interface, axis=0)
                if len(ab_interface_res) > 0:
                    features['ab_interface_res_mean'] = (
                        ab_interface_res[ab_interface_res > 0].mean() if np.any(ab_interface_res > 0) else 0
                    )
                if len(ag_interface_res) > 0:
                    features['ag_interface_res_mean'] = (
                        ag_interface_res[ag_interface_res > 0].mean() if np.any(ag_interface_res > 0) else 0
                    )

        # 3. Distance matrix features
        for obj_type in ['antibody', 'antigen', 'interface']:
            dist_key = f'{obj_type}_dist_matrix' if obj_type != 'interface' else 'interface_dist'
            if dist_key in topo_data:
                dist_matrix = topo_data[dist_key]
                if obj_type != 'interface':
                    valid_dists = dist_matrix[~np.isinf(dist_matrix) & ~np.isnan(dist_matrix) & (dist_matrix > 0)]
                    if len(valid_dists) > 0:
                        features[f'{obj_type}_min_dist'] = np.min(valid_dists)
                        features[f'{obj_type}_mean_dist'] = np.mean(valid_dists)
                        features[f'{obj_type}_median_dist'] = np.median(valid_dists)
                else:
                    valid_dists = dist_matrix[~np.isinf(dist_matrix) & ~np.isnan(dist_matrix)]
                    if len(valid_dists) > 0:
                        features['interface_min_dist'] = np.min(valid_dists)
                        features['interface_mean_dist'] = np.mean(valid_dists)
                        features['interface_contact_count'] = np.sum(valid_dists <= 8.0)  # 8Å threshold

        # 4. Persistent homology features
        for obj_type in ['antibody', 'antigen', 'interface']:
            for dim in range(3):  # 0,1,2 dimensions
                key = f'{obj_type}_persistence_dim{dim}'
                if key in topo_data:
                    diagram = topo_data[key]
                    if isinstance(diagram, np.ndarray) and len(diagram) > 0:
                        lifetimes = diagram[:, 1] - diagram[:, 0]
                        features[f'{obj_type}_topo_dim{dim}_count'] = len(diagram)
                        features[f'{obj_type}_topo_dim{dim}_mean_lifetime'] = lifetimes.mean()
                        features[f'{obj_type}_topo_dim{dim}_max_lifetime'] = lifetimes.max() if len(lifetimes) > 0 else 0
                        features[f'{obj_type}_topo_dim{dim}_sum_lifetime'] = lifetimes.sum()
                        features[f'{obj_type}_topo_dim{dim}_std_lifetime'] = lifetimes.std() if len(lifetimes) > 1 else 0
                        sorted_lifetimes = np.sort(lifetimes)[::-1]
                        for i in range(min(5, len(sorted_lifetimes))):
                            features[f'{obj_type}_topo_dim{dim}_lifetime_{i+1}'] = sorted_lifetimes[i]
                    else:
                        features[f'{obj_type}_topo_dim{dim}_count'] = 0
                        features[f'{obj_type}_topo_dim{dim}_mean_lifetime'] = 0
                        features[f'{obj_type}_topo_dim{dim}_max_lifetime'] = 0
        return features
    except Exception as e:
        print(f"Error processing topo file {topo_file}: {str(e)}")
        return {}

def process_and_save_topo_features(csv_file, topo_dir, output_dir='processed_topo_features'):
    """Process all PDB topo feature files and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    print(f"Dataset size: {len(df)}")
    
    pdb_ids = sorted(set(df['pdb'].str.split('_').str[0]))
    print(f"Unique PDB IDs: {len(pdb_ids)}")
    
    processed_file = os.path.join(output_dir, 'processed_topo_pdbs.txt')
    processed_pdbs = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_pdbs = set(line.strip() for line in f)
        print(f"Already processed: {len(processed_pdbs)} PDBs")
    
    remaining_pdbs = [pdb for pdb in pdb_ids if pdb not in processed_pdbs]
    print(f"Remaining to process: {len(remaining_pdbs)} PDBs")
    
    topo_features_dict = {}
    with open(processed_file, 'a') as f_processed:
        for pdb_id in tqdm(remaining_pdbs, desc="Processing topo features"):
            topo_file = os.path.join(topo_dir, f"{pdb_id}_topo_features.npz.gz")
            
            if not os.path.exists(topo_file):
                print(f"Warning: topo file not found for {pdb_id}")
                continue
            
            try:
                features = extract_topo_scalar_features(topo_file)
                if features:
                    feature_names = sorted(features.keys())
                    feature_values = np.array([features[name] for name in feature_names])
                    
                    output_file = os.path.join(output_dir, f"{pdb_id}.npz")
                    np.savez_compressed(output_file, feature_values=feature_values, feature_names=feature_names)
                    
                    topo_features_dict[pdb_id] = {
                        'feature_values': feature_values,
                        'feature_names': feature_names
                    }
                    
                    f_processed.write(f"{pdb_id}\n")
                    f_processed.flush()
            except Exception as e:
                print(f"Error processing PDB {pdb_id}: {str(e)}")
    
    for pdb_id in processed_pdbs:
        try:
            feature_file = os.path.join(output_dir, f"{pdb_id}.npz")
            if os.path.exists(feature_file):
                data = np.load(feature_file, allow_pickle=True)
                topo_features_dict[pdb_id] = {
                    'feature_values': data['feature_values'],
                    'feature_names': data['feature_names'].tolist() if hasattr(data['feature_names'], 'tolist') else data['feature_names']
                }
        except Exception as e:
            print(f"Error loading features for PDB {pdb_id}: {str(e)}")
    
    if topo_features_dict:
        sample_pdb = next(iter(topo_features_dict))
        all_feature_names = topo_features_dict[sample_pdb]['feature_names']
        with open(os.path.join(output_dir, 'topo_feature_names.txt'), 'w') as f:
            for name in all_feature_names:
                f.write(f"{name}\n")
    
    with open(os.path.join(output_dir, 'topo_features.pkl'), 'wb') as f:
        pickle.dump(topo_features_dict, f)
    
    print(f"Finished! Total processed topo features: {len(topo_features_dict)} PDBs")
    return topo_features_dict

if __name__ == "__main__":
    csv_file = 'alldata.csv'
    topo_dir = 'topo_features'
    topo_features = process_and_save_topo_features(csv_file, topo_dir)
