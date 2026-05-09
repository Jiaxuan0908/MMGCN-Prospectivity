import os
import numpy as np
import pandas as pd
import dgl
import torch
from typing import Dict, Tuple, List
from collections import defaultdict


def load_node_features(node_path: str) -> Tuple[Dict, Dict, int]:

    node_df = pd.read_csv(
        node_path,
        header=None,
        names=['global_id', 'type'] + [f'feat_{i}' for i in range(5)],
        dtype={'global_id': int}
    )
    node_features = {
        'known': {}, 'structure': {}, 'chemical': {}, 'unknown': {}
    }
    node_id_map = {ntype: [] for ntype in ['known', 'structure', 'chemical', 'unknown']}

    for _, row in node_df.iterrows():
        global_id = row['global_id']
        ntype = row['type']


        if ntype in ['known', 'unknown']:
            node_features[ntype][global_id] = torch.zeros(5, dtype=torch.float32)
            node_id_map[ntype].append(global_id)
            continue

        if ntype == 'structure':
            base_feats = row[['feat_0', 'feat_1', 'feat_2', 'feat_3']].values.astype(np.float32)
            base_feats = np.nan_to_num(base_feats, nan=0.0, posinf=1e5, neginf=-1e5)
            base_feats = np.clip(base_feats, -1e4, 1e4)
            node_features['structure'][global_id] = torch.tensor(
                base_feats,
                dtype=torch.float32
            )
            node_id_map['structure'].append(global_id)

        elif ntype == 'chemical':

            base_feats = row[[f'feat_{i}' for i in range(5)]].values.astype(np.float32)
            base_feats = np.nan_to_num(base_feats, nan=0.0, posinf=1e5, neginf=-1e5)
            base_feats = np.clip(base_feats, -1e4, 1e4)

            sum_feat = np.sum(base_feats).reshape(1, )
            max_feat = np.max(base_feats).reshape(1, )
            combined_feat = np.concatenate([base_feats, sum_feat, max_feat], axis=0)

            node_features['chemical'][global_id] = torch.tensor(
                combined_feat,
                dtype=torch.float32
            )
            node_id_map['chemical'].append(global_id)
        # ======================================================

    if node_id_map['known']:
        sample_elem_feat = node_features['known'][node_id_map['known'][0]]
        print(f"known: {sample_elem_feat.shape}, mean: {sample_elem_feat.mean().item():.6f}")
    if node_id_map['unknown']:
        sample_unk_feat = node_features['unknown'][node_id_map['unknown'][0]]
        print(f"unknown: {sample_unk_feat.shape}, mean: {sample_unk_feat.mean().item():.6f}")
    if node_id_map['structure']:
        sample_struct_feat = node_features['structure'][node_id_map['structure'][0]]
        print(f"structure: {sample_struct_feat.shape}")
    if node_id_map['chemical']:
        sample_chem_feat = node_features['chemical'][node_id_map['chemical'][0]]
        print(f"chemical: {sample_chem_feat.shape}")

    min_max_ids = {
        ntype: (min(ids), max(ids)) for ntype, ids in node_id_map.items() if ids
    }
    unknown_min_id = min(node_id_map['unknown']) if node_id_map['unknown'] else -1
    return node_features, min_max_ids, unknown_min_id


def load_edges(edge_path: str) -> Dict:

    edge_df = pd.read_csv(
        edge_path,
        header=None,
        names=['source', 'target', 'type', 'edge_id', 'attr', 'weight'],
        dtype={'source': int, 'target': int, 'attr': np.float32}
    )
    edge_dict = {
        'structure_to_known': {'src': [], 'dst': [], 'attr': []},
        'chemical_to_known': {'src': [], 'dst': [], 'attr': []},
        'structure_to_unknown': {'src': [], 'dst': [], 'attr': []},
        'chemical_to_unknown': {'src': [], 'dst': [], 'attr': []}
    }
    for _, row in edge_df.iterrows():
        edge_type = row['type']
        if edge_type not in edge_dict:
            continue
        edge_dict[edge_type]['src'].append(row['source'])
        edge_dict[edge_type]['dst'].append(row['target'])
        edge_dict[edge_type]['attr'].append(row['attr'])

    for etype, data in edge_dict.items():
        print(f"egdetype {etype}:  {len(data['src'])} ")
    return edge_dict


def build_multiscale_subgraphs(
        min_max_ids: Dict, edge_dict: Dict, node_features: Dict, label_dict: Dict,
        D1: int = 800,
        D2: int = 1000
) -> Tuple[Dict, Dict]:
    print("\n===== start =====")

    def build_adjacency(etype):
        adj = defaultdict(list)
        for s, t, d in zip(edge_dict[etype]['src'], edge_dict[etype]['dst'], edge_dict[etype]['attr']):
            adj[t].append((s, d))
        return adj

    adj_struct_to_elem = build_adjacency('structure_to_known')
    adj_chem_to_elem = build_adjacency('chemical_to_known')
    adj_struct_to_unk = build_adjacency('structure_to_unknown')
    adj_chem_to_unk = build_adjacency('chemical_to_unknown')

    train_subgraphs = {'fine': [], 'coarse': []}
    element_ids = list(node_features['known'].keys())


    drop_train_struct, drop_train_chem = 0, 0

    for idx, elem_id in enumerate(element_ids):

        valid_struct_edges = [(s, elem_id, d) for s, d in adj_struct_to_elem[elem_id] if d <= D1]
        valid_chem_edges = [(c, elem_id, d) for c, d in adj_chem_to_elem[elem_id] if d <= D2]

        if not valid_struct_edges: drop_train_struct += 1
        if not valid_chem_edges: drop_train_chem += 1

        if not valid_struct_edges or not valid_chem_edges:
            continue

        struct_global_ids = list(set([s for s, t, d in valid_struct_edges]))
        local_src_struct = [{gid: lid for lid, gid in enumerate(struct_global_ids)}[s] for s, t, d in
                            valid_struct_edges]
        local_dst_struct = [0] * len(valid_struct_edges)

        fine_g = dgl.heterograph(
            {('structure', 'structure_to_known', 'known'): (local_src_struct, local_dst_struct)},
            num_nodes_dict={'structure': len(struct_global_ids), 'known': 1})

        struct_feats = torch.stack([node_features['structure'][gid] for gid in struct_global_ids]).squeeze()
        if struct_feats.dim() == 1: struct_feats = struct_feats.unsqueeze(0)
        fine_g.nodes['structure'].data['global_id'] = torch.tensor(struct_global_ids, dtype=torch.int64)
        fine_g.nodes['structure'].data['feat'] = struct_feats
        fine_g.nodes['known'].data['global_id'] = torch.tensor([elem_id], dtype=torch.int64)
        fine_g.nodes['known'].data['feat'] = node_features['known'][elem_id].unsqueeze(0).squeeze(dim=1)
        fine_g.nodes['known'].data['label'] = torch.tensor([label_dict[elem_id]], dtype=torch.int64)


        chem_global_ids = list(set([c for c, t, d in valid_chem_edges]))
        local_src_chem = [{gid: lid for lid, gid in enumerate(chem_global_ids)}[c] for c, t, d in valid_chem_edges]
        local_dst_chem = [0] * len(valid_chem_edges)

        coarse_g = dgl.heterograph(
            {('chemical', 'chemical_to_known', 'known'): (local_src_chem, local_dst_chem)},
            num_nodes_dict={'chemical': len(chem_global_ids), 'known': 1})

        chem_feats = torch.stack([node_features['chemical'][gid] for gid in chem_global_ids]).squeeze()
        if chem_feats.dim() == 1: chem_feats = chem_feats.unsqueeze(0)
        coarse_g.nodes['chemical'].data['global_id'] = torch.tensor(chem_global_ids, dtype=torch.int64)
        coarse_g.nodes['chemical'].data['feat'] = chem_feats
        coarse_g.nodes['known'].data['global_id'] = torch.tensor([elem_id], dtype=torch.int64)
        coarse_g.nodes['known'].data['feat'] = node_features['known'][elem_id].unsqueeze(0).squeeze(dim=1)
        coarse_g.nodes['known'].data['label'] = torch.tensor([label_dict[elem_id]], dtype=torch.int64)

        train_subgraphs['fine'].append(fine_g)
        train_subgraphs['coarse'].append(coarse_g)


    predict_subgraphs = {'fine': [], 'coarse': []}
    unknown_ids = list(node_features['unknown'].keys())

    drop_unk_struct, drop_unk_chem = 0, 0

    for idx, unk_id in enumerate(unknown_ids):

        valid_struct_edges = [(s, unk_id, d) for s, d in adj_struct_to_unk[unk_id] if d <= D1]
        valid_chem_edges = [(c, unk_id, d) for c, d in adj_chem_to_unk[unk_id] if d <= D2]

        if not valid_struct_edges: drop_unk_struct += 1
        if not valid_chem_edges: drop_unk_chem += 1

        if not valid_struct_edges or not valid_chem_edges:
            continue

        struct_global_ids = list(set([s for s, t, d in valid_struct_edges]))
        local_src_struct = [{gid: lid for lid, gid in enumerate(struct_global_ids)}[s] for s, t, d in
                            valid_struct_edges]
        local_dst_struct = [0] * len(valid_struct_edges)

        fine_g = dgl.heterograph(
            {('structure', 'structure_to_unknown', 'unknown'): (local_src_struct, local_dst_struct)},
            num_nodes_dict={'structure': len(struct_global_ids), 'unknown': 1})

        struct_feats = torch.stack([node_features['structure'][gid] for gid in struct_global_ids]).squeeze()
        if struct_feats.dim() == 1: struct_feats = struct_feats.unsqueeze(0)
        fine_g.nodes['structure'].data['global_id'] = torch.tensor(struct_global_ids, dtype=torch.int64)
        fine_g.nodes['structure'].data['feat'] = struct_feats
        fine_g.nodes['unknown'].data['global_id'] = torch.tensor([unk_id], dtype=torch.int64)
        fine_g.nodes['unknown'].data['feat'] = node_features['unknown'][unk_id].unsqueeze(0).squeeze(dim=1)


        chem_global_ids = list(set([c for c, t, d in valid_chem_edges]))
        local_src_chem = [{gid: lid for lid, gid in enumerate(chem_global_ids)}[c] for c, t, d in valid_chem_edges]
        local_dst_chem = [0] * len(valid_chem_edges)

        coarse_g = dgl.heterograph(
            {('chemical', 'chemical_to_unknown', 'unknown'): (local_src_chem, local_dst_chem)},
            num_nodes_dict={'chemical': len(chem_global_ids), 'unknown': 1})

        chem_feats = torch.stack([node_features['chemical'][gid] for gid in chem_global_ids]).squeeze()
        if chem_feats.dim() == 1: chem_feats = chem_feats.unsqueeze(0)
        coarse_g.nodes['chemical'].data['global_id'] = torch.tensor(chem_global_ids, dtype=torch.int64)
        coarse_g.nodes['chemical'].data['feat'] = chem_feats
        coarse_g.nodes['unknown'].data['global_id'] = torch.tensor([unk_id], dtype=torch.int64)
        coarse_g.nodes['unknown'].data['feat'] = node_features['unknown'][unk_id].unsqueeze(0).squeeze(dim=1)

        predict_subgraphs['fine'].append(fine_g)
        predict_subgraphs['coarse'].append(coarse_g)


    return train_subgraphs, predict_subgraphs


def normalize_features(node_features: Dict) -> Dict:

    normalized = {}
    for ntype, feats in node_features.items():
        if not feats:
            normalized[ntype] = feats
            continue

        if ntype in ['known', 'unknown']:
            normalized[ntype] = feats
            continue

        feat_matrix = np.stack([feat.numpy() for feat in feats.values()])
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=1e5, neginf=-1e5)

        mean = np.mean(feat_matrix, axis=0, keepdims=True)
        std = np.std(feat_matrix, axis=0, keepdims=True) + 1e-4

        zero_std_mask = std < 1e-4
        if zero_std_mask.any():
            std[zero_std_mask] = 1.0

        normalized[ntype] = {
            nid: torch.tensor((feat.numpy() - mean) / std, dtype=torch.float32).squeeze()
            for nid, feat in feats.items()
        }
    return normalized


def load_data(
        node_path: str = 'nodes.csv',
        edge_path: str = 'edges.csv',
        label_path: str = 'labels.csv'
) -> Tuple[Dict, Dict, Dict, Dict, int]:


    node_features, min_max_ids, unknown_min_id = load_node_features(node_path)


    edge_dict = load_edges(edge_path)


    label_df = pd.read_csv(
        label_path,
        header=None,
        names=['global_id', 'label'],
        dtype={'global_id': int, 'label': int}
    )
    label_dict = dict(zip(label_df['global_id'], label_df['label']))

    node_features = normalize_features(node_features)

    return node_features, edge_dict, label_dict, min_max_ids, unknown_min_id