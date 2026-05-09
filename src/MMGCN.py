import os
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import pandas as pd
from Data_loader import load_data, build_multiscale_subgraphs, normalize_features
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import nni
from nni.utils import merge_parameter
import wandb
import matplotlib as mpl
import matplotlib.font_manager as fm
import pickle


def save_multiscale_subgraphs(subgraphs, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(subgraphs, f)


def load_multiscale_subgraphs(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"fail: {e}")
    print(f" {file_path} not exist")
    return None


def get_multiscale_subgraphs(
        min_max_ids, edge_dict, node_features, label_dict,
        build_new=True, file_path='multiscale_subgraphs.pkl',
        D1=800, D2=1000
):
    if not build_new:
        if build_new:
            train_subgraphs, predict_subgraphs = build_multiscale_subgraphs(
                min_max_ids, edge_dict, node_features, label_dict, D1=D1, D2=D2
            )
            save_multiscale_subgraphs((train_subgraphs, predict_subgraphs), file_path)
            print(
                f"done：fine_train_subgraphs{len(train_subgraphs['fine'])}，coarse_train_subgraphs{len(train_subgraphs['coarse'])}")
        else:
            loaded = load_multiscale_subgraphs(file_path)
            if loaded:
                train_subgraphs, predict_subgraphs = loaded
                print(
                    f"done：fine_train_subgraphs{len(train_subgraphs['fine'])}，coarse_train_subgraphs{len(train_subgraphs['coarse'])}")
            else:
                train_subgraphs, predict_subgraphs = build_multiscale_subgraphs(
                    min_max_ids, edge_dict, node_features, label_dict, D1=D1, D2=D2
                )
                save_multiscale_subgraphs((train_subgraphs, predict_subgraphs), file_path)
                print(
                    f"subgraphs not exist, rebuild done：fine_train_subgraphs{len(train_subgraphs['fine'])}，coarse_train_subgraphs{len(train_subgraphs['coarse'])}")
    else:
        train_subgraphs, predict_subgraphs = build_multiscale_subgraphs(
            min_max_ids, edge_dict, node_features, label_dict, D1=D1, D2=D2
        )
        save_multiscale_subgraphs((train_subgraphs, predict_subgraphs), file_path)
        print(
            f"done：fine_train_subgraphs{len(train_subgraphs['fine'])}，coarse_train_subgraphs{len(train_subgraphs['coarse'])}")


class StructuralExpert(nn.Module):

    def __init__(self, in_feats, hidden_feats, out_feats, expert_id, use_noise=True):
        super().__init__()
        self.expert_id = expert_id
        self.use_noise = use_noise
        if expert_id == 0:
            self.activation = nn.LeakyReLU(0.01)
        else:
            activations = [nn.ReLU(), nn.GELU(), nn.SiLU(), nn.LeakyReLU(0.2)]
            self.activation = activations[expert_id % len(activations)]

        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            self.activation,
            nn.Linear(hidden_feats, hidden_feats),
            nn.Dropout(0.3),
            nn.Linear(hidden_feats, out_feats)
        )
        self.spatial_attn = nn.Linear(hidden_feats, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.expert_id == 0:
                    nn.init.xavier_uniform_(m.weight, gain=0.3)
                    nn.init.normal_(m.bias, mean=0.0, std=0.003)
                else:
                    gain = 1.2 + (self.expert_id * 0.2)
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                    nn.init.normal_(m.bias, mean=0.0, std=0.01 + self.expert_id * 0.01)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.size(-1))

        x = self.net[:3](x)
        attn = torch.sigmoid(self.spatial_attn(x))
        x = x * attn
        x = self.net[3:](x)

        if self.expert_id != 0 and self.use_noise and self.training:
            x = x + 0.1 * torch.randn_like(x)
        return x


class ChemicalExpert(nn.Module):

    def __init__(self, in_feats, hidden_feats, out_feats, expert_id, use_noise=True):
        super().__init__()
        self.expert_id = expert_id
        self.use_noise = use_noise
        if expert_id == 0:
            self.activation = nn.LeakyReLU(0.01)
        else:
            activations = [nn.ReLU(), nn.GELU(), nn.SiLU(), nn.LeakyReLU(0.2)]
            self.activation = activations[expert_id % len(activations)]

        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.LayerNorm(hidden_feats),
            self.activation,
            nn.Linear(hidden_feats, hidden_feats),
            nn.Dropout(0.3),
            nn.Linear(hidden_feats, out_feats)
        )
        self.anomaly_weight = nn.Parameter(torch.ones(in_feats))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.expert_id == 0:
                    nn.init.xavier_uniform_(m.weight, gain=0.3)
                    nn.init.normal_(m.bias, mean=0.0, std=0.003)
                else:
                    gain = 1.2 + (self.expert_id * 0.2)
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                    nn.init.normal_(m.bias, mean=0.0, std=0.01 + self.expert_id * 0.01)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.view(-1, x.size(-1))

        x = x * self.anomaly_weight
        x = self.net(x)

        if self.expert_id != 0 and self.use_noise and self.training:
            x = x + 0.1 * torch.randn_like(x)
        return x


class MultiscaleMoEHGCN(nn.Module):
    def __init__(self,
                 in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                 hidden_feats=512, out_feats=2, num_layers=4,
                 num_experts=4, gate_temp=3.0,
                 gate_reg_weight=0.1,
                 force_uniform_epochs=5000):
        super().__init__()
        self.hidden_feats = hidden_feats
        self.num_experts = num_experts
        self.gate_temp = gate_temp
        self.gate_reg_weight = gate_reg_weight
        self.force_uniform_epochs = force_uniform_epochs
        self.current_epoch = 0

        self.proj_struct = nn.Linear(in_feats_struct, hidden_feats)
        self.proj_chem = nn.Linear(in_feats_chem, hidden_feats)
        self.proj_element = nn.Linear(in_feats_element, hidden_feats)

        self.fine_feature_norm = nn.LayerNorm(hidden_feats)
        self.coarse_feature_norm = nn.LayerNorm(hidden_feats)

        self.fine_convs = nn.ModuleList([
            dgl.nn.HeteroGraphConv({
                'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)
            }, aggregate='mean') for _ in range(num_layers - 1)
        ])
        self.fine_final_conv = dgl.nn.HeteroGraphConv({
            'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
            'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)
        }, aggregate='mean')

        self.coarse_convs = nn.ModuleList([
            dgl.nn.HeteroGraphConv({
                'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)
            }, aggregate='mean') for _ in range(num_layers - 1)
        ])
        self.coarse_final_conv = dgl.nn.HeteroGraphConv({
            'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
            'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)
        }, aggregate='mean')

        self.struct_experts = nn.ModuleList([
            StructuralExpert(hidden_feats, hidden_feats, hidden_feats, expert_id=i, use_noise=True)
            for i in range(num_experts)
        ])
        self.chem_experts = nn.ModuleList([
            ChemicalExpert(hidden_feats, hidden_feats, hidden_feats, expert_id=i, use_noise=True)
            for i in range(num_experts)
        ])

        self.gate_struct = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 4),
            nn.Tanh(),
            nn.Linear(hidden_feats // 4, num_experts)
        )
        self.gate_chem = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 4),
            nn.Tanh(),
            nn.Linear(hidden_feats // 4, num_experts)
        )

        self.gate_scale = nn.Linear(2 * hidden_feats, 1)

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_feats) for _ in range(num_layers - 1)])
        self.residuals = nn.ModuleList([nn.Linear(hidden_feats, hidden_feats) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(0.3)
        self.fusion_enhancer = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.GELU()
        )
        self.output_layer = nn.Linear(hidden_feats, out_feats)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, fine_g, coarse_g, is_train=True):
        fine_ntype = 'known' if 'known' in fine_g.ntypes else 'unknown'
        fine_feats = {
            'structure': self.proj_struct(fine_g.nodes['structure'].data['feat']),
            fine_ntype: self.proj_element(fine_g.nodes[fine_ntype].data['feat'])  # 处理0值特征
        }

        if is_train and torch.rand(1) < 0.1:
            print(f"fine_feats_mean: {fine_feats[fine_ntype].mean().item():.6f}")

        h_fine = fine_feats[fine_ntype]

        for i, conv in enumerate(self.fine_convs):
            h = conv(fine_g, fine_feats)
            h_node = h[fine_ntype]
            h_fine = h_node + self.residuals[i](h_fine)
            h_fine = self.bns[i](h_fine)
            h_fine = torch.relu(h_fine)
            h_fine = self.dropout(h_fine) if is_train else h_fine
            fine_feats[fine_ntype] = h_fine

        h_final = self.fine_final_conv(fine_g, fine_feats)
        h_fine = h_final[fine_ntype]
        h_fine = self.fine_feature_norm(h_fine)

        coarse_ntype = 'known' if 'known' in coarse_g.ntypes else 'unknown'
        coarse_feats = {
            'chemical': self.proj_chem(coarse_g.nodes['chemical'].data['feat']),
            coarse_ntype: self.proj_element(coarse_g.nodes[coarse_ntype].data['feat'])
        }

        h_coarse = coarse_feats[coarse_ntype]

        for i, conv in enumerate(self.coarse_convs):
            h = conv(coarse_g, coarse_feats)
            h_node = h[coarse_ntype]
            h_coarse = h_node + self.residuals[i](h_coarse)
            h_coarse = self.bns[i](h_coarse)
            h_coarse = torch.relu(h_coarse)
            h_coarse = self.dropout(h_coarse) if is_train else h_coarse
            coarse_feats[coarse_ntype] = h_coarse

        h_final_coarse = self.coarse_final_conv(coarse_g, coarse_feats)
        h_coarse = h_final_coarse[coarse_ntype]
        h_coarse = self.coarse_feature_norm(h_coarse)

        struct_expert_outs = [expert(h_fine) for expert in self.struct_experts]
        chem_expert_outs = [expert(h_coarse) for expert in self.chem_experts]

        struct_gate_logits = self.gate_struct(h_fine) / self.gate_temp
        chem_gate_logits = self.gate_chem(h_coarse) / self.gate_temp

        if is_train and self.current_epoch < self.force_uniform_epochs:
            struct_gate = torch.ones_like(struct_gate_logits) / self.num_experts
            chem_gate = torch.ones_like(chem_gate_logits) / self.num_experts
        else:
            struct_gate = torch.softmax(struct_gate_logits, dim=1)
            chem_gate = torch.softmax(chem_gate_logits, dim=1)

        self.gate_reg_loss = self.gate_reg_weight * (
                torch.mean(torch.sum(struct_gate ** 2, dim=1)) +
                torch.mean(torch.sum(chem_gate ** 2, dim=1))
        )

        self.h_struct = torch.sum(struct_gate.unsqueeze(-1) * torch.stack(struct_expert_outs, dim=1), dim=1)
        self.h_chem = torch.sum(chem_gate.unsqueeze(-1) * torch.stack(chem_expert_outs, dim=1), dim=1)

        scale_gate = torch.sigmoid(self.gate_scale(torch.cat([self.h_struct, self.h_chem], dim=1)))
        h_fused = scale_gate * self.h_struct + (1 - scale_gate) * self.h_chem
        h_fused = self.fusion_enhancer(h_fused)
        h_fused = self.dropout(h_fused) if is_train else h_fused

        return self.output_layer(h_fused)


# =====================================================================
#  (Baseline Models: GCN, GraphSAGE, GAT)
# =====================================================================

class BaselineGCN(nn.Module):

    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2):
        super().__init__()
        self.proj_struct = nn.Linear(in_feats_struct, hidden_feats)
        self.proj_chem = nn.Linear(in_feats_chem, hidden_feats)
        self.proj_element = nn.Linear(in_feats_element, hidden_feats)

        self.conv1_fine = dgl.nn.HeteroGraphConv({
            'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
            'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)
        }, aggregate='mean')
        self.conv1_coarse = dgl.nn.HeteroGraphConv({
            'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
            'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)
        }, aggregate='mean')

        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_feats, out_feats)
        )
        self.h_struct = None
        self.h_chem = None

    @property
    def gate_reg_loss(self):

        return torch.tensor(0.0, device=self.classifier[0].weight.device)

    def set_epoch(self, epoch):
        pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fine_ntype = 'known' if 'known' in fine_g.ntypes else 'unknown'
        coarse_ntype = 'known' if 'known' in coarse_g.ntypes else 'unknown'

        fine_feats = {
            'structure': torch.relu(self.proj_struct(fine_g.nodes['structure'].data['feat'])),
            fine_ntype: torch.relu(self.proj_element(fine_g.nodes[fine_ntype].data['feat']))
        }
        coarse_feats = {
            'chemical': torch.relu(self.proj_chem(coarse_g.nodes['chemical'].data['feat'])),
            coarse_ntype: torch.relu(self.proj_element(coarse_g.nodes[coarse_ntype].data['feat']))
        }

        h_fine = self.conv1_fine(fine_g, fine_feats)[fine_ntype]
        h_coarse = self.conv1_coarse(coarse_g, coarse_feats)[coarse_ntype]

        self.h_struct = h_fine
        self.h_chem = h_coarse

        h_fused = torch.cat([h_fine, h_coarse], dim=1)
        return self.classifier(h_fused)


class BaselineSAGE(nn.Module):

    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2):
        super().__init__()
        self.proj_struct = nn.Linear(in_feats_struct, hidden_feats)
        self.proj_chem = nn.Linear(in_feats_chem, hidden_feats)
        self.proj_element = nn.Linear(in_feats_element, hidden_feats)

        self.conv1_fine = dgl.nn.HeteroGraphConv({
            'structure_to_known': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean'),
            'structure_to_unknown': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean')
        }, aggregate='mean')
        self.conv1_coarse = dgl.nn.HeteroGraphConv({
            'chemical_to_known': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean'),
            'chemical_to_unknown': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean')
        }, aggregate='mean')

        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_feats, out_feats)
        )
        self.h_struct, self.h_chem = None, None

    @property
    def gate_reg_loss(self): return torch.tensor(0.0, device=self.classifier[0].weight.device)

    def set_epoch(self, epoch): pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fine_ntype = 'known' if 'known' in fine_g.ntypes else 'unknown'
        coarse_ntype = 'known' if 'known' in coarse_g.ntypes else 'unknown'

        fine_feats = {
            'structure': torch.relu(self.proj_struct(fine_g.nodes['structure'].data['feat'])),
            fine_ntype: torch.relu(self.proj_element(fine_g.nodes[fine_ntype].data['feat']))
        }
        coarse_feats = {
            'chemical': torch.relu(self.proj_chem(coarse_g.nodes['chemical'].data['feat'])),
            coarse_ntype: torch.relu(self.proj_element(coarse_g.nodes[coarse_ntype].data['feat']))
        }

        h_fine = self.conv1_fine(fine_g, fine_feats)[fine_ntype]
        h_coarse = self.conv1_coarse(coarse_g, coarse_feats)[coarse_ntype]

        self.h_struct, self.h_chem = h_fine, h_coarse
        return self.classifier(torch.cat([h_fine, h_coarse], dim=1))


class BaselineGAT(nn.Module):

    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2):
        super().__init__()
        self.proj_struct = nn.Linear(in_feats_struct, hidden_feats)
        self.proj_chem = nn.Linear(in_feats_chem, hidden_feats)
        self.proj_element = nn.Linear(in_feats_element, hidden_feats)

        self.conv1_fine = dgl.nn.HeteroGraphConv({
            'structure_to_known': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1),
            'structure_to_unknown': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1)
        }, aggregate='mean')
        self.conv1_coarse = dgl.nn.HeteroGraphConv({
            'chemical_to_known': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1),
            'chemical_to_unknown': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1)
        }, aggregate='mean')

        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_feats, out_feats)
        )
        self.h_struct, self.h_chem = None, None

    @property
    def gate_reg_loss(self):
        return torch.tensor(0.0, device=self.classifier[0].weight.device)

    def set_epoch(self, epoch):
        pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fine_ntype = 'known' if 'known' in fine_g.ntypes else 'unknown'
        coarse_ntype = 'known' if 'known' in coarse_g.ntypes else 'unknown'

        fine_feats = {
            'structure': torch.relu(self.proj_struct(fine_g.nodes['structure'].data['feat'])),
            fine_ntype: torch.relu(self.proj_element(fine_g.nodes[fine_ntype].data['feat']))
        }
        coarse_feats = {
            'chemical': torch.relu(self.proj_chem(coarse_g.nodes['chemical'].data['feat'])),
            coarse_ntype: torch.relu(self.proj_element(coarse_g.nodes[coarse_ntype].data['feat']))
        }

        h_fine = self.conv1_fine(fine_g, fine_feats)[fine_ntype]
        h_coarse = self.conv1_coarse(coarse_g, coarse_feats)[coarse_ntype]

        if h_fine.dim() == 3: h_fine = h_fine.mean(dim=1)
        if h_coarse.dim() == 3: h_coarse = h_coarse.mean(dim=1)

        self.h_struct, self.h_chem = h_fine, h_coarse
        return self.classifier(torch.cat([h_fine, h_coarse], dim=1))


# =====================================================================

class MultiscaleFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, weight=None, mu=0.001):
        super().__init__()
        self.focal = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        self.alpha = alpha
        self.gamma = gamma
        self.base_mu = mu
        self.warmup_epochs = 50

    def forward(self, inputs, targets, h_struct, h_chem, current_epoch):

        ce_loss = self.focal(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if h_struct.shape != h_chem.shape:
            min_size = min(h_struct.size(0), h_chem.size(0))
            h_struct = h_struct[:min_size]
            h_chem = h_chem[:min_size]
        consistency_loss = torch.norm(h_struct - h_chem, p=1).mean()

        if current_epoch < self.warmup_epochs:
            current_mu = self.base_mu * (current_epoch / self.warmup_epochs)
        else:
            current_mu = self.base_mu

        total_loss = focal_loss + current_mu * consistency_loss
        return focal_loss, consistency_loss, current_mu, total_loss

def collate_multiscale_fn(samples):
    fine_graphs, coarse_graphs, labels = zip(*samples)
    batched_fine = dgl.batch(fine_graphs)
    batched_coarse = dgl.batch(coarse_graphs)
    return batched_fine, batched_coarse, torch.tensor(labels)


def train_model(params):

    experiment_id = nni.get_experiment_id()

    wandb.init(
        project="MMGCN_Rebuttal_Project",
        name=f"nni_exp_{experiment_id}",
        config=params
    )

    model_save_dir = os.path.join('/MMGCN/', f"exp_{experiment_id}")  # todo
    os.makedirs(model_save_dir, exist_ok=True)

    epoch_model_dir = os.path.join(model_save_dir, "epoch_models")
    os.makedirs(epoch_model_dir, exist_ok=True)

    node_features, edge_dict, label_dict, min_max_ids, unknown_min_id = load_data()
    node_features = normalize_features(node_features)

    sample_elem_id = next(iter(node_features['known'].keys()))
    print(f"node_features-mean: {node_features['known'][sample_elem_id].mean().item():.6f}")
    print(f"build_new: {params.get('build_new', True)}")

    train_subgraphs, predict_subgraphs = get_multiscale_subgraphs(
        min_max_ids, edge_dict, node_features, label_dict,
        build_new=params.get('build_new', True),
        D1=params['D1'], D2=params['D2']
    )

    min_len = min(len(train_subgraphs['fine']), len(train_subgraphs['coarse']))
    train_fine = train_subgraphs['fine'][:min_len]
    train_coarse = train_subgraphs['coarse'][:min_len]

    labels = [g.nodes['known'].data['label'][0].item() for g in train_fine]
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"label {label}：{count}，ratio {count / len(labels):.2%}")

    train_fine, test_fine, train_coarse, test_coarse, train_labels, test_labels = train_test_split(
        train_fine, train_coarse, labels, train_size=0.8, stratify=labels, random_state=42
    )

    train_dataset = list(zip(train_fine, train_coarse, train_labels))
    test_dataset = list(zip(test_fine, test_coarse, test_labels))

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                              collate_fn=collate_multiscale_fn)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                             collate_fn=collate_multiscale_fn)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    class_weight = torch.FloatTensor(class_weights).to(device)
    print(f"class_weight: {class_weight}")


    model_type = params.get('model_type', 'MMGCN')
    print(f"\n model_type: {model_type}")

    if model_type == 'MMGCN':
        model = MultiscaleMoEHGCN(
            in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
            hidden_feats=params['hidden_feats'], num_layers=params['num_layers'],
            num_experts=params['num_experts'], gate_temp=params['gate_temp'],
            gate_reg_weight=params['gate_reg_weight'], force_uniform_epochs=params['force_uniform_epochs']
        ).to(device)
    elif model_type == 'GCN':
        model = BaselineGCN(in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                            hidden_feats=params['hidden_feats']).to(device)
    elif model_type == 'SAGE':
        model = BaselineSAGE(in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                             hidden_feats=params['hidden_feats']).to(device)
    elif model_type == 'GAT':
        model = BaselineGAT(in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                            hidden_feats=params['hidden_feats']).to(device)
    else:
        raise ValueError(f"unknown_model: {model_type}")

    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)
    criterion = MultiscaleFocalLoss(
        alpha=params['focal_alpha'], gamma=params['focal_gamma'],
        weight=class_weight, mu=params['consistency_mu']
    )

    best_test_acc = 0.0
    for epoch in range(params['epochs']):
        model.set_epoch(epoch)
        model.train()
        print(f"\nEpoch {epoch + 1}/{params['epochs']} - training mode: {'open' if model.training else 'close'}")

        train_loss, train_preds = 0.0, []
        train_true_labels = []
        batch_count = 0
        total_focal_loss, total_consistency_loss, total_gate_loss = 0.0, 0.0, 0.0
        total_current_mu = 0.0

        for fine_g, coarse_g, labels in train_loader:
            batch_count += 1
            fine_g, coarse_g, labels = fine_g.to(device), coarse_g.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(fine_g, coarse_g, is_train=True)
            h_struct = model.h_struct
            h_chem = model.h_chem

            loss_focal, loss_consistency, current_mu, loss_main = criterion(
                outputs, labels, h_struct, h_chem, current_epoch=epoch
            )

            loss = loss_main + model.gate_reg_loss
            loss.backward()

            total_norm = 0.0
            grad_check = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    if 'proj_element' in name or 'fine_convs' in name:
                        grad_check[name] = param_norm.item()
            total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            total_focal_loss += loss_focal.item()
            total_consistency_loss += loss_consistency.item()
            total_gate_loss += model.gate_reg_loss.item()
            total_current_mu += current_mu

            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())


        avg_train_loss = train_loss / len(train_loader)
        avg_focal = total_focal_loss / len(train_loader)
        avg_consistency = total_consistency_loss / len(train_loader)
        avg_gate = total_gate_loss / len(train_loader)
        avg_mu = total_current_mu / len(train_loader)

        train_acc = accuracy_score(train_true_labels, train_preds)
        train_f1 = f1_score(train_true_labels, train_preds, average='weighted')

        model.eval()
        test_loss, test_preds = 0.0, []
        test_true_labels = []
        test_focal, test_consistency, test_mu = 0.0, 0.0, 0.0

        with torch.no_grad():
            for fine_g, coarse_g, labels in test_loader:
                fine_g, coarse_g, labels = fine_g.to(device), coarse_g.to(device), labels.to(device)
                outputs = model(fine_g, coarse_g, is_train=False)

                h_struct_test = model.h_struct
                h_chem_test = model.h_chem

                loss_f, loss_c, current_mu_t, loss_m = criterion(
                    outputs, labels, h_struct_test, h_chem_test, current_epoch=epoch
                )

                test_loss += loss_m.item() + model.gate_reg_loss.item()
                test_focal += loss_f.item()
                test_consistency += loss_c.item()
                test_mu += current_mu_t

                test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())  # 测试集同步收集标签

        avg_test_loss = test_loss / len(test_loader)
        avg_test_focal = test_focal / len(test_loader)
        avg_test_consistency = test_consistency / len(test_loader)
        avg_test_mu = test_mu / len(test_loader)

        test_acc = accuracy_score(test_true_labels, test_preds)
        test_f1 = f1_score(test_true_labels, test_preds, average='weighted')

        # W&B
        wandb.log({
            'epoch': epoch + 1,
            'Loss/Train': avg_train_loss,
            'Loss/Test': avg_test_loss,
            'Focal_Loss/Train': avg_focal,
            'Focal_Loss/Test': avg_test_focal,
            'Accuracy/Train': train_acc,
            'Accuracy/Test': test_acc,
            'Consistency/Mu': avg_mu,
            'Learning_Rate': optimizer.param_groups[0]['lr']
        })

        nni.report_intermediate_result({"test_acc": test_acc, "test_f1": test_f1})


        epoch_model_path = os.path.join(epoch_model_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"save epoch {epoch + 1} to: {epoch_model_path}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"bestmodel，Test Acc: {best_test_acc:.4f}，to: {best_model_path}")

        scheduler.step()

    wandb.finish()

    nni.report_final_result({"best_test_acc": best_test_acc})
    print(f"end，best_test_acc: {best_test_acc:.4f}")
    print(f"all epoch models are saved to: {epoch_model_dir}")
    return best_test_acc


if __name__ == "__main__":
    params = {
        'model_type': 'MMGCN',  # 'MMGCN', 'GCN', 'SAGE', 'GAT'
        'learning_rate': 0.0001,
        'batch_size': 64,
        'hidden_feats': 256,
        'num_layers': 4,
        'num_experts': 4,
        'gate_temp': 3.0,
        'gate_reg_weight': 0.005,
        'force_uniform_epochs': 5000,
        'D1': 800,
        'D2': 1000,
        'focal_alpha': 0.6,
        'focal_gamma': 1,
        'consistency_mu': 0.00001,
        'weight_decay': 1e-4,
        'epochs': 500,
        'build_new': Ture
    }

    optimized_params = nni.get_next_parameter()
    params = merge_parameter(params, optimized_params)

    train_model(params)