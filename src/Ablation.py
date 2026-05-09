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
    print(f"subgraphs are saved to {file_path}")


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
        loaded = load_multiscale_subgraphs(file_path)
        if loaded:
            train_subgraphs, predict_subgraphs = loaded
            print(
                f"done：fine_train_subgraphs{len(train_subgraphs['fine'])}，coarse_train_subgraphs{len(train_subgraphs['coarse'])}")
            return train_subgraphs, predict_subgraphs
        else:
            print("subgraphs not exist, start rebuilding...")

    train_subgraphs, predict_subgraphs = build_multiscale_subgraphs(
        min_max_ids, edge_dict, node_features, label_dict, D1=D1, D2=D2
    )
    save_multiscale_subgraphs((train_subgraphs, predict_subgraphs), file_path)
    print(
        f"done：fine_train_subgraphs{len(train_subgraphs['fine'])}，coarse_train_subgraphs{len(train_subgraphs['coarse'])}")
    return train_subgraphs, predict_subgraphs


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
            nn.Linear(in_feats, hidden_feats), nn.BatchNorm1d(hidden_feats), self.activation,
            nn.Linear(hidden_feats, hidden_feats), nn.Dropout(0.3), nn.Linear(hidden_feats, out_feats)
        )
        self.spatial_attn = nn.Linear(hidden_feats, 1)

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
            nn.Linear(in_feats, hidden_feats), nn.LayerNorm(hidden_feats), self.activation,
            nn.Linear(hidden_feats, hidden_feats), nn.Dropout(0.3), nn.Linear(hidden_feats, out_feats)
        )
        self.anomaly_weight = nn.Parameter(torch.ones(in_feats))

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

    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2,
                 num_layers=4, num_experts=4, gate_temp=3.0, gate_reg_weight=0.1, force_uniform_epochs=5000):
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
            dgl.nn.HeteroGraphConv({'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                                    'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)},
                                   aggregate='mean') for _ in range(num_layers - 1)
        ])
        self.fine_final_conv = dgl.nn.HeteroGraphConv(
            {'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
             'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)}, aggregate='mean')

        self.coarse_convs = nn.ModuleList([
            dgl.nn.HeteroGraphConv({'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                                    'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)},
                                   aggregate='mean') for _ in range(num_layers - 1)
        ])
        self.coarse_final_conv = dgl.nn.HeteroGraphConv(
            {'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
             'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)}, aggregate='mean')

        self.struct_experts = nn.ModuleList(
            [StructuralExpert(hidden_feats, hidden_feats, hidden_feats, expert_id=i) for i in range(num_experts)])
        self.chem_experts = nn.ModuleList(
            [ChemicalExpert(hidden_feats, hidden_feats, hidden_feats, expert_id=i) for i in range(num_experts)])

        self.gate_struct = nn.Sequential(nn.Linear(hidden_feats, hidden_feats // 4), nn.Tanh(),
                                         nn.Linear(hidden_feats // 4, num_experts))
        self.gate_chem = nn.Sequential(nn.Linear(hidden_feats, hidden_feats // 4), nn.Tanh(),
                                       nn.Linear(hidden_feats // 4, num_experts))
        self.gate_scale = nn.Linear(2 * hidden_feats, 1)

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_feats) for _ in range(num_layers - 1)])
        self.residuals = nn.ModuleList([nn.Linear(hidden_feats, hidden_feats) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(0.3)
        self.fusion_enhancer = nn.Sequential(nn.Linear(hidden_feats, hidden_feats), nn.BatchNorm1d(hidden_feats),
                                             nn.GELU())
        self.output_layer = nn.Linear(hidden_feats, out_feats)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, fine_g, coarse_g, is_train=True):
        fine_ntype = 'known' if 'known' in fine_g.ntypes else 'unknown'
        fine_feats = {
            'structure': self.proj_struct(fine_g.nodes['structure'].data['feat']),
            fine_ntype: self.proj_element(fine_g.nodes[fine_ntype].data['feat'])
        }
        h_fine = fine_feats[fine_ntype]
        for i, conv in enumerate(self.fine_convs):
            h = conv(fine_g, fine_feats)[fine_ntype]
            h_fine = h + self.residuals[i](h_fine)
            h_fine = torch.relu(self.bns[i](h_fine))
            h_fine = self.dropout(h_fine) if is_train else h_fine
            fine_feats[fine_ntype] = h_fine
        h_fine = self.fine_feature_norm(self.fine_final_conv(fine_g, fine_feats)[fine_ntype])

        coarse_ntype = 'known' if 'known' in coarse_g.ntypes else 'unknown'
        coarse_feats = {
            'chemical': self.proj_chem(coarse_g.nodes['chemical'].data['feat']),
            coarse_ntype: self.proj_element(coarse_g.nodes[coarse_ntype].data['feat'])
        }
        h_coarse = coarse_feats[coarse_ntype]
        for i, conv in enumerate(self.coarse_convs):
            h = conv(coarse_g, coarse_feats)[coarse_ntype]
            h_coarse = h + self.residuals[i](h_coarse)
            h_coarse = torch.relu(self.bns[i](h_coarse))
            h_coarse = self.dropout(h_coarse) if is_train else h_coarse
            coarse_feats[coarse_ntype] = h_coarse
        h_coarse = self.coarse_feature_norm(self.coarse_final_conv(coarse_g, coarse_feats)[coarse_ntype])

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
                    torch.mean(torch.sum(struct_gate ** 2, dim=1)) + torch.mean(torch.sum(chem_gate ** 2, dim=1)))

        self.h_struct = torch.sum(struct_gate.unsqueeze(-1) * torch.stack(struct_expert_outs, dim=1), dim=1)
        self.h_chem = torch.sum(chem_gate.unsqueeze(-1) * torch.stack(chem_expert_outs, dim=1), dim=1)

        scale_gate = torch.sigmoid(self.gate_scale(torch.cat([self.h_struct, self.h_chem], dim=1)))
        h_fused = scale_gate * self.h_struct + (1 - scale_gate) * self.h_chem
        h_fused = self.fusion_enhancer(h_fused)
        h_fused = self.dropout(h_fused) if is_train else h_fused
        return self.output_layer(h_fused)


class NoMoE_HGCN(nn.Module):

    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2,
                 num_layers=4):
        super().__init__()
        self.proj_struct = nn.Linear(in_feats_struct, hidden_feats)
        self.proj_chem = nn.Linear(in_feats_chem, hidden_feats)
        self.proj_element = nn.Linear(in_feats_element, hidden_feats)

        self.fine_feature_norm = nn.LayerNorm(hidden_feats)
        self.coarse_feature_norm = nn.LayerNorm(hidden_feats)

        self.fine_convs = nn.ModuleList([
            dgl.nn.HeteroGraphConv({'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                                    'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)},
                                   aggregate='mean') for _ in range(num_layers - 1)
        ])
        self.fine_final_conv = dgl.nn.HeteroGraphConv(
            {'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
             'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)}, aggregate='mean')

        self.coarse_convs = nn.ModuleList([
            dgl.nn.HeteroGraphConv({'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                                    'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)},
                                   aggregate='mean') for _ in range(num_layers - 1)
        ])
        self.coarse_final_conv = dgl.nn.HeteroGraphConv(
            {'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
             'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)}, aggregate='mean')

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_feats) for _ in range(num_layers - 1)])
        self.residuals = nn.ModuleList([nn.Linear(hidden_feats, hidden_feats) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(0.3)

        self.fusion_enhancer = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.BatchNorm1d(hidden_feats),
            nn.GELU()
        )
        self.output_layer = nn.Linear(hidden_feats, out_feats)

    @property
    def gate_reg_loss(self):

        return torch.tensor(0.0, device=self.output_layer.weight.device)

    def set_epoch(self, epoch):
        pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fine_ntype = 'known' if 'known' in fine_g.ntypes else 'unknown'
        fine_feats = {
            'structure': self.proj_struct(fine_g.nodes['structure'].data['feat']),
            fine_ntype: self.proj_element(fine_g.nodes[fine_ntype].data['feat'])
        }
        h_fine = fine_feats[fine_ntype]
        for i, conv in enumerate(self.fine_convs):
            h_fine = torch.relu(self.bns[i](conv(fine_g, fine_feats)[fine_ntype] + self.residuals[i](h_fine)))
            fine_feats[fine_ntype] = self.dropout(h_fine) if is_train else h_fine
        h_fine = self.fine_feature_norm(self.fine_final_conv(fine_g, fine_feats)[fine_ntype])

        coarse_ntype = 'known' if 'known' in coarse_g.ntypes else 'unknown'
        coarse_feats = {
            'chemical': self.proj_chem(coarse_g.nodes['chemical'].data['feat']),
            coarse_ntype: self.proj_element(coarse_g.nodes[coarse_ntype].data['feat'])
        }
        h_coarse = coarse_feats[coarse_ntype]
        for i, conv in enumerate(self.coarse_convs):
            h_coarse = torch.relu(self.bns[i](conv(coarse_g, coarse_feats)[coarse_ntype] + self.residuals[i](h_coarse)))
            coarse_feats[coarse_ntype] = self.dropout(h_coarse) if is_train else h_coarse
        h_coarse = self.coarse_feature_norm(self.coarse_final_conv(coarse_g, coarse_feats)[coarse_ntype])

        self.h_struct, self.h_chem = h_fine, h_coarse

        h_fused = torch.cat([h_fine, h_coarse], dim=1)
        h_fused = self.fusion_enhancer(h_fused)
        h_fused = self.dropout(h_fused) if is_train else h_fused
        return self.output_layer(h_fused)


class BaselineGCN(nn.Module):
    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2):
        super().__init__()
        self.proj_struct, self.proj_chem, self.proj_element = nn.Linear(in_feats_struct, hidden_feats), nn.Linear(
            in_feats_chem, hidden_feats), nn.Linear(in_feats_element, hidden_feats)
        self.conv1_fine = dgl.nn.HeteroGraphConv({'structure_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                                                  'structure_to_unknown': dgl.nn.GraphConv(hidden_feats, hidden_feats)},
                                                 aggregate='mean')
        self.conv1_coarse = dgl.nn.HeteroGraphConv({'chemical_to_known': dgl.nn.GraphConv(hidden_feats, hidden_feats),
                                                    'chemical_to_unknown': dgl.nn.GraphConv(hidden_feats,
                                                                                            hidden_feats)},
                                                   aggregate='mean')
        self.classifier = nn.Sequential(nn.Linear(hidden_feats * 2, hidden_feats), nn.BatchNorm1d(hidden_feats),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_feats, out_feats))
        self.h_struct, self.h_chem = None, None

    @property
    def gate_reg_loss(self): return torch.tensor(0.0, device=self.classifier[0].weight.device)

    def set_epoch(self, epoch): pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fn, cn = ('known' if 'known' in fine_g.ntypes else 'unknown'), (
            'known' if 'known' in coarse_g.ntypes else 'unknown')
        ff, cf = {'structure': torch.relu(self.proj_struct(fine_g.nodes['structure'].data['feat'])),
                  fn: torch.relu(self.proj_element(fine_g.nodes[fn].data['feat']))}, {
            'chemical': torch.relu(self.proj_chem(coarse_g.nodes['chemical'].data['feat'])),
            cn: torch.relu(self.proj_element(coarse_g.nodes[cn].data['feat']))}
        h_f, h_c = self.conv1_fine(fine_g, ff)[fn], self.conv1_coarse(coarse_g, cf)[cn]
        self.h_struct, self.h_chem = h_f, h_c
        return self.classifier(torch.cat([h_f, h_c], dim=1))


class BaselineSAGE(nn.Module):
    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2):
        super().__init__()
        self.proj_struct, self.proj_chem, self.proj_element = nn.Linear(in_feats_struct, hidden_feats), nn.Linear(
            in_feats_chem, hidden_feats), nn.Linear(in_feats_element, hidden_feats)
        self.conv1_fine = dgl.nn.HeteroGraphConv(
            {'structure_to_known': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean'),
             'structure_to_unknown': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean')}, aggregate='mean')
        self.conv1_coarse = dgl.nn.HeteroGraphConv(
            {'chemical_to_known': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean'),
             'chemical_to_unknown': dgl.nn.SAGEConv(hidden_feats, hidden_feats, 'mean')}, aggregate='mean')
        self.classifier = nn.Sequential(nn.Linear(hidden_feats * 2, hidden_feats), nn.BatchNorm1d(hidden_feats),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_feats, out_feats))
        self.h_struct, self.h_chem = None, None

    @property
    def gate_reg_loss(self): return torch.tensor(0.0, device=self.classifier[0].weight.device)

    def set_epoch(self, epoch): pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fn, cn = ('known' if 'known' in fine_g.ntypes else 'unknown'), (
            'known' if 'known' in coarse_g.ntypes else 'unknown')
        ff, cf = {'structure': torch.relu(self.proj_struct(fine_g.nodes['structure'].data['feat'])),
                  fn: torch.relu(self.proj_element(fine_g.nodes[fn].data['feat']))}, {
            'chemical': torch.relu(self.proj_chem(coarse_g.nodes['chemical'].data['feat'])),
            cn: torch.relu(self.proj_element(coarse_g.nodes[cn].data['feat']))}
        h_f, h_c = self.conv1_fine(fine_g, ff)[fn], self.conv1_coarse(coarse_g, cf)[cn]
        self.h_struct, self.h_chem = h_f, h_c
        return self.classifier(torch.cat([h_f, h_c], dim=1))


class BaselineGAT(nn.Module):
    def __init__(self, in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=256, out_feats=2):
        super().__init__()
        self.proj_struct, self.proj_chem, self.proj_element = nn.Linear(in_feats_struct, hidden_feats), nn.Linear(
            in_feats_chem, hidden_feats), nn.Linear(in_feats_element, hidden_feats)
        self.conv1_fine = dgl.nn.HeteroGraphConv(
            {'structure_to_known': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1),
             'structure_to_unknown': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1)}, aggregate='mean')
        self.conv1_coarse = dgl.nn.HeteroGraphConv(
            {'chemical_to_known': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1),
             'chemical_to_unknown': dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads=1)}, aggregate='mean')
        self.classifier = nn.Sequential(nn.Linear(hidden_feats * 2, hidden_feats), nn.BatchNorm1d(hidden_feats),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_feats, out_feats))
        self.h_struct, self.h_chem = None, None

    @property
    def gate_reg_loss(self):
        return torch.tensor(0.0, device=self.classifier[0].weight.device)

    def set_epoch(self, epoch):
        pass

    def forward(self, fine_g, coarse_g, is_train=True):
        fn, cn = ('known' if 'known' in fine_g.ntypes else 'unknown'), (
            'known' if 'known' in coarse_g.ntypes else 'unknown')
        ff, cf = {'structure': torch.relu(self.proj_struct(fine_g.nodes['structure'].data['feat'])),
                  fn: torch.relu(self.proj_element(fine_g.nodes[fn].data['feat']))}, {
            'chemical': torch.relu(self.proj_chem(coarse_g.nodes['chemical'].data['feat'])),
            cn: torch.relu(self.proj_element(coarse_g.nodes[cn].data['feat']))}
        h_f, h_c = self.conv1_fine(fine_g, ff)[fn], self.conv1_coarse(coarse_g, cf)[cn]
        if h_f.dim() == 3: h_f = h_f.mean(dim=1)
        if h_c.dim() == 3: h_c = h_c.mean(dim=1)
        self.h_struct, self.h_chem = h_f, h_c
        return self.classifier(torch.cat([h_f, h_c], dim=1))


class MultiscaleFocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2, weight=None, mu=0.001):
        super().__init__()
        self.focal = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        self.alpha, self.gamma, self.base_mu, self.warmup_epochs = alpha, gamma, mu, 50

    def forward(self, inputs, targets, h_struct, h_chem, current_epoch):
        ce_loss = self.focal(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if h_struct.shape != h_chem.shape:
            min_size = min(h_struct.size(0), h_chem.size(0))
            h_struct, h_chem = h_struct[:min_size], h_chem[:min_size]
        consistency_loss = torch.norm(h_struct - h_chem, p=1).mean()

        current_mu = self.base_mu * (
                    current_epoch / self.warmup_epochs) if current_epoch < self.warmup_epochs else self.base_mu
        total_loss = focal_loss + current_mu * consistency_loss
        return focal_loss, consistency_loss, current_mu, total_loss


class Ablation_CE_ConsistencyLoss(nn.Module):

    def __init__(self, weight=None, mu=0.001):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        self.base_mu, self.warmup_epochs = mu, 50

    def forward(self, inputs, targets, h_struct, h_chem, current_epoch):
        loss_main = self.ce_loss(inputs, targets)
        if h_struct.shape != h_chem.shape:
            min_size = min(h_struct.size(0), h_chem.size(0))
            h_struct, h_chem = h_struct[:min_size], h_chem[:min_size]
        consistency_loss = torch.norm(h_struct - h_chem, p=1).mean()

        current_mu = self.base_mu * (
                    current_epoch / self.warmup_epochs) if current_epoch < self.warmup_epochs else self.base_mu
        total_loss = loss_main + current_mu * consistency_loss
        return loss_main, consistency_loss, current_mu, total_loss


def collate_multiscale_fn(samples):
    fine_graphs, coarse_graphs, labels = zip(*samples)
    return dgl.batch(fine_graphs), dgl.batch(coarse_graphs), torch.tensor(labels)


def train_model(params):
    model_type = params.get('model_type', 'NoFocal')
    print(f"\n===== start training: model_type -> [{model_type}] =====")

    experiment_id = nni.get_experiment_id() if nni.get_experiment_id() else "standalone"
    wandb.init(project="MMGCN_Rebuttal_Project", name=f"{model_type}_exp_{experiment_id}", config=params)

    model_save_dir = os.path.join('/model_path_NoFocal/', f"exp_{model_type}_{experiment_id}")
    epoch_model_dir = os.path.join(model_save_dir, "epoch_models")
    os.makedirs(epoch_model_dir, exist_ok=True)

    node_features, edge_dict, label_dict, min_max_ids, unknown_min_id = load_data()
    node_features = normalize_features(node_features)

    train_subgraphs, predict_subgraphs = get_multiscale_subgraphs(
        min_max_ids, edge_dict, node_features, label_dict, build_new=params.get('build_new', False), D1=params['D1'],
        D2=params['D2']
    )

    min_len = min(len(train_subgraphs['fine']), len(train_subgraphs['coarse']))
    train_fine, train_coarse = train_subgraphs['fine'][:min_len], train_subgraphs['coarse'][:min_len]
    labels = [g.nodes['element'].data['label'][0].item() for g in train_fine]

    train_fine, test_fine, train_coarse, test_coarse, train_labels, test_labels = train_test_split(
        train_fine, train_coarse, labels, train_size=0.8, stratify=labels, random_state=42
    )

    train_loader = DataLoader(list(zip(train_fine, train_coarse, train_labels)), batch_size=params['batch_size'],
                              shuffle=True, collate_fn=collate_multiscale_fn)
    test_loader = DataLoader(list(zip(test_fine, test_coarse, test_labels)), batch_size=params['batch_size'],
                             shuffle=False, collate_fn=collate_multiscale_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight = torch.FloatTensor(class_weights).to(device)

    if model_type in ['MMGCN', 'NoConsistency', 'NoFocal']:
        model = MultiscaleMoEHGCN(
            in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=params['hidden_feats'],
            num_layers=params['num_layers'], num_experts=params['num_experts'], gate_temp=params['gate_temp'],
            gate_reg_weight=params['gate_reg_weight'], force_uniform_epochs=params['force_uniform_epochs']
        ).to(device)
    elif model_type == 'NoMoE':
        model = NoMoE_HGCN(in_feats_struct=4, in_feats_chem=7, in_feats_element=5, hidden_feats=params['hidden_feats'],
                           num_layers=params['num_layers']).to(device)
    elif model_type == 'GCN':
        model = BaselineGCN(in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                            hidden_feats=params['hidden_feats']).to(device)
    elif model_type == 'SAGE':
        model = BaselineSAGE(in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                             hidden_feats=params['hidden_feats']).to(device)
    elif model_type == 'GAT':
        model = BaselineGAT(in_feats_struct=4, in_feats_chem=7, in_feats_element=5,
                            hidden_feats=params['hidden_feats']).to(device)

    if model_type == 'NoFocal':
        criterion = Ablation_CE_ConsistencyLoss(weight=class_weight, mu=params['consistency_mu'])
    else:
        current_mu = 0.0 if model_type == 'NoConsistency' else params['consistency_mu']
        if model_type == 'NoConsistency': print(">>  [NoConsistency]")
        criterion = MultiscaleFocalLoss(alpha=params['focal_alpha'], gamma=params['focal_gamma'], weight=class_weight,
                                        mu=current_mu)

    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)

    best_test_acc = 0.0
    for epoch in range(params['epochs']):
        model.set_epoch(epoch)
        model.train()
        train_loss, train_preds, train_true_labels = 0.0, [], []
        total_focal_loss, total_consistency_loss = 0.0, 0.0

        for fine_g, coarse_g, labels in train_loader:
            fine_g, coarse_g, labels = fine_g.to(device), coarse_g.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(fine_g, coarse_g, is_train=True)

            loss_focal, loss_consistency, current_mu, loss_main = criterion(outputs, labels, model.h_struct,
                                                                            model.h_chem, epoch)
            loss = loss_main + model.gate_reg_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            total_focal_loss += loss_focal.item()
            total_consistency_loss += loss_consistency.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_true_labels, train_preds)

        model.eval()
        test_loss, test_preds, test_true_labels = 0.0, [], []
        with torch.no_grad():
            for fine_g, coarse_g, labels in test_loader:
                fine_g, coarse_g, labels = fine_g.to(device), coarse_g.to(device), labels.to(device)
                outputs = model(fine_g, coarse_g, is_train=False)
                loss_f, loss_c, _, loss_m = criterion(outputs, labels, model.h_struct, model.h_chem, epoch)
                test_loss += loss_m.item() + model.gate_reg_loss.item()
                test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_true_labels, test_preds)
        test_f1 = f1_score(test_true_labels, test_preds, average='weighted')

        wandb.log({
            'epoch': epoch + 1,
            'Accuracy/Train': train_acc, 'Accuracy/Test': test_acc,
            'Loss/Train': train_loss / len(train_loader),
            'Learning_Rate': optimizer.param_groups[0]['lr']
        })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
        scheduler.step()

    wandb.finish()
    print(f"[{model_type}] done，best_test_acc: {best_test_acc:.4f}")
    return best_test_acc


if __name__ == "__main__":
    params = {
        # ==========================================================
        # 'MMGCN'
        # 'NoMoE'
        # 'NoConsistency'
        # 'NoFocal'
        # 'GCN' / 'SAGE' / 'GAT'
        # ==========================================================
        'model_type': 'NoFocal',

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
        'epochs': 200,
        'build_new': False  
    }

    optimized_params = nni.get_next_parameter()
    params = merge_parameter(params, optimized_params)
    train_model(params)