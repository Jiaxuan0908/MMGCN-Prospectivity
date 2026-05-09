import os
import torch
import dgl
import numpy as np
import pandas as pd
from Data_loader import load_data
from MMGCN import MultiscaleMoEHGCN, get_multiscale_subgraphs


def predict(model_path,
            node_path='nodes.csv',
            edge_path='edges.csv',
            label_path='labels.csv',
            subgraph_file='multiscale_subgraphs.pkl',
            output_csv='prediction_results.csv',
            D1=800,
            D2=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    node_features, edge_dict, label_dict, min_max_ids, unknown_min_id = load_data(
        node_path=node_path,
        edge_path=edge_path,
        label_path=label_path
    )

    _, predict_subgraphs = get_multiscale_subgraphs(
        min_max_ids, edge_dict, node_features, label_dict,
        build_new=False,
        file_path=subgraph_file,
        D1=D1, D2=D2
    )

    model = MultiscaleMoEHGCN(
        in_feats_struct=4,
        in_feats_chem=7,
        in_feats_element=5,
        hidden_feats=256,
        num_layers=4,
        num_experts=4,
        gate_temp=3.0
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results = []
    min_len = min(len(predict_subgraphs['fine']), len(predict_subgraphs['coarse']))

    if min_len == 0:
        print("\n warning：without any predict_subgraph！")
        return

    with torch.no_grad():
        for i in range(min_len):
            fine_g = predict_subgraphs['fine'][i].to(device)
            coarse_g = predict_subgraphs['coarse'][i].to(device)

            global_id = fine_g.nodes['unknown'].data['global_id'].item()

            outputs = model(fine_g, coarse_g, is_train=False)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            probability = float(probs[1])

            results.append({
                'global_id': global_id,
                'probability': round(probability, 6)
            })
            # ==============================

    df = pd.DataFrame(results)

    df = df[['global_id', 'probability']]
    df.to_csv(output_csv, index=False)
    print(f"done！probability is saved to: {output_csv}")


if __name__ == "__main__":
    predict(
        model_path='/epoch_models/model_epoch_210.pth',
        D1=800,
        D2=1000
    )