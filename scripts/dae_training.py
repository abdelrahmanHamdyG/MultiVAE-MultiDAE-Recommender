
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from Dataset import Val_Dataset,DAE_Dataset
import numpy as np
from models.multi_vae import MultiVAE
from models.multi_dae import MultiDAE

import pandas as pd 

import matplotlib.pyplot as plt 


def ndcg_k(pred_scores, ground_truth, k=100):
    """
    pred_scores: 1D numpy array of predicted scores for all items
    ground_truth: set of item indices actually interacted with
    """
    top_k = np.argsort(-pred_scores)[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def recall_at_k(pred_scores, ground_truth, k=50):
    """
    pred_scores: 1D numpy array of predicted scores for all items
    ground_truth: set of item indices actually interacted with
    """
    top_k = np.argsort(-pred_scores)[:k]
    hits = len(set(top_k) & ground_truth)
    total_relevant = len(ground_truth)
    return hits / total_relevant if total_relevant > 0 else 0.0

def val_collate_fn(batch):
    batch_x = torch.stack([x for x, _, _ in batch])
    batch_targets = [target for _, target, _ in batch]  # keep as list of variable-length lists
    batch_users = [user for _, _, user in batch]
    return batch_x, batch_targets, batch_users


def evaluate(model, val_loader, device):
    model.eval()
    ndcgs = []
    recalls = []

    with torch.no_grad():
        for batch_x, batch_targets, _ in val_loader:
            batch_x = batch_x.to(device)

            output = model(batch_x)

            scores = output.cpu().numpy()

            for i in range(batch_x.size(0)):
                seen_items = batch_x[i].nonzero().squeeze().cpu().numpy()
                if seen_items.ndim == 0:
                    seen_items = [seen_items]
                scores[i][seen_items] = -np.inf

                gt_items = set(batch_targets[i])
                ndcgs.append(ndcg_k(scores[i], gt_items, k=100))
                recalls.append(recall_at_k(scores[i], gt_items, k=50))

    mean_ndcg = np.mean(ndcgs)
    mean_recall = np.mean(recalls)

    print(f"Mean NDCG@100: {mean_ndcg:.4f} | Mean Recall@50: {mean_recall:.4f}")
    return mean_ndcg, mean_recall

def dae_loss_function(x, recon_x):
    log_softmax = F.log_softmax(recon_x, dim=1)
    nll = -torch.sum(log_softmax * x, dim=1).mean()
    return nll

def main():
    # Load data
    train_data = pd.read_csv('processed/train.csv')
    val_tr = pd.read_csv('processed/val_tr.csv')
    val_te = pd.read_csv('processed/val_te.csv')

    num_users = train_data['user'].nunique()
    num_items = train_data['movie'].nunique()

    os.makedirs("checkpoints_dae", exist_ok=True)

    # Prepare datasets
    train_dataset = DAE_Dataset(num_users, num_items, train_data)
    val_dataset = Val_Dataset(num_items, val_tr, val_te)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=val_collate_fn)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MultiDAE(num_items,hidden_dim=600,latent_dim=200,drop_out_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.01)

    # Training loop
    num_epochs = 200
    total_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = dae_loss_function(batch, output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        total_losses.append(avg_loss)

        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            evaluate(model, val_loader, device)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f'checkpoints_dae/model_epoch_{epoch+1}.pth')

            plt.figure()
            plt.plot(range(1, len(total_losses)+1), total_losses)
            plt.title("Training Loss - DAE")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(f'loss_plot_dae_epoch_{epoch+1}.png')
            plt.close()


if __name__ == '__main__':
    main()
