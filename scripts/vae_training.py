import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from Dataset import VAE_Dataset,Val_Dataset
import numpy as np
from models.multi_vae import MultiVAE

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
            output, _, _ = model(batch_x)
            scores = output.cpu().numpy()

            for i in range(batch_x.size(0)):
                seen_items = batch_x[i].nonzero().squeeze().cpu().numpy()
                if seen_items.ndim == 0:
                    seen_items = [seen_items]
                scores[i][seen_items] = -np.inf

                gt_items = set(batch_targets[i])  # still works
                ndcgs.append(ndcg_k(scores[i], gt_items, k=100))
                recalls.append(recall_at_k(scores[i], gt_items, k=50))

    mean_ndcg = np.mean(ndcgs)
    mean_recall = np.mean(recalls)

    print(f"Mean NDCG@100: {mean_ndcg:.4f} | Mean Recall@50: {mean_recall:.4f}")
    return mean_ndcg, mean_recall


def loss_function(x,recon_x,mu,logvar,beta=1.0):
    log_softmax= F.log_softmax(recon_x,dim=1)
    
    recon_loss= -torch.sum(log_softmax*x,dim=1).mean()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


    return recon_loss + beta * kl_loss, recon_loss, kl_loss

train_data = pd.read_csv('processed/train.csv')
val_tr = pd.read_csv('processed/val_tr.csv')
val_te = pd.read_csv('processed/val_te.csv')
test_tr = pd.read_csv('processed/test_tr.csv')
test_te = pd.read_csv('processed/test_te.csv')

# print(val_te)
# evaluate()

# exit()

num_users = train_data['user'].nunique()
num_items = train_data['movie'].nunique()

os.makedirs("checkpoints_vae", exist_ok=True)



train_dataset=VAE_Dataset(num_users,num_items,train_data)

train_dataloader=DataLoader(train_dataset,512,shuffle=True)



num_users_val=val_tr["user"].nunique()

val_dataset = Val_Dataset( num_items, val_tr,val_te)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,collate_fn=val_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"the device is {device}")
model=MultiVAE(num_items,hidden_dim=600,latent_dim=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


num_epochs=200
total_anneal_steps = 200000
anneal_cap=0.2
update_count=0
total_losses=[]
total_recons=[]
total_kls=[]

for epoch in range(num_epochs):
    model.train()

    total_loss=0
    total_recon = 0
    total_kl = 0

    for idx,batch in enumerate(train_dataloader):
        
        batch=batch.to(device)
        optimizer.zero_grad()
        output,mu,logvar=model(batch)


        if total_anneal_steps > 0:
            anneal = min(anneal_cap, update_count / total_anneal_steps)
        else:
            anneal = anneal_cap

        full_loss,recon_loss,kl_loss=loss_function(batch,output,mu=mu,logvar=logvar,beta=anneal)
        full_loss.backward()
        optimizer.step()
        update_count+=1
        total_loss+=full_loss.item()
        total_kl+=kl_loss.item()
        total_recon+=recon_loss.item()
    num_batches = len(train_dataloader)
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches

    total_losses.append(avg_loss)
    total_kls.append(avg_kl)
    total_recons.append(avg_recon)
    if (epoch + 1) % 2 == 0:
        evaluate(model,val_loader,device)
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(total_losses)+1))

        checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'update_count': update_count
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')

        


        plt.plot(epochs, total_losses, label='Total Loss')
        plt.plot(epochs, total_recons, label='Reconstruction Loss')
        plt.plot(epochs, total_kls, label='KL Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_plot_epoch_{epoch+1}.png')
        plt.close()
    
    print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} Avg Recon = {avg_recon:.4f} Avg KL = {avg_kl:.4f}")

