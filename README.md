# VAE and DAE for Collaborative Filtering

A PyTorch reimplementation of **Variational Autoencoders for Collaborative Filtering** comparing Variational Autoencoders (VAE) and Denoising Autoencoders (DAE) for recommendation systems.

## 📄 Paper Reference

This project reimplements the methods described in:
> **Variational Autoencoders for Collaborative Filtering**  
> Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, Tony Jebara  
> [arXiv:1802.05814](https://arxiv.org/pdf/1802.05814)

## 🎯 Overview

This repository contains implementations of two autoencoder architectures for collaborative filtering:

- **Variational Autoencoder (VAE)**: Uses a probabilistic approach with latent variable modeling and β-annealing technique
- **Denoising Autoencoder (DAE)**: Uses deterministic reconstruction with input corruption

Both models learn user preference embeddings and generate personalized movie recommendations through multinomial likelihood optimization.

## 📊 Dataset

**MovieLens-20M (ML-20M)**
- 20 million ratings from 138,000 users on 27,000 movies
- Ratings range from 1 to 5 stars
- Used for implicit feedback by binarizing ratings (≥4 as positive feedback)

## 🏗️ Architecture Details

### Training Configuration
- **VAE**: 150 epochs with β-annealing from 0 to 0.2
- **DAE**: 80 epochs with input dropout regularization
- **Latent dimension**: 200
- **Hidden layers**: [600]
- **Activation**: tanh

### Multinomial Loss Function
The multinomial likelihood treats user behavior as draws from a multinomial distribution:

```
L = -∑(u,i) x_ui * log(softmax(f_θ(z_u))_i)
```

Where:
- `x_ui` = 1 if user u interacted with item i, 0 otherwise
- `f_θ(z_u)` = decoder output for user u's latent representation
- The softmax ensures the output represents a valid probability distribution

## 📈 Results

### Training Progress

#### Variational Autoencoder (VAE)
<div align="center">
  <img src="https://github.com/abdelrahmanHamdyG/MultiVAE-MultiDAE-Recommender/blob/main/github_assets/vae_recon_kl_loss.png" alt="VAE Loss Evolution" width="600"/>
  <p><em>VAE Loss Evolution: Reconstruction Loss and KL Divergence over 150 epochs</em></p>
</div>

<div align="center">
  <img src="https://github.com/abdelrahmanHamdyG/MultiVAE-MultiDAE-Recommender/blob/main/github_assets/vae_metrics.png" alt="VAE Performance Metrics" width="600"/>
  <p><em>VAE Performance Metrics: Recall@50 and NDCG@100 progression</em></p>
</div>

#### Denoising Autoencoder (DAE)
<div align="center">
  <img src="https://github.com/abdelrahmanHamdyG/MultiVAE-MultiDAE-Recommender/blob/main/github_assets/Figure_1_DAE_loss.png" alt="DAE Loss Evolution" width="600"/>
  <p><em>DAE Loss Evolution: Reconstruction Loss over 80 epochs</em></p>
</div>

<div align="center">
  <img src="https://github.com/abdelrahmanHamdyG/MultiVAE-MultiDAE-Recommender/blob/main/github_assets/Figure_1_DAE.png" alt="DAE Performance Metrics" width="600"/>
  <p><em>DAE Performance Metrics: Recall@50 and NDCG@100 progression</em></p>
</div>

### Performance Comparison

| Model | Recall@50 | NDCG@100 | Training Epochs |
|-------|-----------|----------|-----------------|
| **VAE** | **0.5315** | **0.430** | 150 |
| **DAE** | 0.4957 | 0.400 | 80 |

## 🔄 Recommendation Generation

After training, the models generate recommendations by:

1. **Encoding** user interaction vectors into latent representations
2. **Decoding** latent vectors to item probability distributions using multinomial likelihood
3. **Ranking** items by probability scores, excluding previously interacted items
4. **Selecting** top-K items as recommendations

The multinomial approach ensures the model learns to predict the relative likelihood of user interactions across all items simultaneously.

## 🛠️ Implementation Details

- **Framework**: PyTorch
- **VAE Training**: Uses β-annealing technique, gradually increasing β from 0 to 0.2 for optimal performance
- **DAE Training**: Input dropout regularization to prevent overfitting
- **Evaluation**: Top-K recommendation metrics (Recall@50, NDCG@100)
- **Optimizer**: Adam with learning rate 1e-3
