import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class SpatialAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=200):
        super(SpatialAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


def train_spatial_ae(adata, adj, epochs=150, lr=1e-3, lambda_spatial=0.1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = adata.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        X = torch.tensor(X, dtype=torch.float32).to(device)

        # 邻接矩阵
        if not isinstance(adj, np.ndarray):
            adj = adj.toarray()
        adj = torch.tensor(adj, dtype=torch.float32).to(device)

        model = SpatialAE(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    
        for epoch in range(epochs):
            model.train()
            
            z, x_rec = model(X)

            # 重构损失
            loss_rec = F.mse_loss(x_rec, X)

            # 空间平滑损失
            z_smooth = torch.matmul(adj, z)
            loss_spatial = F.mse_loss(z, z_smooth)

            
            loss = (loss_rec + lambda_spatial * loss_spatial)*0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"AE Epoch {epoch}, Loss: {loss.item():.4f}")

        adata.obsm["AE"] = z.detach().cpu().numpy()
        return adata