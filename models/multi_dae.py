import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, drop_out_rate=0.5):
        super(MultiDAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.drop_out = nn.Dropout(p=drop_out_rate)

        # Xavier init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.drop_out(x)

        z = self.encoder(x)
        output = self.decoder(z)

        return output 
