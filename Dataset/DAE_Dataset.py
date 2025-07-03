import torch
import torch.nn as nn

class DAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, num_users, num_items, df, subtract=0, noise_rate=0.0):
        self.num_users = num_users
        self.num_items = num_items
        self.subtract = subtract
        self.noise_rate = noise_rate
        self.user_data = [[] for _ in range(num_users + 1)]
        print("number of users", self.num_users)
        for row in df.itertuples():
            self.user_data[row.user - subtract].append(row.movie)

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        x = torch.zeros(self.num_items)
        movies = self.user_data[idx - self.subtract]
        x[movies] = 1

        # Denoising autoencoder: randomly zero-out inputs
        if self.noise_rate > 0:
            mask = torch.rand(self.num_items) > self.noise_rate
            x = x * mask

        return x
