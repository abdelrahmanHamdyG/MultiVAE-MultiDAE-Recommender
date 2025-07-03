import torch
import torch.nn as nn


class Val_Dataset(torch.utils.data.Dataset):
    def __init__(self, num_items, val_tr_df, val_te_df):
        self.num_items = num_items

        # build a user -> items list mapping
        self.user_inputs = {}  # from val_tr
        self.user_targets = {}  # from val_te

        for row in val_tr_df.itertuples():
            self.user_inputs.setdefault(row.user, []).append(row.movie)

        for row in val_te_df.itertuples():
            self.user_targets.setdefault(row.user, set()).add(row.movie)

        # only keep users who have both val_tr and val_te
        self.user_list = list(set(self.user_inputs.keys()) & set(self.user_targets.keys()))

    def __len__(self):
        return len(self.user_list)
    def __getitem__(self, idx):
        user_id = self.user_list[idx]
        x = torch.zeros(self.num_items)
        x[self.user_inputs[user_id]] = 1
        target = list(self.user_targets[user_id])  
        return x, target, user_id
