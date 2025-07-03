import torch
import torch.nn as nn


class VAE_Dataset(torch.utils.data.Dataset):
    def __init__(self,num_users,num_items,df,subtract=0):
        self.num_users=num_users
        self.num_items=num_items
        self.subtract=subtract 
        self.user_data= [[] for _ in range(num_users+1)]
        print("number of users", self.num_users )
        for row in df.itertuples():
            if subtract>0:
                print(f"{row.user} - {subtract} ={row.user-subtract} ")
            self.user_data[row.user-subtract].append(row.movie)    


    def __len__(self):
        return self.num_users


    def __getitem__(self, idx):
        x=torch.zeros(self.num_items)

        x[self.user_data[idx-self.subtract]]=1
        # print(f"x for me is  {x}")
        return x
        
