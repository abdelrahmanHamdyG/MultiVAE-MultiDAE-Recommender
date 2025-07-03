import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split



np.random.seed(42)


df=pd.read_csv("ml-20m/ratings.csv")


# keep only thoose with rating 4 or above
print(len(df))
df = df[df['rating'] >3.5]






users_rated_count=df["userId"].value_counts()
df=df[df['userId'].isin(users_rated_count[users_rated_count>=5].index)]

# split the data
users=df["userId"].unique()
train_users, heldout_users = train_test_split(users, test_size=20000, random_state=42)
val_users, test_users = train_test_split(heldout_users, test_size=0.5, random_state=42)
train_df=df[df["userId"].isin(train_users)]

val_df=df[df["userId"].isin(val_users)]
test_df=df[df["userId"].isin(test_users)]


# 
movies_id=pd.unique(train_df["movieId"])
movieId_map = {mid: idx for idx, mid in enumerate(movies_id)}



val_df = val_df[val_df['movieId'].isin(movieId_map.keys())]
test_df = test_df[test_df['movieId'].isin(movieId_map.keys())]


userId_all = pd.concat([train_df, val_df, test_df])['userId'].unique()
userId_map={user:idx for idx,user in enumerate(userId_all)}


def numerize(df):
    return pd.DataFrame({
        'user': df['userId'].map(userId_map),
        'movie': df['movieId'].map(movieId_map)
})

train_data = numerize(train_df)
val_data_full = numerize(val_df)
test_data_full = numerize(test_df)

def split_holdout(df):
    grouped=df.groupby("user")
    train_part=[]
    test_part=[]
    for uid,group in grouped:
        if len(group) < 5 :
            train_part.append(group)
            continue
        
        n_test=int(np.ceil(len(group))*0.2)
        indices=np.random.choice(len(group),size=n_test,replace=False)
        mask=np.zeros(len(group),dtype=bool)
        mask[indices]=True
        
        test_part.append(group[mask])
        train_part.append(group[~mask])
    return pd.concat(train_part),pd.concat(test_part)

val_tr, val_te = split_holdout(val_data_full)
test_tr, test_te = split_holdout(test_data_full)

os.makedirs('processed', exist_ok=True)
train_data.to_csv('processed/train.csv', index=False)
val_tr.to_csv('processed/val_tr.csv', index=False)
val_te.to_csv('processed/val_te.csv', index=False)
test_tr.to_csv('processed/test_tr.csv', index=False)
test_te.to_csv('processed/test_te.csv', index=False)
