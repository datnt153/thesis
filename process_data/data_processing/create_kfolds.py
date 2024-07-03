import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path

# Create a list
# df = pd.read_csv('total_data.csv')
list_user = [
    "user_id_13522",
    "user_id_28557",
    "user_id_30932",
    "user_id_31903",
    "user_id_39367",
    "user_id_42711",
    "user_id_46856",
    "user_id_47457",
    "user_id_57207",
    "user_id_59581",
    "user_id_60167",
    "user_id_60768",
    "user_id_61962",
    "user_id_63513",
    "user_id_63764",
    "user_id_71436",
    "user_id_78052",
    "user_id_83323",
    "user_id_83756",
    "user_id_84935",
    "user_id_85870",
    "user_id_86356",
    "user_id_86952",
    "user_id_96269",
    "user_id_99882"
]

# # Set the number of folds
# k = 5

# # Initialize the KFold object
# kf = KFold(n_splits=k, shuffle=True, random_state=42)
df = pd.read_csv("all.csv")

val_index = [0, 5, 10, 14, 20 ]
train_index =  [i  for i in range(25) if i not in val_index]
fold=0
Path(f"folds/fold_{fold}").mkdir(parents=True, exist_ok=True)
train_data = [list_user[i] for i in train_index]
val_data = [list_user[i] for i in val_index]
print(f"val data: {val_data}")

train = df[~df['user_id'].isin(val_data)].sample(frac=1).reset_index(drop=True)
val = df[df['user_id'].isin(val_data)].sample(frac=1).reset_index(drop=True)

train.to_csv(f"folds/fold_{fold}/train_{fold}.csv")
val.to_csv(f"folds/fold_{fold}/val_{fold}.csv")

with open(f"folds/fold_{fold}/val_id.txt", "w") as file:
    for item in val_data:
        file.write(item + "\n")


# # Loop through each fold
# for fold, (train_index, val_index) in enumerate(kf.split(list_user)):
#     # Get the training and validation data for this fold
#     Path(f"folds/fold_{fold}").mkdir(parents=True, exist_ok=True)
#     train_data = [list_user[i] for i in train_index]
#     val_data = [list_user[i] for i in val_index]
#     print(f"val data: {val_data}")

#     train = df[~df['user_id'].isin(val_data)].sample(frac=1).reset_index(drop=True)
#     val = df[df['user_id'].isin(val_data)].sample(frac=1).reset_index(drop=True)

#     train.to_csv(f"folds/fold_{fold}/train_{fold}.csv")
#     val.to_csv(f"folds/fold_{fold}/val_{fold}.csv")

#     with open(f"folds/fold_{fold}/val_id.txt", "w") as file:
#         for item in val_data:
#             file.write(item + "\n")

