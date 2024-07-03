import pandas as pd

df  = pd.read_csv("output_test.csv")


def calculate_columns(x):
    if x.split("_")[0] == "Dashboard":
        column2  = "Dashboard"
    elif x.split("_")[0] == "Right":
        column2 = "Right_side_window"
    else:
        column2 = "Rear_view"

    start = x.find("user")
    column3 = x[start:start+13]

    return column2, column3

# Apply the function and assign the results to new columns
df['view'], df['user_id'] = zip(*df['folder_name'].map(calculate_columns))

df.to_csv("all.csv", index=False)
