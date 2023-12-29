import pandas as pd

def check_file(view):
    # Paths to your CSV files
    file_path1 = f'preds/image-tf_efficientnetv2_l_in21k_view_{view}.csv'
    file_path2 = f'preds/pose-tf_efficientnetv2_l_in21k_view_{view}.csv'

    # Read the CSV files
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Drop the 'Unnamed:' column if it exists in the DataFrames
    df1 = df1.drop(columns=[col for col in df1.columns if 'Unnamed:' in col], errors='ignore')
    df2 = df2.drop(columns=[col for col in df2.columns if 'Unnamed:' in col], errors='ignore')


    # Join the DataFrames on the 'file' column
    # You can change the 'how' parameter to 'left', 'right', 'outer', or 'inner' depending on the type of join you need
    df = pd.merge(df1, df2, on='file', how='inner', suffixes=('_image', '_pose'))

    df = df[(df['predict_image'] != df['targets_image']) & (df['predict_pose'] == df['targets_pose'])]

    # Print the grouped DataFrame
    print(df)

    # Optionally, you can save this grouped data to a new CSV
    df.to_csv(f'diff_preds/{view}_results.csv', index=False)


views = ['Dashboard', 'Rear_view', 'Right_side_window']

for view in views:
    check_file(view=view)