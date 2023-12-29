import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# List of your CSV file names
file_names = ['predicts/image-tf_efficientnetv2_l_in21k_view_Dashboard.csv',
              'predicts/image-tf_efficientnetv2_l_in21k_view_Rear_view.csv',
              'predicts/image-tf_efficientnetv2_l_in21k_view_Right_side_window.csv',
              'predicts/pose-tf_efficientnetv2_l_in21k_view_Dashboard.csv',
              'predicts/pose-tf_efficientnetv2_l_in21k_view_Rear_view.csv',
              'predicts/pose-tf_efficientnetv2_l_in21k_view_Right_side_window.csv']

# Initialize an empty list to store the accuracies for each file
all_accuracies = []

# Process each file
for file in file_names:
    # Read the file
    df = pd.read_csv(file)

    # Extract the 'predict' and 'targets' columns
    y_pred = df['predict']
    y_true = df['targets']
    print( file, " acc: ", round(accuracy_score(y_true, y_pred) * 100, 2))

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(16))  # Assuming class labels are from 0 to 15

    # Handle cases where some classes might not be present in the predictions
    class_accuracies = cm.diagonal() / cm.sum(axis=1, where=cm.sum(axis=1) != 0)

    # Fill missing accuracies with NaN or a specific value
    full_accuracies = [round(class_accuracies[i]*100, 2) if i < len(class_accuracies) else float('nan') for i in range(16)]

    # Append the accuracies for this file to the list
    all_accuracies.append(full_accuracies)

# Convert the list of lists to a DataFrame
result_df = pd.DataFrame(all_accuracies, index=file_names, columns=[f'Class_{i}' for i in range(16)])

# Save the DataFrame to a new CSV file
# result_df.to_csv('accuracy_results.csv', index_label='File')

# Print the DataFrame for verification
print(result_df)
