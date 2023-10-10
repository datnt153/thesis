import re
import glob
import os 
import pandas as pd
# Retrieve the filename from command-line arguments

# Initialize an empty DataFrame
data = pd.DataFrame()
def extract_infor(file_path):

    # Read the contents of the file
    with open(file_path, "r") as file:
        text = file.read()
    
    file_name = file_path.split("/")[-1].replace("txt", "csv")

    # Pattern to extract epoch values
    pattern_epoch = r"-------------------------epoch (\d+) -------------------"

    # Extract epoch values using regular expression
    epoch_matches = re.findall(pattern_epoch, text)

    # Pattern to extract accuracy value
    matches = re.findall(r'(?<=\n)accuracy: (\d+\.\d+)', text)
    acc_matches=[]
    if matches:
        acc_matches  = [float(match) for match in matches]
        # print(acc_matches)

    # print(epoch_matches)
    # print(acc_matches)
    pure_filename = os.path.splitext(os.path.basename(file_path))[0]    # Create a DataFrame from the extracted values
        # Add the data for this file to the DataFrame
    data[pure_filename] = pd.Series(acc_matches)
    # data = pd.DataFrame({"epoch": epoch_matches, "acc": acc_matches})

    # # Save the DataFrame to a CSV file
    # data.to_csv(file_name, index=False)


# Path to the directory containing the text files
directory_path = './'

# List to store the data for all files
all_data = []

# Process each text file in the directory
for filename in glob.glob(os.path.join(directory_path, '**', '**', '*.txt'), recursive=True):
    print(filename)
    extract_infor(filename)

data.to_csv("epoch_accuracy.csv", index=False)
