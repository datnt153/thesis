import pandas as pd
import cv2
import os

# Define paths and other constants
data_path = "/home/datnt114/thesis/aicity2023/code/tmp"
folder_name = "val"
output_path = f"video/{folder_name}"
output_txt = f"video_{folder_name}.txt"

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load the CSV file
csv_file = f"folds/fold_0/{folder_name}_0.csv"  # Replace with your actual CSV file name
df = pd.read_csv(csv_file)


# Function to append video paths and class names to a text file
def append_to_txt(file_path, video_path, class_name):
    with open(file_path, 'a') as f:
        f.write(f"{video_path}, {class_name}\n")


# Function to create video from frames
def create_video(row, data_path, output_path, video_index):
    class_name = row['class_name']
    folder_name = row['folder_name']
    frame_index = row['frame_index']
    view = row['view']

    # Create output directory for the view if it doesn't exist
    view_output_path = os.path.join(output_path, view)
    if not os.path.exists(view_output_path):
        os.makedirs(view_output_path)

    # Define video output file path
    video_file = f"{view_output_path}/{folder_name}_{video_index}.mp4"

    # Define video parameters
    frame_width = 512  # Change as per your image resolution
    frame_height = 512  # Change as per your image resolution
    fps = 30  # Frames per second

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

    # Add frames to the video
    for i in range(frame_index, frame_index + 61):
        img_file = f"{data_path}/{class_name}/{folder_name}/img_{i:06d}.jpg"
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            video_writer.write(img)
        else:
            print(f"Warning: {img_file} does not exist.")
            break

    video_writer.release()
    print(f"Video saved to {video_file}")

    # Append the video path and class name to the text file
    append_to_txt(output_txt, video_file, class_name)


# Clear the output text file before starting
open(output_txt, 'w').close()

# Iterate through the DataFrame and create videos
for index, row in df.iterrows():
    create_video(row, data_path, output_path, index)
