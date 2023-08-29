#!/bin/bash

output_file="output_test.csv"
echo "class_name,folder_name,frame_index" > "$output_file"

root_folder="./tmp/"

class_folders=("$root_folder"*/)
total_folders=${#class_folders[@]}
processed_folders=0

# Function to print the progress bar
print_progress() {
  local width=50
  local progress=$(( $processed_folders * 100 / $total_folders ))
  local completed=$(( $progress * $width / 100 ))
  local remaining=$(( $width - $completed ))

  printf "\r[%-${completed}s%-${remaining}s] %d%%" \
    "$([[ $completed -gt 0 ]] && printf '=%.0s' $(seq -s ' ' $completed))" \
    "$([[ $remaining -gt 0 ]] && printf ' %.0s' $(seq -s ' ' $remaining))" \
    "$progress"
}

for class_folder in "${class_folders[@]}"; do
  class_name=$(basename "$class_folder")
  class_name=${class_name%/}

  for folder_path in "$class_folder"*/; do
    folder_name=$(basename "$folder_path")

    img_files=("$folder_path"/img_*.jpg)
    num_files=${#img_files[@]}

    frame_index=1

#    while [ $frame_index -le $num_files ]; do
   while ((frame_index + 60 <= num_files)); do
      echo "$class_name,$folder_name,$frame_index" >> "$output_file"
      ((frame_index+=60))
    done
  done

  ((processed_folders++))
  print_progress
done

# Print final progress bar with 100%
echo

