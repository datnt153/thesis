# I. Install mmaction2

### 1. Clone and Install mmaction2
```bash
# Clone mmaction2 repository
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2

# Install mmaction2 and its dependencies
pip install -r requirements/build.txt
pip install -v -e .

# Install other necessary dependencies
mim install "mmaction2>=0.24.0"
```

---

# II. Data Preparation

- Merge all frames into 2-second video clips. The resulting videos and their labels should be placed in the `data/video` folder.

---

# III. Training

1. After installing **mmaction2**, copy all the required configuration files to the **mmaction2** folder.
   
2. Run the following commands to start the training process:
   
   ```bash
   conda activate mmaction2
   bash run_thesis
   ```

---

# IV. Logs and Results

Due to the large size of the model, only logs are submitted. For more detailed information, refer to the logs provided in this link: [mmaction2 Logs](https://drive.google.com/file/d/1qltb7iWwjwrTkwuXJ_9BESFSIoK4Y6ni/view?usp=sharing).