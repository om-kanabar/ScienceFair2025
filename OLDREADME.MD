# Science Fair 2025
### By Om Kanabar
## LICENSE NOTE
Please review the LICENSE file to understand the terms and conditions for using this project.

## TOOLS NOTE
This project was made using [Visual Studio Code](https://code.visualstudio.com/) on **MacOS** *Tahoe (26.1)* with a **2022** *M2* MacBookAir with 16 GB of RAM, times may vary with different computers.

---
## A bit about the structure of this project
```ScienceFair2025/Learning```, the learning folder contains practice scripts and experiments, showing the process of exploration and how the final project was developed from these experiments.

```ScienceFair2025/Data```, where data (other than EMINST ByClass) is stored.

```ScienceFair2025/Scripts```, where the non-neural-network essential scripts are stored.

```ScienceFair2025/Models```, where the best models are stored. 
---

## Goal of Project
The goal of the project is to make a lightweight convoluted neural network in Python for recognizing handwritten characters (A-Z, 0-9). It will be trained on the EMNIST Byclass dataset. 

---
## Overview
## Experimental Design (draft)
- Control group: 3×3 kernel
- Experimental group: 2×2, 4×4, 5×5 kernels
- Train 3 networks per kernel size
- Run 3 trials per network
- (Results will be logged in CSV)
2. **DV:** Model Performance Score = `accuracy / (accuracy + (0.75 • inference_time))`

---
# Instructions for Rerunning the Project

## ERROR NOTE

If any errors are found please create an issue at: https://github.com/om-kanabar/ScienceFair2025/issues/ if you are not able to do that please contact Om Kanabar at https://omkanabar.com/contact 

## Before Starting 

### Installing Requirements Before Starting Project

Before running the project, make sure you have Python 3 installed.

1. Open the terminal **in the project folder**:
   - In Finder, right‑click the project folder and select **"New Terminal at Folder"**, *or*
   - Open Terminal and type `cd`, then **drag the ScienceFair2025 folder into the Terminal window** and press Enter.

2. Create and activate a virtual environment (optional but recommended):

   ```python -m venv venv```
   ```source venv/bin/activate```

3. Install all required packages:

   ```pip install -r requirements.txt```

### Check for corrupted data

1. Open the terminal **in the project folder**:
   - In Finder, right‑click the project folder and select **"New Terminal at Folder"**, *or*
   - Open Terminal and type `cd` , then **drag the ScienceFair2025 folder into the Terminal window** and press Enter.

2. Run the file 
#### (Images can be flipped and rotated. This is *normal* and not an issue)

```python3 Scripts/datacheck.py```

3. **Wait for the dataset to load**

4. Follow in-terminal instructions

### 


