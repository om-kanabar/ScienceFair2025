# Science Fair 2025
### By Om Kanabar
This is my project for CPS science fair 2025.
## LICENSE NOTE
Please review the LICENSE file to understand the terms and conditions for using this project.

---
## A bit about the structure of this project
```ScienceFair2025/Learning```, the learning folder contains practice scripts and experiments, showing the process of exploration and how the final project was developed from these experiments.

```ScienceFair2025/Data```, where data (other than EMINST ByClass) is stored.

```ScienceFair2025/Miscellaneous```, where other things (excluding the main project files [the files needed for running the project]) is stored.

---
## Installing Requirements Before Starting Project (For redoing the experiment)

Before running the project, make sure you have Python 3 installed.

1. Open Command Prompt or PowerShell.
2. Navigate to the project folder, e.g.:

   ```cd path\to\ScienceFair2025```

3. Create and activate a virtual environment (optional but recommended):

   ```python -m venv venv```
   ```.\venv\Scripts\activate```

4. Install all required packages:

   ```pip install -r requirements.txt```


## Goal of Project
The goal of the project is to make a lightweight neural network in Python for recognizing handwritten charecters (A-Z, 0-9). It will be trained on the EMNIST Mixed dataset.

---
## Overview
1. **IV:** Number of neurons per hidden layer 
- **Experimental Group**: 256, 512, 2048, 4096; 2<sup>8</sup>, 2<sup>9</sup>, 2<sup>11</sup>, 2<sup>12</sup>
- **Control Group** 1024; 2<sup>10</sup>
2. **DV:** 
```
accuracy / (accuracy + (0.75 * model_time))
```


