# Fine-Tuning an OpenAI Model for Decision Prediction

This repository provides a script or Jupyter notebook for fine-tuning an OpenAI model to predict paper acceptance or rejection decisions based on metadata and reviewer feedback from OpenReview.

## Features
- **OpenAI Model Fine-Tuning**
  - Predict decisions using:
    - Only paper metadata (`title`, `abstract`, `authors`).
    - Metadata + reviewer scores.
    - Metadata + reviewer scores + comments.
- **Intermediate Results Logging**: Clearly logs intermediate training steps for easy debugging.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Step-by-Step Setup Instructions](#step-by-step-setup-instructions)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)

## Installation

You need:

- **Python 3.9+** (or recent version; we use Python 3.11.11)
- **OpenAI API Key** (set via environment variable or directly in the script)
- **OpenAI** (for using the OpenAI API functions)  
- **NumPy, Pandas** (for data management)  
- **Scikit-Learn** (for model performance evaluation)

## Usage

1. The fine-tuning code is provided in a Python script (`Finetune_OpenAI_Model.py`) or a Jupyter notebook (`Finetune_OpenAI_Model.ipynb`).

2. It automatically:
   - Processes OpenReview metadata and reviewer feedback.
   - Prepares data in a format suitable for OpenAI fine-tuning.
   - Fine-tunes the OpenAI model.
   - Evaluates and outputs accuracy, precision, recall, and F1-score.

3. Key variables to configure before running:

   ```python
   DATASET_PATH = "path/to/your/dataset.json"  # Change this to your dataset file path
   OPENAI_API_KEY = "your-api-key"             # Set via environment variable or directly in script
   ```

4. **Monitoring Fine-tuning Progress:**
   - The script polls the status of the fine-tuning job regularly (default: every 60 seconds).
   - Prints current job status, final model ID upon success, or error message upon failure.

5. **Managing Fine-tuning Jobs and Models:**
   - Lists all fine-tuning jobs associated with your API key.
   - Retrieves details of specific fine-tuning jobs or fine-tuned models (handles permission errors gracefully).
   - Prints accessible models and their IDs.

## Step-by-Step Setup Instructions

The following guide can be followed by anyone, including those new to Python and the OpenAI API:

1. **Install Python** if you do not already have it. You may obtain it from [python.org](https://www.python.org/downloads/). Version 3.9 or higher is recommended.

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   ```
   Activate it:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```cmd
     venv\Scripts\activate
     ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn openai
   ```
   Make sure `pip` is installing into your newly created environment by checking with `which pip` (Linux/Mac) or `where pip` (Windows).

3. **Set OpenAI API Key**:

   - **Option A (Recommended)**: Set as an environment variable

     - Linux/Mac:
       ```bash
       export OPENAI_API_KEY='your-api-key'
       ```

     - Windows:
       ```cmd
       set OPENAI_API_KEY='your-api-key'
       ```

   - **Option B**: Manually in the script (uncomment and replace):

     ```python
     openai.api_key = "your-api-key"
     ```

3. **Download or Clone this repository**:
   ```bash
    git clone https://github.com/Williamzcy0929/PaperEval.git
    cd PaperEval/Finetune OpenAI Model
   ```

## Run the Fine-Tuning Script

1. Activate your environment if you have not done so:
   ```bash
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

2. Update the script to point to your training dataset:
   ```python
   with open('paper_data_file', 'r') as f:
       papers = json.load(f)  # replace with your own data file
   ```

3. Split your data into training and testing sets ($80\%$ for training set, $20\%$ for test set):
   ```python
   train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)
   ```

4. Run the Python script (`Finetune OpenAI Model.py`) directly:
   ```bash
   python Finetune_OpenAI_Model.py
   ```
   Or, if you have a Jupyter notebook (`Finetune OpenAI Model.ipynb`), open Jupyter and run all cells.

### During Execution:
- The script processes and prepares data.
- Intermediate results are printed for tracking and debugging purposes.
- The model is fine-tuned using OpenAI’s API.
- Evaluation metrics (Accuracy, Precision, Recall, F1-score) are displayed upon completion.

### Choosing the Best Model:
The provided fine-tuning script selects the best-performing model variant. The default best model is usually "with_comments" (trained and tested on the `ICLR 2024` dataset), but this might vary depending on your specific dataset. Perform tuning and evaluation to identify the optimal model based on your test results.

## Evaluation Metrics

### Accuracy
- **Definition**: The proportion of correctly predicted cases out of all predictions made.
- **Use Case**: Useful when the dataset is balanced.
- **Formula**:

$$Accuracy = \frac{TP + TN}{TP + FP + TN + FN}$$

### Precision
- Measures how many of the items labeled as positive are correctly labeled.
- **Use Case**: Useful when false positives are costly.
- **Formula**:

$$Precision = \frac{TP}{TP + FP}$$

### Recall
- Measures how many of the true positives the model correctly identified.
- **Use Case**: Important when false negatives carry high costs.
- **Formula**:

$$Recall = \frac{TP}{TP + FN}$$

### F1-Score
- The harmonic mean of precision and recall, balancing both metrics.
- **Use Case**: Suitable when you want a balance between precision and recall.
- **Formula**:

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

- **Note**: TP (True Positives), TN (True Negatives), FP (False Positives), FN (False Negatives).

## Limitations
- Model accuracy depends heavily on data quality and completeness.
- Ensure dataset consistency and adequate size for meaningful results.
- This script is designed specifically for OpenReview data; modifications may be necessary for other data sources or formats.
- The best model (among `metadata`, `metadata + reviewer scores`, `metadata + reviewer scores + comments`) might vary from different datasets; need to select the model with best performance based on test set performance.

If you have questions about the code or wish to contribute enhancements, please open an issue or pull request on this repository, or contact Changyue (William) Zhao via email ([zhao1944 (at) umn (dot) edu](mailto:zhao1944@umn.edu), [williamzcy929 (at) icloud (dot) com](mailto:williamzcy929@icloud.com)).
