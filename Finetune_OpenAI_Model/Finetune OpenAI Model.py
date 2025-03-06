# coding: utf-8
"""
Paper Review Aggregation Script (Modified with Detailed Comments).

This script demonstrates how to collect metadata about academic papers from a specified
website or repository (e.g., OpenReview). It uses Selenium for web automation and 
PyMuPDF (fitz) for extracting textual data from downloaded PDF files, including email 
addresses (with support for bracketed formats like {name1, name2}@domain.com).

Prerequisites:
    - Python 3.9+ (recommended)
    - Selenium (pip install selenium)
    - Requests (pip install requests)
    - PyMuPDF (pip install pymupdf)
    - A working ChromeDriver that matches your local Chrome browser version

Usage:
    1. Update the variables: `CONFERENCE_URL`, `OUTPUT_FILE`, and `ERROR_FILE` 
       to reflect your target conference or data repository and output preferences.
    2. Install the required Python libraries as listed in the prerequisites.
    3. Ensure `chromedriver` is accessible via your system PATH, or specify its 
       location when initializing `webdriver.Chrome`.
    4. Execute the script:
         python paper_review_aggregator.py

What It Does:
    - Navigates the given conference or repository web page and scrolls down 
      to load additional content dynamically.
    - Collects links to individual paper forums or pages.
    - Skips already-processed items if partial results exist in `OUTPUT_FILE`.
    - For each paper, gathers essential info: title, abstract, authors, decisions,
      reviewer comments, PDF link, etc.
    - Downloads each PDF (if present) and extracts any email addresses it can find, 
      including bracketed forms.
    - Logs results incrementally to `OUTPUT_FILE` and any errors to `ERROR_FILE`.

After successful completion, you should see two JSON files:
    - OUTPUT_FILE: Contains metadata for successfully processed papers.
    - ERROR_FILE: Contains records of any papers that could not be fully processed 
      due to scraping or download issues.
"""

import json
import os
import time
import openai
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Retrieve and print your OpenAI API key from environment variables (recommended approach).
# If you prefer, you can set openai.api_key directly instead of using environment variables.
# openai.api_key = "some OpenAI API key"
openai.api_key = os.environ['OPENAI_API_KEY']
print(openai.api_key)

# Check if the API key was successfully loaded.
if not openai.api_key:
    raise ValueError("No OPENAI_API_KEY found in environment variables.")

# Load the JSON file containing paper data. In this example, the file is named "iclr2024.json".
# Adjust the filename if your data file is named differently or located elsewhere.
with open('paper_data_file', 'r') as f: # replace with your own data file
    papers = json.load(f)

def preprocess_data(papers):
    """
    Preprocesses each paper by extracting relevant fields: title, abstract, authors,
    reviewer scores, reviewer comments, and the final decision (label).

    Args:
        papers (list): A list of dictionaries, where each dictionary represents a paper
                       and contains fields like 'title', 'abstract', 'authors', 'reviews',
                       and 'decision'.

    Returns:
        list: A list of dictionaries, each containing:
            - title (str)
            - abstract (str)
            - authors (str, comma-separated if multiple authors)
            - scores (str, comma-separated reviewer scores)
            - comments (str, concatenated reviewer comments)
            - label (str, either "Accept" or "Reject")
    """
    dataset = []
    for paper in papers:
        # Extract paper title, abstract, authors
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        authors = ', '.join(paper.get('authors', []))

        # Normalize the acceptance decision to "Accept" or "Reject" 
        # by checking if it starts with "accept" (case-insensitive).
        decision = paper.get('decision', '').strip()
        label = "Accept" if decision.lower().startswith('accept') else "Reject"

        # Combine reviewer scores and comments.
        reviews = paper.get('reviews', [])
        # If there are reviews, join scores with commas; if not, use "N/A".
        scores = ", ".join(str(r.get('score', 'N/A')) for r in reviews) if reviews else "N/A"
        # Similarly, join all reviewer comments into one string; otherwise "N/A".
        comments = " ".join(r.get('comment', '').strip() for r in reviews) if reviews else "N/A"

        dataset.append({
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "scores": scores,
            "comments": comments,
            "label": label
        })
    return dataset

# Call the preprocess function on the loaded papers.
dataset = preprocess_data(papers)
print("Number of papers:", len(dataset))

# Split the dataset into training and test sets, typically 80% train / 20% test. 
# The random_state ensures reproducibility of the split.
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)
print("Number of training samples:", len(train_set))
print("Number of test samples:", len(test_set))

# The best feature set (based on experiments) is "with_comments",
# which includes metadata + reviewer scores + reviewer comments in the prompt.
best_feature_set = "with_comments"

def build_user_content(entry, feature_type):
    """
    Builds the content for the user's message depending on the chosen feature type.

    Args:
        entry (dict): A paper's information containing 'title', 'abstract', 'authors',
                      'scores', 'comments', and 'label'.
        feature_type (str): Determines which fields to include. Possible values:
            - "metadata_only": Only title, abstract, authors.
            - "with_scores": Metadata + scores.
            - "with_comments": Metadata + scores + comments.

    Returns:
        str: A formatted string to be used as the content of a user message in a ChatCompletion.
    """
    # Base text always includes title, abstract, and authors.
    base_text = f"Title: {entry['title']}\nAbstract: {entry['abstract']}\nAuthors: {entry['authors']}"

    if feature_type == "metadata_only":
        # Return only metadata plus a final "Decision:" line.
        user_content = base_text + "\nDecision:"
    elif feature_type == "with_scores":
        # Return metadata plus reviewer scores.
        user_content = base_text + f"\nReviewer Scores: {entry['scores']}\nDecision:"
    else:
        # "with_comments" => metadata, scores, and comments.
        user_content = (
            base_text +
            f"\nReviewer Scores: {entry['scores']}" +
            f"\nReviewer Comments: {entry['comments']}\nDecision:"
        )
    
    return user_content

def fine_tune_model(train_data, feature_type):
    """
    Prepares a JSONL file in the Chat Format required by OpenAI, then creates a fine-tuning job.

    Args:
        train_data (list): The training dataset (list of dicts) containing paper metadata
                           and the final 'label' ("Accept"/"Reject").
        feature_type (str): Which features to include in the user prompt 
                            ("metadata_only", "with_scores", "with_comments").

    Returns:
        str: The ID of the newly created fine-tuning job.
    """
    data_for_finetuning = []
    for entry in train_data:
        # Build the content for the user message.
        user_message_content = build_user_content(entry, feature_type)
        
        # The Chat Format requires an array of messages. The final message (role=assistant)
        # contains the ground-truth label for supervised fine-tuning.
        chat_item = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI that predicts whether a paper should be Accept or Reject."
                },
                {
                    "role": "user",
                    "content": user_message_content
                },
                {
                    "role": "assistant",
                    "content": entry["label"]  # "Accept" or "Reject"
                }
            ]
        }
        data_for_finetuning.append(chat_item)
    
    # Save the training set to a local JSONL file named "chat_fine_tune_data.jsonl".
    jsonl_path = "chat_fine_tune_data.jsonl"
    with open(jsonl_path, "w") as f:
        for record in data_for_finetuning:
            json.dump(record, f)
            f.write("\n")
    
    # Upload the JSONL file to OpenAI for fine-tuning.
    upload_response = openai.File.create(
        file=open(jsonl_path, "rb"),
        purpose="fine-tune"
    )
    file_id = upload_response["id"]
    print(f"Uploaded file ID: {file_id}")

    # Create the fine-tuning job with a specified base model and a custom suffix.
    fine_tune_response = openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-4o-2024-08-06",  # change to the model you want to fine-tune
        suffix="PaperEval",
    )
    
    return fine_tune_response["id"]

# Initiate the fine-tuning process on the training data using the chosen feature set.
fine_tune_job_id = fine_tune_model(train_set, best_feature_set)
print(f"Fine-Tuning Job started: {fine_tune_job_id}")

# Poll the fine-tuning job status at regular intervals (in seconds) until it completes.
poll_interval = 60
while True:
    job_status = openai.FineTuningJob.retrieve(fine_tune_job_id)
    status = job_status["status"]
    print(f"Current fine-tuning job status: {status}")
    
    if status == "succeeded":
        # Retrieve the final model ID upon success.
        final_model_id = job_status["fine_tuned_model"]
        print(f"Fine-tuning succeeded! Model ID: {final_model_id}")
        break
    elif status == "failed":
        # If the job fails, raise an exception.
        raise RuntimeError("Fine-tuning job failed.")
    else:
        # If not succeeded or failed, wait and re-check.
        time.sleep(poll_interval)

def evaluate_model(test_data, model_id, feature_type):
    """
    Uses the specified fine-tuned chat model to generate predictions for each paper in the test set,
    then computes and prints Accuracy, Precision, Recall, and F1-score.

    Args:
        test_data (list): The test dataset (list of dicts) where each dict includes 'label'.
        model_id (str): The ID of the fine-tuned model to be used for inference.
        feature_type (str): The type of features to include in the user prompt (e.g., "with_comments").
    """
    # Build the prompts from the test data.
    test_prompts = [build_user_content(entry, feature_type) for entry in test_data]
    # Extract the ground-truth labels.
    true_labels = [entry["label"] for entry in test_data]

    predictions = []
    for prompt in test_prompts:
        # Call the fine-tuned model with a system prompt + user prompt.
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI that predicts whether a paper should be Accept or Reject."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1,
            temperature=0.0
        )
        # Extract the assistant's text response.
        pred = response['choices'][0]['message']['content'].strip()
        # Normalize the output to "Accept" or "Reject".
        pred_label = "Accept" if pred.lower().startswith("accept") else "Reject"
        predictions.append(pred_label)
    
    # Calculate evaluation metrics.
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, pos_label="Accept")
    rec = recall_score(true_labels, predictions, pos_label="Accept")
    f1 = f1_score(true_labels, predictions, pos_label="Accept")

    # Print out the results.
    print("\nEvaluation on Test Set:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

def evaluate_model(test_data, model_id, feature_type):
    """
    Alternative evaluate_model function that handles potential API key permissions differently,
    but fundamentally does the same steps:
      1) Builds user prompts
      2) Calls the fine-tuned model
      3) Extracts predictions ("Accept"/"Reject")
      4) Computes Accuracy, Precision, Recall, and F1-score

    Args:
        test_data (list): List of dictionaries with 'title', 'abstract', 'authors', etc.
        model_id (str): ID of the fine-tuned model.
        feature_type (str): Indicates which features to use in the prompt
                            ("metadata_only", "with_scores", "with_comments").
    """
    test_prompts = [build_user_content(entry, feature_type) for entry in test_data]
    true_labels = [entry["label"] for entry in test_data]

    predictions = []
    for prompt in test_prompts:
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI that predicts whether a paper should be Accept or Reject."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1,
            temperature=0.0
        )
        pred_text = response['choices'][0]['message']['content'].strip()
        pred_label = "Accept" if pred_text.lower().startswith("accept") else "Reject"
        predictions.append(pred_label)
    
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, pos_label="Accept")
    rec = recall_score(true_labels, predictions, pos_label="Accept")
    f1 = f1_score(true_labels, predictions, pos_label="Accept")

    print("\nEvaluation on Test Set:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

# Print a list of all fine-tuning jobs available to this API key
jobs = openai.FineTuningJob.list()
for job in jobs.data:
    print(job)

# Attempt to retrieve the details of a specific fine-tuning job
job_id = "some fine-tuning job ID"
try:
    job_info = openai.FineTuningJob.retrieve(job_id)
    print("Found job:", job_info)
except openai.error.PermissionError:
    print("No permission to access this Fine-Tuning Job.")

# Print the list of models accessible with the current API key
models = openai.Model.list()
model_ids = [m["id"] for m in models["data"]]
print("Models you can access:", model_ids)

# Retrieve details of a specific fine-tuned model, if you know its ID
model_info = openai.Model.retrieve("some fine-tuned model ID")
print(model_info)

# Attempt to retrieve another model by ID, demonstrating error handling for permissions
model_id = "some fine-tuned model ID"
try:
    model_info = openai.Model.retrieve(model_id)
    print("Found model:", model_info)
except openai.error.PermissionError:
    print("No permission to access this model.")

# Finally, specify the ID of the fine-tuned model that succeeded
final_model_id = "some fine-tuned model ID"  # Replace with your actual fine-tuned model ID
print("Final model ID:", final_model_id)

# Evaluate the final model on the test set
evaluate_model(test_set, final_model_id, best_feature_set)