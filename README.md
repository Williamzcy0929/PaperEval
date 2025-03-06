# PaperEval (AI Evaluation of OpenReview Papers)

## Overview
This project aims to automate the extraction of papers from OpenReview for recent ML conferences (e.g., NeurIPS, ICLR, ICML). The scraped data will be structured into a dictionary containing detailed metadata for each paper. Then, we will fine-tune an OpenAI model to predict the final decision based on submission details, reviewer comments, and scores.

## Final Goal
- Automate large-scale scraping of ML conference papers.
- Provide insights into how reviewer feedback affects decisions.

## Phase 1: Web Scraping OpenReview
- Scrape OpenReview to extract metadata for papers from recent ML conferences.
- Organize the extracted data in a structured dictionary format.
- Store the scraped data in a JSON file for later fine-tuning.

### Data Fields to Extract
The output should be a dictionary with the following keys:

| Key | Description |
|---------|---------------|
| `title` | Paper title |
| `author1_name`, `author1_email` | First author’s name and email |
| `author2_name`, `author2_email` | Second author’s name and email (if available) |
| `last_author_name`, `last_author_email` | Last author’s name and email |
| `abstract` | Paper abstract |
| `pdf_link` | Downloadable link to the paper PDF |
| `reviewer1_score` | Score given by reviewer 1 |
| `reviewer1_comments` | All comments from reviewer 1 (concatenated as a string) |
| `reviewer2_score` | Score given by reviewer 2 |
| `reviewer2_comments` | All comments from reviewer 2 (concatenated as a string) |
  ...
| `final_decision` | Acceptance/Rejection |

### Used Tools
- Web Scraping: `request`, `Selenium`
- Dealing with PDF File: `PyMuPDF` (`fitz`)
- Data Storage: JSON file

### Deliverable
- A Python script to:
  - Crawl and scrape OpenReview for recent ML conference submissions.
  - Store the extracted metadata in JSON format.
- README file with:
  - Step-by-step setup instructions.
  - Example data output.

---

## Phase 2: Fine-Tuning an OpenAI Model for Decision Prediction
### Expected Solution
- Train an OpenAI model to predict the final decision of a paper.
- The model can use:
  - Only paper metadata (`title`, `abstract`, `authors`).
  - Paper metadata + reviewer scores.
  - Paper metadata + reviewer scores + reviewer comments.
- Evaluate the model’s accuracy in decision prediction.

### Used Tools
- Model Fine-Tuning through OpenAI API
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

### Deliverable
- A fine-tuning script that:
  - Processes OpenReview data into training format.
  - Fine-tunes an OpenAI model to predict acceptance/rejection.
  - Evaluates prediction performance.
- README file with:
  - Model fine-tuning instructions.
  - Training dataset structure.
  - Evaluation results and benchmarks.
