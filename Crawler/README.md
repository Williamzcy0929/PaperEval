# Crawler for Papers on OpenReview

This repository contains a Python-based crawler that uses **Selenium** to scrape papers from [OpenReview](https://openreview.net/) (e.g., [ICLR 2024](https://openreview.net/group?id=ICLR.cc/2024/Conference#tab-accept-oral)), then downloads each paper’s PDF to extract emails (including bracketed variants like `{name1, name2}@domain.com`). It also extracts other metadata such as title, abstract, authors, and official reviews.

## Features

1. **Headless Selenium**: Scrapes the OpenReview submissions page without opening a visible browser.  
2. **Metadata Extraction**: Title, abstract, decision, authors (first, second, last), and official reviews.  
3. **PDF Download & Email Parsing**:
   - Downloads each paper’s PDF.
   - Uses [PyMuPDF (fitz)](https://pypi.org/project/PyMuPDF/) to extract text.
   - Finds standard emails like `user@domain.com` and bracketed ones such as `{alice, bob}@example.org`.
   - Attaches up to three PDF-based email addresses to the first, second, and last authors under a `"full_email"` key.
   - Stores the entire list of PDF-extracted emails in the `"pdf_emails"` field.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Step-by-Step Setup Instructions](#step-by-step-setup-instructions)
- [Run the Crawler](#run-the-crawler)
- [Example JSON Output](#example-json-output)
- [Limitations](#limitations)

## Installation

You need:

- **Python 3.9+** (or a recent version; we use Python 3.11.11)
- **Selenium** (for web automation)
- **Requests** (for PDF download)
- **PyMuPDF** (for PDF text extraction)
- **ChromeDriver** (to drive Chrome in headless mode)

## Usage

1. The code is generally contained in a Python script or Jupyter notebook (e.g., `crawler_pdf.py` or `crawler.ipynb`).  
2. It automatically:
   - Opens the OpenReview tab (for instance, for [ICLR 2024](https://openreview.net/group?id=ICLR.cc/2024/Conference#tab-accept-oral)).
   - Finds all forum links.
   - Scrapes title, abstract, authors, decision, reviews.
   - Downloads and parses the PDF for emails.
   - Saves the data in JSON.

3. Key variables to change (in the script) before running:

   ```python
   CONFERENCE_URL = "https://openreview.net/..."  # change this to the conference URL
   OUTPUT_FILE = "output.json"                    # change this to the output file name
   ERROR_FILE = "errors.json"                     # change this to the error file name
   ```
4. **Checkpoint / Resume**: By default, the crawler reads/writes a JSON file (e.g. `output.json`) and checks which papers have been completed. If it’s stopped midway, it can restart without duplicating entries or re-processing the same papers.

## Step-by-Step Setup Instructions

The following guide can be followed by anyone, including those new to Python and Selenium:

1. **Install Python** if you do not already have it. You may obtain it from [python.org](https://www.python.org/downloads/). Version 3.9 or higher is recommended.

2. **Install Chrome** and **ChromeDriver**:
   - Ensure that you have Google Chrome installed.
   - Check your ChromeDriver version to match or be compatible with your Chrome browser. You can download ChromeDriver from [chromedriver.chromium.org](https://chromedriver.chromium.org/downloads).

3. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   ```
   Then activate it:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install selenium requests pymupdf
   ```
   Make certain that `pip` is installing into your newly created environment (check with `which pip` or `where pip` if needed).

5. **Download or Clone this repository**:
   ```bash
    git clone https://github.com/Williamzcy0929/PaperEval.git
    cd PaperEval/Crawler
   ```
6. Place your crawler code (e.g., `crawler.py` or `crawler.ipynb`) in this folder if it is not already present.

## Run the Crawler

1. Activate your environment if you have not done so:
   ```bash
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

2. Update the code to point to your desired conference, output file, and error file, as described above (under **Usage**).

3. Run the Python script (`crawler.py`) directly:
   ```bash
   python crawler.py
   ```
   Or, if you have a Jupyter notebook (`crawler.ipynb`), open Jupyter and run all cells.

While running:
- Selenium will navigate to the relevant OpenReview tab.
- It collects all forum links.
- For each paper, it scrapes key metadata and, if a PDF is found, downloads and parses it using PyMuPDF.
- Extracted emails go to `paper_info["pdf_emails"]`, along with `"full_email"` for the first, second, and last authors.
- If the script encounters an error or is terminated unexpectedly, re-running it will pick up from the last processed paper (i.e., it supports checkpoint/resume out of the box).

### ICLR 2024 Example

If you want to specifically crawl ICLR 2024 (Oral Accepts tab), you can set:

```python
CONFERENCE_URL = "https://openreview.net/group?id=ICLR.cc/2024/Conference#tab-accept-oral"
OUTPUT_FILE = "iclr_2024.json"
ERROR_FILE = "errors_iclr_2024.json"
```

Then, run:

```bash
python crawler_pdf.py
```

The script will produce two JSON files:  
- **iclr_2024.json**: The metadata and extracted emails for each paper.  
- **errors_iclr_2024.json**: Any partial error information if issues arise.

## Example JSON Output

When the crawler finishes, it creates (by default) an output JSON file (e.g., `output.json` or `iclr_2024.json`) and an error file (e.g., `errors.json` or `errors_iclr_2024.json`). Below is an example snippet from the main JSON file:

```json
[
  {
    "id": "SAMPLE_FORUM_ID",
    "url": "https://openreview.net/forum?id=SAMPLE_FORUM_ID",
    "title": "Example Paper Title",
    "abstract": "This paper explores ...",
    "decision": "Accept (Oral)",
    "first_author": {
      "name": "Alice Wonderland",
      "id": "~Alice_Wonderland1",
      "email_domain": "example.org",
      "full_email": "alice@cs.example.org"
    },
    "second_author": {
      "name": "Bob Marley",
      "id": "~Bob_Marley1",
      "email_domain": "example.org",
      "full_email": "bob@cs.example.org"
    },
    "last_author": {
      "name": "Zed Last",
      "id": "~Zed_Last1",
      "email_domain": "example.org",
      "full_email": "zed@cs.example.org"
    },
    "reviews": [
      {
        "summary": "...",
        "strengths": "...",
        "weaknesses": "...",
        "questions": "...",
        "ethics": "",
        "rating": "8: Accept",
        "confidence": "3: Fairly confident",
        "code_of_conduct": "Yes"
      }
      /* possibly more reviews */
    ],
    "pdf_link": "https://openreview.net/pdf?id=SAMPLE_FORUM_ID",
    "pdf_emails": [
      "alice@cs.example.org",
      "bob@cs.example.org",
      "zed@cs.example.org"
    ]
  }
]
```

Here, `pdf_emails` is a list of all addresses extracted from the PDF, and each author’s dictionary can contain a `full_email` if assigned from that list.

## Limitations

- Different conferences or different years' OpenReview page structures may differ from that of ICLR 2024 and 2025. If you plan to crawl other conferences (e.g., NeurIPS, ICML), you need to adapt the code accordingly (especially regarding the nested structure of reviews).
- Some conferences may not publish reviews (e.g., ICML), or the reviews links may differ from ICLR. In such cases, the crawler can simply return an empty list for `reviews`.
- Because some PDFs may be anonymized during the double-blind review period, they might not contain any author names or emails. In that case, both `pdf_emails` and `full_email` will be empty.
- This crawler has only been tested in specific environments (Chrome/ChromeDriver, Python 3.9+, Selenium, etc.) and might not be fully compatible with other setups.
- For concurrency or larger-scale crawling, additional mechanisms (e.g., job queues, load balancing, proxies) may be required. The current version does not provide built-in support for these scenarios.

If you have questions about the code or wish to contribute enhancements, please open an issue or pull request on this repository, or contact Changyue (William) Zhao via email ([zhao1944 (at) umn (dot) edu](mailto:zhao1944@umn.edu), [williamzcy929 (at) icloud (dot) com](mailto:williamzcy929@icloud.com)).
