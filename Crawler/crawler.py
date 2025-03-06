# coding: utf-8
"""
Crawler Script for Papers on OpenReview (Modified with Detailed Comments).

This script uses Selenium to automatically crawl paper details—such as title, abstract,
authors, decisions, reviews, etc.—from an OpenReview conference URL. It also downloads
the corresponding PDFs to extract email addresses (including bracketed variants like
{name1, name2}@domain.com) using PyMuPDF (fitz).

Prerequisites:
    - Python 3.9+ (recommended)
    - Selenium (pip install selenium)
    - Requests (pip install requests)
    - PyMuPDF (pip install pymupdf)
    - A matching version of ChromeDriver for your local Chrome browser

Usage:
    1. Set CONFERENCE_URL, OUTPUT_FILE, and ERROR_FILE to desired values.
    2. Install necessary Python packages listed above.
    3. Ensure that ChromeDriver is in your PATH, or specify its location when creating
       the webdriver.Chrome object.
    4. Run this script: python crawler.py

The script will:
    - Scroll through the provided conference page, collecting forum links.
    - Skip any forum IDs already processed if partial data is saved in OUTPUT_FILE.
    - For each forum link:
        * Scrape the paper's title, abstract, authors (plus domain), PDF link, reviews, etc.
        * Download the PDF (if available) and parse text for email addresses.
        * Save results in OUTPUT_FILE incrementally.
        * Log errors in ERROR_FILE.

After the run completes, it will produce two JSON files:
    - OUTPUT_FILE: Contains full metadata for successfully crawled papers.
    - ERROR_FILE: Contains partial metadata for any failed attempts (e.g., scraping errors).
"""

import os
import json
import re
import time
import requests      # For downloading PDFs
import fitz          # PyMuPDF for PDF text extraction

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

# -----------------------------------------------------------------------------
# --- Global Configuration ---
# -----------------------------------------------------------------------------

# Change the following three variables to suit your target conference and desired outputs.
CONFERENCE_URL = "https://openreview.net/..."  # change this to the conference URL
OUTPUT_FILE = "output.json"                    # change this to the output file name
ERROR_FILE = "errors.json"                     # change this to the error file name

# -----------------------------------------------------------------------------
# --- Selenium Setup ---
# -----------------------------------------------------------------------------

# Create a ChromeOptions object to run Chrome headless (without UI)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-dev-shm-usage")

# Instantiate the WebDriver (Chrome). If chromedriver is not in PATH, specify its path here.
driver = webdriver.Chrome(options=chrome_options)

# -----------------------------------------------------------------------------
# --- Checkpoints: Load Existing Data If Any ---
# -----------------------------------------------------------------------------
"""
This section attempts to load any previously saved paper data and errors.
- papers_data: a list of dictionaries with each paper's details.
- errors_data: a list of dictionaries for any encountered errors.
- processed_ids: a set of already processed forum IDs (so we don't re-scrape).
"""

papers_data = []
errors_data = []
processed_ids = set()

# Attempt to load existing output data if the file exists
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        if isinstance(existing, list):
            papers_data = existing
            for p in papers_data:
                if "id" in p:
                    processed_ids.add(p["id"])
        print(f"[INFO] Loaded {len(processed_ids)} papers from {OUTPUT_FILE}; will skip these.")
    except (json.JSONDecodeError, OSError):
        print("[WARN] Could not parse existing success file. Starting fresh.")
        papers_data = []
        processed_ids = set()

# Attempt to load existing error data if the file exists
if os.path.exists(ERROR_FILE):
    try:
        with open(ERROR_FILE, 'r', encoding='utf-8') as f:
            existing_errors = json.load(f)
        if isinstance(existing_errors, list):
            errors_data = existing_errors
    except (json.JSONDecodeError, OSError):
        print("[WARN] Could not parse existing error file. Starting fresh for errors.")
        errors_data = []
else:
    errors_data = []

# -----------------------------------------------------------------------------
# --- Gathering Links from the Conference Page ---
# -----------------------------------------------------------------------------
"""
Here, we:
1. Open the main conference page.
2. Wait until the forum links appear.
3. Scroll multiple times to load additional links if the page is lazy-loaded.
4. Collect unique forum URLs into forum_urls.
"""

print(f"\n[INFO] Opening main tab page: {CONFERENCE_URL}")
driver.get(CONFERENCE_URL)

try:
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/forum?id=')]"))
    )
except Exception:
    print("[ERROR] Forum links did not load; quitting.")
    driver.quit()
    raise

link_elems = driver.find_elements(By.XPATH, "//a[contains(@href, '/forum?id=')]")

# Attempt to scroll the page to load more links
scroll_attempts = 0
while True:
    driver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(2)
    new_elems = driver.find_elements(By.XPATH, "//a[contains(@href, '/forum?id=')]")
    if len(new_elems) > len(link_elems):
        link_elems = new_elems
        scroll_attempts = 0
    else:
        scroll_attempts += 1
        # If no new elements appear after 3 consecutive scrolls, assume we've loaded all
        if scroll_attempts > 3:
            break

forum_urls = []
seen_forum_ids = set()
for elem in link_elems:
    href = elem.get_attribute('href')
    if href and "/forum?id=" in href:
        fid = href.split("id=")[-1]
        if fid not in seen_forum_ids:
            seen_forum_ids.add(fid)
            forum_urls.append(href)

print(f"[INFO] Found {len(forum_urls)} unique forum links.")

# -----------------------------------------------------------------------------
# --- PDF Download and Extraction Utilities ---
# -----------------------------------------------------------------------------
"""
These helper functions handle:
1. download_pdf: download a PDF from a given URL to a local path.
2. extract_text_from_pdf: read the downloaded PDF using PyMuPDF and retrieve all page text.
3. extract_emails_from_text: parse text for both standard (user@domain.com) 
   and bracketed variants ({user1, user2}@domain.com).
4. extract_emails_from_pdf: orchestrates the download and extraction steps, returning a list of emails.
"""

def download_pdf(pdf_url, output_path="temp.pdf"):
    """
    Downloads a PDF from the provided pdf_url, saving it locally to output_path.
    Returns the path to the saved file.
    Raises HTTPError if the download fails (non-200 status).
    """
    print(f"[INFO] Downloading PDF from: {pdf_url}")
    response = requests.get(pdf_url)
    response.raise_for_status()  # If status != 200, raise an HTTPError
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"[INFO] Saved PDF to {output_path}")
    return output_path

def extract_text_from_pdf(pdf_path):
    """
    Opens the local PDF file using PyMuPDF (fitz) and extracts text from all pages.
    Returns a single string containing all text joined by newlines.
    """
    print(f"[INFO] Extracting text from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        page_text = page.get_text("text")
        all_text.append(page_text)
    doc.close()
    return "\n".join(all_text)

def extract_emails_from_text(text):
    """
    Handles both standard emails and bracketed variants like {name1, name2}@domain.com.
    Splits local parts by commas, then reconstructs them with the domain.

    Returns a list of all extracted emails (or an empty list).
    """
    # Regex pattern for bracketed local parts, e.g. {user1, user2}@domain.com
    bracketed_pat = r"\{([^}@]+)\}@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
    # Regex pattern for standard addresses, e.g. user@domain.com
    standard_pat  = r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"

    found_emails = []

    # 1) bracketed pattern matches
    bracketed = re.findall(bracketed_pat, text)
    for local_parts, domain in bracketed:
        sub_parts = [x.strip() for x in local_parts.split(',')]
        for sub in sub_parts:
            found_emails.append(f"{sub}@{domain}")

    # 2) standard pattern matches
    standard = re.findall(standard_pat, text)
    found_emails.extend(standard)

    # Deduplicate
    found_emails = list(set(found_emails))
    return found_emails

def extract_emails_from_pdf(pdf_url, output_path="temp.pdf"):
    """
    1. Download the PDF from pdf_url to output_path.
    2. Extract text with PyMuPDF.
    3. Parse all email addresses (bracketed and standard).
    4. Delete the temporary PDF file.
    5. Return a list of emails or an empty list if none are found.
    """
    local_path = download_pdf(pdf_url, output_path)
    try:
        pdf_text = extract_text_from_pdf(local_path)
        email_list = extract_emails_from_text(pdf_text)
        return email_list
    finally:
        # Ensure we remove the temporary PDF even if extraction fails
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"[INFO] Deleted temporary PDF: {local_path}")

# -----------------------------------------------------------------------------
# --- Author Profile Scraping Utility ---
# -----------------------------------------------------------------------------
"""
Function to open each author's profile in a new browser tab,
extracting the author's name and email domain. This may require
switching window handles in Selenium.
"""

def scrape_author_profile(profile_url):
    """
    Given a profile_url, this function:
    1. Opens the author's profile in a new tab.
    2. Extracts the author's name (if available).
    3. Extracts the email domain from page text (e.g., 'example.com').
    4. Closes the tab and returns to the main window.
    Returns a dict with keys: {name, id, email_domain}, plus an 'error' if any exception occurs.
    """
    profile_data = {"name": None, "id": None, "email_domain": None}
    if not profile_url:
        return profile_data

    try:
        author_id = profile_url.split("id=")[-1]
        profile_data["id"] = author_id

        main_window = driver.current_window_handle
        driver.execute_script("window.open(arguments[0], '_blank');", profile_url)
        WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) == 2)
        driver.switch_to.window(driver.window_handles[-1])

        print(f"[INFO] Scraping author profile: {profile_url}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Attempt h1, fallback to h2 for name
        try:
            h1_elem = driver.find_element(By.TAG_NAME, 'h1')
            profile_data["name"] = h1_elem.text.strip()
        except NoSuchElementException:
            try:
                h2_elem = driver.find_element(By.TAG_NAME, 'h2')
                profile_data["name"] = h2_elem.text.strip()
            except NoSuchElementException:
                profile_data["name"] = None

        # Attempt to parse the email domain from the page text
        page_text = driver.find_element(By.TAG_NAME, "body").text
        email_match = re.search(r'@([\w\.-]+\.\w+)\s+\(Confirmed\)', page_text)
        if not email_match:
            email_match = re.search(r'@([\w\.-]+\.\w+)', page_text)
        if email_match:
            profile_data["email_domain"] = email_match.group(1)

        print(f"[INFO] Author name: {profile_data['name']}, domain: {profile_data['email_domain']}")
    except Exception as e:
        profile_data["error"] = f"Author profile error: {str(e)}"
        print(f"[WARN] Could not scrape author profile: {e}")
    finally:
        # Close the new tab and return to main window
        if len(driver.window_handles) == 2:
            driver.close()
        driver.switch_to.window(main_window)

    return profile_data

# -----------------------------------------------------------------------------
# --- Official Reviews Parsing ---
# -----------------------------------------------------------------------------
"""
Functions to parse the textual blocks containing official reviews, 
splitting them into summary, strengths, weaknesses, etc.
"""

def parse_official_review_text(text_block):
    """
    Parses a single review text block into a dict with fields:
        summary, strengths, weaknesses, questions, ethics, rating, confidence, code_of_conduct
    We look for heading lines (e.g. "Summary:", "Strengths:", "Weaknesses:", etc.)
    and capture all subsequent lines until the next recognized heading.
    """
    review_dict = {
        "summary": "",
        "strengths": "",
        "weaknesses": "",
        "questions": "",
        "ethics": "",
        "rating": "",
        "confidence": "",
        "code_of_conduct": ""
    }

    lines = text_block.splitlines()

    # Sometimes the first line might have "Official Review of Submission"
    if lines and "Official Review of Submission" in lines[0]:
        lines = lines[1:]  # skip that line

    current_field = None
    headings_map = {
        "summary": "summary",
        "strengths": "strengths",
        "weaknesses": "weaknesses",
        "questions": "questions",
        "flag for ethics review": "ethics",
        "rating": "rating",
        "confidence": "confidence",
        "code of conduct": "code_of_conduct"
    }

    # Iterate line by line, checking if it starts with a known heading
    for line in lines:
        stripped = line.strip()
        lower_line = stripped.lower()

        matched_heading = None
        for heading_text, key in headings_map.items():
            if lower_line.startswith(heading_text):
                matched_heading = key
                colon_idx = stripped.find(":")
                if colon_idx != -1:
                    remainder = stripped[colon_idx + 1:].strip()
                    review_dict[key] = remainder
                else:
                    review_dict[key] = ""
                break

        if matched_heading:
            current_field = matched_heading
        else:
            # If the line doesn't match a new heading, it's part of the current field
            if current_field:
                review_dict[current_field] += " " + stripped

    return review_dict

# -----------------------------------------------------------------------------
# --- Core Paper Data Extraction ---
# -----------------------------------------------------------------------------
"""
Main function to scrape the details of a single paper:
1. Navigate to forum URL.
2. Extract title, abstract, decision, authors' info, PDF link.
3. Download the PDF (if any) and parse for emails.
4. Attach up to three extracted emails to the first, second, and last authors.
5. Scrape official reviews if present.
"""

def extract_paper_data(forum_url):
    """
    Given a forum_url, returns a dictionary with the paper's details:
        {
            "id": <forum_id>,
            "url": <forum_url>,
            "title": <paper title or None>,
            "abstract": <paper abstract or None>,
            "decision": <paper decision or None>,
            "first_author": {name, id, email_domain, full_email?},
            "second_author": { ... },
            "last_author": { ... },
            "reviews": [ ... list of parsed reviews ... ],
            "pdf_link": <str or None>,
            "pdf_emails": [ ... list of extracted emails from PDF ... ]
        }
    """
    paper_info = {
        "id": forum_url.split("id=")[-1],
        "url": forum_url,
        "title": None,
        "abstract": None,
        "decision": None,
        "first_author": None,
        "second_author": None,
        "last_author": None,
        "reviews": [],
        "pdf_link": None,
        "pdf_emails": []
    }

    print(f"[INFO] Navigating to forum: {forum_url}")
    driver.get(forum_url)

    # Wait for the forum content to load or for the 'Loading' element to vanish
    try:
        loading_elem = driver.find_element(By.XPATH, "//*[text()='Loading']")
        WebDriverWait(driver, 15).until(EC.staleness_of(loading_elem))
    except NoSuchElementException:
        # If 'Loading' element not found, wait for a known text to appear
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Abstract') or contains(text(),'Official Review')]"))
        )
    except Exception as e:
        print(f"[WARN] Timeout waiting for forum content. {e}")

    # ------------------- Title -------------------
    print("[INFO] Scraping title...")
    try:
        title_elem = driver.find_element(By.TAG_NAME, 'h2')
        title_text = title_elem.text.strip()
        # If the text includes "[Download PDF]", split it out
        if "Download PDF" in title_text:
            title_text = title_text.split("[")[0].strip()
        paper_info["title"] = title_text
    except Exception as e:
        paper_info["title"] = None
        raise Exception(f"Failed to extract title: {e}")

    # ------------------- Abstract -------------------
    print("[INFO] Scraping abstract...")
    abstract_text = ""
    try:
        # Attempt to locate an element containing "Abstract" (case-insensitive)
        abstract_label = driver.find_element(
            By.XPATH,
            "//*[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'abstract')]"
        )
        parent_text = abstract_label.find_element(By.XPATH, "..").text
        abstract_text = parent_text.replace("Abstract:", "").replace("ABSTRACT:", "").strip()
    except NoSuchElementException:
        print("[WARN] Could not locate an element containing 'Abstract'. Abstract set to None.")
    paper_info["abstract"] = abstract_text

    # ------------------- Decision -------------------
    print("[INFO] Scraping decision...")
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text
        decision_match = re.search(r'Decision:\s*(.+)', body_text)
        if decision_match:
            decision_str = decision_match.group(1).split('\n')[0].strip()
            paper_info["decision"] = decision_str
        else:
            # Fallback to searching for an Accept(...) or Reject or Withdrawn pattern
            pattern = re.search(r'Accept\s*\([\w\s]+\)|Reject|Withdrawn', body_text)
            if pattern:
                paper_info["decision"] = pattern.group(0)
            else:
                paper_info["decision"] = None
    except Exception as e:
        print(f"[WARN] Could not parse decision: {e}")
        paper_info["decision"] = None

    # ------------------- Authors -------------------
    print("[INFO] Scraping authors...")
    author_elems = driver.find_elements(By.XPATH, "//h3//a")
    author_urls = [elem.get_attribute('href') for elem in author_elems]
    num_authors = len(author_urls)

    if num_authors >= 1:
        paper_info["first_author"] = scrape_author_profile(author_urls[0])
    if num_authors >= 2:
        paper_info["second_author"] = scrape_author_profile(author_urls[1])
    if num_authors == 1:
        paper_info["last_author"] = paper_info["first_author"]
    elif num_authors == 2:
        paper_info["last_author"] = paper_info["second_author"]
    elif num_authors >= 3:
        paper_info["last_author"] = scrape_author_profile(author_urls[-1])

    # ------------------- PDF Link & Email Extraction -------------------
    print("[INFO] Scraping PDF link...")
    try:
        pdf_elem = driver.find_element(By.XPATH, "//a[@class='citation_pdf_url']")
        pdf_link = pdf_elem.get_attribute("href")
        paper_info["pdf_link"] = pdf_link
    except NoSuchElementException:
        paper_info["pdf_link"] = None

    # If PDF link found, download PDF and parse for emails
    if paper_info["pdf_link"]:
        try:
            print("[INFO] Downloading and parsing PDF for emails...")
            local_pdf_path = download_pdf(paper_info["pdf_link"], "temp_openreview.pdf")
            pdf_text = extract_text_from_pdf(local_pdf_path)
            emails_in_pdf = extract_emails_from_text(pdf_text)
            paper_info["pdf_emails"] = emails_in_pdf

            # We can remove the local PDF after extraction
            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)

            print(f"[INFO] Found {len(emails_in_pdf)} email(s) in PDF.")

            # Attach up to three emails to first, second, and last authors
            if emails_in_pdf:
                if paper_info["first_author"]:
                    paper_info["first_author"]["full_email"] = emails_in_pdf[0]
                if len(emails_in_pdf) >= 2 and paper_info["second_author"]:
                    paper_info["second_author"]["full_email"] = emails_in_pdf[1]
                if len(emails_in_pdf) >= 3 and paper_info["last_author"]:
                    paper_info["last_author"]["full_email"] = emails_in_pdf[-1]
        except Exception as e:
            print(f"[WARN] Could not download/parse PDF: {e}")

    # ------------------- Reviews -------------------
    print("[INFO] Scraping reviews...")
    reviews_data = []
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((
                By.XPATH,
                "//h4/span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'official review of submission')]"
            ))
        )
    except Exception:
        print("[WARN] No 'Official Review of Submission' found after 10 seconds.")
        page_html = driver.page_source
        print("[DEBUG] Partial Page Source (first 2000 chars):")
        print(page_html[:2000])
        all_spans = driver.find_elements(By.XPATH, "//h4/span")
        print(f"[DEBUG] Found {len(all_spans)} <h4><span> elements. Listing their text:")
        for i, sp in enumerate(all_spans, start=1):
            print(f"  Span #{i} text = {repr(sp.text)}")
    else:
        # Attempt to find official review blocks
        review_blocks = driver.find_elements(
            By.XPATH,
            "//h4/span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), "
            "'official review of submission')]/../../following-sibling::div[contains(@class,'note-content') "
            "or contains(@class,'note-content-container')]"
        )
        for block in review_blocks:
            review_text_block = block.text.strip()
            parsed_review = parse_official_review_text(review_text_block)
            reviews_data.append(parsed_review)

    paper_info["reviews"] = reviews_data

    print("[INFO] Finished scraping this paper.")
    return paper_info

# -----------------------------------------------------------------------------
# --- Main Loop Over Forum URLs ---
# -----------------------------------------------------------------------------
"""
Finally, we iterate through each forum URL:
1. Skip if already processed (in processed_ids).
2. Extract the paper data.
3. Save success results to OUTPUT_FILE incrementally.
4. On error, log partial data to errors_list.
5. After the loop, merge errors_list into errors_data and write to ERROR_FILE.
"""

errors_list = []
print(f"[INFO] Found {len(forum_urls)} forum links. Beginning scraping...")

for i, link in enumerate(forum_urls, start=1):
    fid = link.split('id=')[-1]
    if fid in processed_ids:
        print(f"[{i}/{len(forum_urls)}] Skipping already processed paper (Forum ID={fid}).")
        continue

    print(f"\n[{i}/{len(forum_urls)}] Processing forum: {fid}")
    partial_data = {"id": fid, "url": link}

    try:
        paper_data = extract_paper_data(link)
        papers_data.append(paper_data)
        processed_ids.add(fid)

        # Write to OUTPUT_FILE incrementally (so we can resume if interrupted)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2)
        print(f"[INFO] Successfully saved data for forum {fid}.")
    except Exception as e:
        partial_data["error_message"] = str(e)
        errors_list.append(partial_data)
        print(f"[ERROR] Encountered an error with forum {fid}: {str(e)}")

    # Sleep a bit between requests to avoid potential rate-limiting
    time.sleep(5)

# Merge new errors into existing errors_data
errors_data.extend(errors_list)

if errors_data:
    with open(ERROR_FILE, 'w', encoding='utf-8') as f:
        json.dump(errors_data, f, indent=2)
    print(f"[INFO] Wrote {len(errors_data)} total errors to {ERROR_FILE}.")

# Close the WebDriver
driver.quit()
print(f"\n[INFO] All done! Processed {len(papers_data)} papers successfully, with {len(errors_data)} errors.")