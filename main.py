import os
import re
import requests
import time
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import openai
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import tiktoken
from datetime import datetime
import threading
import queue

# Load OpenAI API key
with open("apikey.txt", "r") as key_file:
    OPENAI_API_KEY = key_file.read().strip()

# Retry mechanism for requests
session = requests.Session()
retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

pdf_queue = queue.Queue()
response_queue = queue.Queue()

# Global file counter
file_id_counter = 1  # Start from 1
file_id_dict = {}

def get_file_id(title):
    global file_id_counter
    if title not in file_id_dict:
        file_id_dict[title] = file_id_counter
        file_id_counter += 1
    return file_id_dict[title]

def sanitize_title(title):
    return re.sub(r'[^a-zA-Z0-9]', '_', title)[:50]


def extract_text_from_bytes(byte_content):
    with BytesIO(byte_content) as open_pdf_file:
        reader = PyPDF2.PdfReader(open_pdf_file)
        return ''.join(page.extract_text() for page in reader.pages[:2])


def fetch_pdf_and_extract_text(url, title, save_path):
    # Check if PDF already exists
    if os.path.exists(save_path):
        print(f"[{get_file_id(title)}] PDF \"{title}\" already exists. Skipping download.")
        with open(save_path, "rb") as f:
            content = f.read()
    else:
        response = session.get(url)
        response.raise_for_status()
        content = response.content

        # Save PDF
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(content)

        time.sleep(1)  # Respect 1 request per second limit

    return extract_text_from_bytes(content)


def construct_directory_path(entry_date, title):
    sanitized_title = sanitize_title(title)
    parsed_date = datetime.strptime(entry_date, '%Y-%m-%dT%H:%M:%SZ')
    return os.path.join("out", str(parsed_date.year), str(parsed_date.month), sanitized_title)


def chat_with_gpt(file_contents, prompt, title):
    def get_token_count(text):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

    MAX_TOKENS = 4000
    combined_content = prompt + file_contents
    combined_tokens = get_token_count(combined_content)

    print(f"[{get_file_id(title)}] Tokens: {combined_tokens}/{MAX_TOKENS}")

    if combined_tokens > MAX_TOKENS:
        print(f"[{get_file_id(title)}] Truncating.")
        truncate_length = MAX_TOKENS - get_token_count(prompt)
        file_contents = file_contents[:truncate_length]

    openai.api_key = OPENAI_API_KEY
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": file_contents},
    ]

    for attempt in range(5):
        try:
            print(f"[{get_file_id(title)}] Querying OpenAI...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                temperature=0.7,
            )
            print(f"[{get_file_id(title)}] Done!")
            return response.choices[0].message.content.strip() + '\n'
        except (openai.error.OpenAIError, requests.exceptions.RequestException) as e:
            print(f"[{get_file_id(title)}] OpenAI API error: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise Exception(f"[{get_file_id(title)}] Failed to get a response from OpenAI after 5 retries.")


def download_pdfs(entries, prompt):
    for entry in entries:
        title = entry.title.text
        entry_date = entry.published.text

        for link in entry.findAll('link'):
            if link.get('title') == 'pdf':
                pdf_url = link['href'] + ".pdf"
                base_filename = os.path.basename(pdf_url)
                dir_path = construct_directory_path(entry_date, title)

                pdf_save_path = os.path.join(dir_path, base_filename)

                # Fetch and process PDF
                print(f"[{get_file_id(title)}] Fetching PDF \"{title}\": {pdf_url}")
                try:
                    extracted_text = fetch_pdf_and_extract_text(pdf_url, title, pdf_save_path)
                except PyPDF2.errors.PdfReadError:
                    print(f"Error processing \"{title}\". EOF marker not found.")
                    if os.path.exists(pdf_save_path):
                        os.remove(pdf_save_path)
                    continue

                response_save_path = os.path.join(dir_path, base_filename + '_response.txt')

                pdf_queue.put((extracted_text, prompt, response_save_path, title))


def process_pdfs():
    while True:
        extracted_text, prompt, response_save_path, title = pdf_queue.get()

        if extracted_text == "TERMINATE":
            break

        # Check if response already exists
        if os.path.exists(response_save_path):
            print(f"[{get_file_id(title)}] Response for \"{title}\" already exists. Skipping OpenAI query.")
            continue

        # Communicate with OpenAI and save response
        print(f"[{get_file_id(title)}] Querying OpenAI for \"{title}\".")
        response_text = chat_with_gpt(extracted_text, prompt, title)
        with open(response_save_path, 'w') as f:
            f.write(response_text)

        time.sleep(1)  # Respect 1 request per second limit


if __name__ == "__main__":
    base_url = "https://export.arxiv.org/api/query?"
    search_term = "(ti:Radiance AND (ti:Fields OR ti:Neural)) ((ti:Neural OR ti:NeRF) AND (ti:Radiance OR ti:Radial) AND ti:Fields) OR ti:NeRF"
    start_index = 0
    max_results = 50

    with open("./prompt.txt", "r") as file:
        prompt = file.read()

    pdf_processor_thread = threading.Thread(target=process_pdfs)
    pdf_processor_thread.start()

    try:
        while True:
            query = f"search_query={search_term}&start={start_index}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            print(f"Querying URL: {base_url}{query}")
            response = session.get(base_url + query)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'xml')
            entries = soup.findAll('entry')

            if not entries:
                print("No entries found :(")
                break

            print(f"{len(entries)} entries found!")

            download_pdfs(entries, prompt)

            start_index += max_results
    finally:
        pdf_queue.put(("TERMINATE", "", "", ""))
        pdf_processor_thread.join()

