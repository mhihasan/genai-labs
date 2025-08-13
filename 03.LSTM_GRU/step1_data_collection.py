import os
import requests
from typing import List

class DataCollector:
    """
    Handles collection and combination of text data from various sources.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data collector.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_gutenberg_books(self, book_urls: List[str]) -> str:
        """
        Download books from Project Gutenberg.

        Args:
            book_urls: List of URLs to download

        Returns:
            Combined text content
        """
        combined_text = ""

        for i, url in enumerate(book_urls):
            try:
                print(f"Downloading book {i+1}/{len(book_urls)}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Basic cleaning of Gutenberg headers/footers
                text = response.text
                start_marker = "*** START OF THE PROJECT GUTENBERG"
                end_marker = "*** END OF THE PROJECT GUTENBERG"

                start_idx = text.find(start_marker)
                end_idx = text.find(end_marker)

                if start_idx != -1 and end_idx != -1:
                    text = text[start_idx:end_idx]

                combined_text += text + "\n\n"

            except Exception as e:
                print(f"Error downloading {url}: {e}")
                continue

        return combined_text

    def save_text_data(self, text: str, filename: str) -> None:
        """Save text data to file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved text data to {filepath}")

    def load_text_data(self, filename: str) -> str:
        """Load text data from file."""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"File {filepath} not found.")
            return ""


### Initialize data collector
data_collector = DataCollector()

# Download sample books from Project Gutenberg
gutenberg_urls = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    "https://www.gutenberg.org/files/74/74-0.txt",      # The Adventures of Tom Sawyer
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes
]

# Download and combine text data
print("Collecting training data...")
raw_text = data_collector.download_gutenberg_books(gutenberg_urls)
data_collector.save_text_data(raw_text, "combined_books.txt")

print(f"Total characters collected: {len(raw_text):,}")
print(f"First 500 characters:\n{raw_text[:500]}")

