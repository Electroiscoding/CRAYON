import urllib.request
import json
import urllib.error

url = "https://raw.githubusercontent.com/MrLabbrow/All-English-Words/refs/heads/main/Top%2010000%20Words.txt"

try:
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    words = [line.strip() for line in data.split('\n') if line.strip()]
    
    with open('top_10000_words.txt', 'w', encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')
    print(f"Downloaded {len(words)} words.")
except Exception as e:
    print(f"Error: {e}")
