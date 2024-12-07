import requests
from bs4 import BeautifulSoup
import json

def htm_to_json(url):
   headers = {'User-Agent': 'markchung21@hotmail.com'}
   response = requests.get(url, headers=headers)
   soup = BeautifulSoup(response.content, 'html.parser')
   
   data = {
       'title': soup.title.text if soup.title else '',
       'sections': []
   }
   
   sections = soup.find_all(['h1', 'h2', 'h3', 'p'])
   for section in sections:
       data['sections'].append({
           'type': section.name,
           'content': section.get_text().strip()
       })
   
   with open('edgar_data.json', 'w', encoding='utf-8') as f:
       json.dump(data, f, indent=2, ensure_ascii=False)

# Usage
url = "https://www.sec.gov/ix?doc=/Archives/edgar/data/1280452/000143774924006133/mpwr20231231_10k.htm"
htm_to_json(url)