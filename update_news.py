import sys
import os
import pandas as pd
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from crawler import get_latest_articles
# from model import Model, DataLoader


def update_news():
    articles = []
    for site in ['digi24', 'hotnews', 'antena3', 'protv']:
        print(site)
        articles.extend(get_latest_articles(site))

    with open('crawler/links.json', 'r') as f:
        links = json.load(f)
    last_date = datetime.fromisoformat(links['last_date'])
    print(str(last_date))

    articles = [x for x in articles if x['date'] > last_date]
    links['last_date'] = str(max([x['date'] for x in articles]))

    with open('crawler/links.json', 'w') as f:
        json.dump(links, f, indent=4)

    data = pd.DataFrame(articles)
    print(data.head())


if __name__ == '__main__':
    update_news()
