import requests
import json
from crawler.scrappers import *


def get_html(url):
    headers = {
        "User-Agent": "curl/7.64.1"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"A aparut o eroare: {e}")
        return None


def get_latest_articles(site):
    if site == 'digi24':
        scrapper = Digi24()
    elif site == 'hotnews':
        scrapper = HotNews()
    elif site == 'antena3':
        scrapper = Antena3()
    elif site == 'protv':
        scrapper = ProTV()
    else:
        raise ValueError('Site-ul nu este recunoscut')
    with open ('crawler/links.json', 'r') as f:
        links = json.load(f)
    mainUrl = links[site]['mainUrl']
    newsUrl = links[site]['newsUrl']
    html = get_html(newsUrl)
    articles = scrapper.get_articles(html)
    links = scrapper.get_articles_links(articles)
    articles = []
    for link in links:
        html = get_html(mainUrl + link)
        title, content, date = scrapper.get_article_title_content_and_date(html)
        articles.append({
            'title': title,
            'content': content,
            'date': date,
            'source': site,
            'link': mainUrl + link
        })
    return articles

