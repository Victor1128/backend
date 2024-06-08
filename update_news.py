import pandas as pd
import json
from datetime import datetime

from crawler import get_latest_articles
from model import Model


def update_news():
    articles = []
    for site in ['digi24', 'hotnews', 'antena3', 'protv']:
        print(site)
        articles.extend(get_latest_articles(site))

    with open('crawler/links.json', 'r') as f:
        links = json.load(f)
    last_date = datetime.fromisoformat(links['last_date'])

    articles = [x for x in articles if x['date'] > last_date]
    links['last_date'] = str(max([x['date'] for x in articles]))

    with open('crawler/links.json', 'w') as f:
        json.dump(links, f, indent=4)

    df = pd.DataFrame(articles)
    df_title = df.drop(columns=['content'])
    df_title = df_title.rename(columns={'title': 'text'})
    model = Model('models')
    model.load_data(df)
    satire_content = list(map(lambda x: x[1], model.predict()))
    model.load_data(df_title)
    satire_title = list(map(lambda x: x[1], model.predict()))

    satire = list(filter(lambda x: x[1] >= 0.1 or x[2] >= 0.1, zip(range(len(satire_title)), satire_title, satire_content)))

    df_final = df.iloc[list(map(lambda x: x[0], satire))]
    df_final['satire_title'] = list(map(lambda x: x[1], satire))
    df_final['satire_content'] = list(map(lambda x: x[2], satire))
    df_final.drop(columns=['text'], inplace=True)

    df_final.sort_values(by='date', ascending=False, inplace=True)

    try:
        df_satire = pd.read_csv('satire.csv')
        if len(df_satire) + len(df_final) > 100:
            df_satire.drop(df.tail(len(df_satire) + len(df_final) - 100).index, inplace=True)
    except:
        df_satire = pd.DataFrame()

    df_final = pd.concat([df_final, df_satire])
    df_final.to_csv('satire.csv', index=False)


if __name__ == '__main__':
    update_news()
