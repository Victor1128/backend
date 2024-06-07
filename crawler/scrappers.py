from bs4 import BeautifulSoup
from datetime import datetime


days = {
    "Luni": "Monday",
    "Marți": "Tuesday",
    "Miercuri": "Wednesday",
    "Joi": "Thursday",
    "Vineri": "Friday",
    "Sâmbătă": "Saturday",
    "Duminică": "Sunday"
}

months = {
    "Ianuarie": "January",
    "Februarie": "February",
    "Martie": "March",
    "Aprilie": "April",
    "Mai": "May",
    "Iunie": "June",
    "Iulie": "July",
    "August": "August",
    "Septembrie": "September",
    "Octombrie": "October",
    "Noiembrie": "November",
    "Decembrie": "December",
    "Ian": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Iun": "June",
    "Iul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Noi": "November",
    "Dec": "December"
}


def translate_date(date_str):
    # Înlocuirea zilelor
    for ro, en in days.items():
        date_str = date_str.replace(ro, en)
    # Înlocuirea lunilor
    for ro, en in months.items():
        date_str = date_str.replace(ro, en)
    return date_str

class Digi24:

    @staticmethod
    def get_articles(html):
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('article')
        return articles

    @staticmethod
    def get_articles_links(articles):
        links = []
        for article in articles:
            try:
                link = article.find('h2', class_='article-title').a['href']
                links.append(link)
            except:
                pass
        return links

    @staticmethod
    def get_article_title_content_and_date(article):
        soup = BeautifulSoup(article, 'html.parser')
        article = soup.find('article')
        title = article.find('h1').text.strip()
        content = [x.text.strip() for x in
                   article.find('div', class_='data-app-meta-article').find_all('p', attrs={'data-index': True})
                   if x.text != '' and x.text.replace(u'\xa0', u'') != ''
                   ]
        content = ' '.join(content)
        date = datetime.fromisoformat(article.find('time')['datetime'])
        # make date naive
        date = date.replace(tzinfo=None)
        return title, content, date


class HotNews:

    @staticmethod
    def get_articles(html):
        soup = BeautifulSoup(html, 'html.parser')
        articles = filter(lambda x: x['href'].startswith('/stiri'), soup.find('div', class_='stiri').find_all('a', href=True))
        return articles

    @staticmethod
    def get_articles_links(articles):
        links = []
        for a in articles:
            try:
                link = a['href']
                links.append(link)
            except:
                pass
        return links

    @staticmethod
    def get_article_title_content_and_date(article):
        soup = BeautifulSoup(article, 'html.parser')
        article = soup.find('article')
        title = article.find('h1').text.strip()
        content = [x.text.strip() for x in
                   article.find('div', class_='article-body').find_all('p')
                   if x.text != '' and x.text.replace(u'\xa0', u'') != ''
                   ]
        content = ' '.join(content)
        translated_date = translate_date(article.find('span', class_='ora').text)
        format_str = "%A, %d %B %Y, %H:%M"
        date = datetime.strptime(translated_date, format_str)
        return title, content, date


class Antena3:

    @staticmethod
    def get_articles(html):
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('article')
        return articles

    @staticmethod
    def get_articles_links(articles):
        links = []
        for article in articles:
            try:
                link = article.a['href']
                links.append(link)
            except:
                pass
        return links

    @staticmethod
    def get_article_title_content_and_date(article):
        soup = BeautifulSoup(article, 'html.parser')
        article = soup.find('div', class_="articol")
        title = article.find('h1').text.strip()
        content = [x.text.strip() for x in
                   article.find('div', class_='text').find_all('p')
                   if x.text != '' and x.text.replace(u'\xa0', u'') != '' and x.text != ' '
                   ]
        content = ' '.join(content)
        date = ' '.join(filter(lambda x: x[0].isalnum(), article.find('div', class_='autor-ora-comentarii').find('span', class_=False).text.split()))
        translated_date = translate_date(date)
        format_str = "%d %B %Y %H:%M"
        date = datetime.strptime(translated_date, format_str)
        return title, content, date


class ProTV:

    @staticmethod
    def get_articles(html):
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('article')
        return articles

    @staticmethod
    def get_articles_links(articles):
        links = []
        for article in articles:
            try:
                link = article.a['href']
                links.append(link)
            except:
                pass
        return links

    @staticmethod
    def get_article_title_content_and_date(article):
        soup = BeautifulSoup(article, 'html.parser')
        article = soup.find('article')
        title = article.find('h1').text.strip()
        content = [x.text.strip() for x in
                   article.find('div', class_='article--text').find_all('p', recursive=False)
                   if x.text != '' and x.text.replace(u'\xa0', u'') != '' and x.text != ' '
                   ]
        content = ' '.join(content)
        date = ' '.join(map(lambda x: x.strip(), article.find('div', class_='article--published-date').text.split('|')))
        format_str = '%d-%m-%Y %H:%M'
        date = datetime.strptime(date, format_str)
        return title, content, date

