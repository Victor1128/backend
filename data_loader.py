import pandas as pd
from stop_words import get_stop_words
from string import punctuation
import re
import spacy


class DataLoader:
    def __init__(self, data_path=None):
        '''
        :param data_path: path to a folder that has train.csv, test.csv, validation.csv
        '''
        if data_path and data_path[-1] != '/':
            data_path += '/'
        self.tokenized_data = None
        self.data = None
        self.validation_df = None
        self.test_df = None
        self.train_df = None
        self.punctuation = punctuation.replace('$', '').replace('-', '')
        self.stop_words = get_stop_words('ro')
        self.ner_labels = ['DATETIME', 'EVENT', 'GPE', 'LANGUAGE', 'LOC', 'MONEY', 'NAT_REL_POL', 'NUMERIC_VALUE',
                           'ORGANIZATION', 'PERIOD', 'PERSON', 'QUANTITY', 'WORK_OF_ART']
        self.data_path = data_path
        if data_path:
            self.load_data()

    def load_data(self):
        '''
        Load the data from the data_path
        '''
        self.train_df = pd.read_csv(self.data_path + 'train.csv')
        self.test_df = pd.read_csv(self.data_path + 'test.csv')
        self.validation_df = pd.read_csv(self.data_path + 'validation.csv')
        self.merge_data()

    def remove_diacritics(self):
        self.data['text'] = self.data['text'].replace("ă", "a", regex=True).replace("â", "a", regex=True)\
            .replace("î", "i", regex=True).replace("ș", "s", regex=True).replace("ț", "t", regex=True).replace("Ă", "A", regex=True)\
            .replace("Â", "A", regex=True).replace("Î", "I", regex=True).replace("Ș", "S", regex=True).replace("Ț", "T", regex=True)

    def load_custom_data(self, df):
        self.data = df

    def load_texts(self, data):
        '''
        :param data: pandas DataFrame with 'title' and 'content' columns
        '''
        self.data = data

    def preprocess_raw_data(self):
        self.add_first_sentence_to_title()
        self.remove_nan()
        if len(self.data) == 0:
            raise ValueError('The data is empty')
        self.remove_cedilla()
        self.remove_html_tags()
        self.tokenize_data()
        self.add_ner_labels()
        self.remove_stop_words()
        self.remove_punctuation()
        self.remove_multiple_spaces()
        return self.data

    def tokenize_data(self):
        '''
        Tokenize the data
        '''
        nlp = spacy.load('ro_core_news_md')
        self.tokenized_data = []
        for i, text in enumerate(self.data['text']):
            print(f'\rTokenizing {i + 1}/{len(self.data)}', end='')
            self.tokenized_data.append(nlp(text))
        print()

    def add_ner_labels(self):
        final_texts = []
        for tokenized_text in self.tokenized_data:
            last_label = None
            text = []
            for token in tokenized_text:
                if token.ent_type_ in self.ner_labels:
                    if token.ent_type_ != last_label:
                        text.append(f'${token.ent_type_}$')
                # elif not token.is_stop:
                #     text.append(token.text)
                else:
                    text.append(token.text)
                last_label = token.ent_type_
            final_texts.append(' '.join(text))
        self.data['text'] = final_texts
    def merge_data(self):
        '''
        Merge the data from the dataframes
        '''
        self.data = pd.concat([self.train_df, self.test_df, self.validation_df])

    def remove_nan(self):
        '''
        Remove the rows that have NaN values
        '''
        self.data = self.data.dropna()
        try:
            self.data = self.data.drop(self.data[self.data['text'].apply(lambda x: x == '')].index)
        except:
            pass

    def keep_only_title_with_balance(self):
        '''
        Keep only the title column
        '''
        def f(x):
            if pd.isnull(x):
                return ''
            x = x.split('.')
            if len(x[0].split()) < 5:
                return ' '.join(x[:2])
            return x[0]

        self.data['text'] = self.data['title'].apply(lambda x: x if not pd.isnull(x) else '') + '-----' + self.data[
            'content'].apply(f)
        self.data = self.data.drop(columns=['title', 'content'])
        self.remove_nan()
        self.balance_data()
        self.data['text'] = self.data['text'].apply(lambda x: x.split('-----')[0])

    def add_first_sentence_to_title(self):
        '''
        Add the first sentence of the content to the title.
        '''
        def f(x):
            if pd.isnull(x):
                return ''
            x = x.split('.')
            if len(x[0].split()) < 5:
                return ' '.join(x[:2])
            return x[0]

        self.data['text'] = self.data['title'].apply(lambda x: x if not pd.isnull(x) else '') + '. ' + self.data[
            'content'].apply(f)
        self.data = self.data.drop(columns=['title', 'content'])
        self.data['text'] = self.data['text'].apply(lambda x: x if not x.startswith('. ') else x[2:])

    def add_content_to_title(self, chars_num):
        '''
        Add the first chars_num characters of the content to the title.
        '''
        def f(text):
            if pd.isnull(text) or len(text) == 0:
                return ''
            final = ""
            for sent in text.split('.'):
                sent = sent.strip()
                if len(final + sent) < chars_num:
                    final += sent + '. '
            return final
        self.data['text'] = self.data['title'].apply(lambda x: x if not pd.isnull(x) else '') + '. ' + self.data[
            'content'].apply(f)
        self.data = self.data.drop(columns=['title', 'content'])
        for i, x in self.data.iterrows():
            if type(x['text']) == float:
                print(i, x)
        self.data['text'] = self.data['text'].apply(lambda x: x if not x.startswith('. ') else x[2:])

    def balance_data(self):
        satirical_df = self.data[self.data['label'] == 1]
        non_satirical_df = self.data[self.data['label'] == 0]
        min_len = min(len(satirical_df), len(non_satirical_df))
        satirical_df = satirical_df.sample(min_len, random_state=42)
        non_satirical_df = non_satirical_df.sample(min_len, random_state=42)
        self.data = pd.concat([satirical_df, non_satirical_df])

    def remove_multiple_spaces(self):
        self.data['text'] = self.data['text'].replace(r'\s+', ' ', regex=True)

    def remove_html_tags(self):
        self.data['text'] = self.data['text'].replace(r'<[^>]+>', '', regex=True)

    def remove_stop_words(self):
        self.data['text'] = self.data['text'].apply(
            lambda x: ' '.join([token for token in x.split() if token == 'nu' or token not in self.stop_words]))
        # Remove words that start or end with '-', they are left after stopwords removal
        self.data['text'] = self.data['text'].apply(
            lambda x: ' '.join([token for token in x.split() if token[0] != '-' and token[-1] != '-']))

    def remove_punctuation(self):
        pattern = r'[' + re.escape(self.punctuation) + ']'
        self.data['text'] = self.data['text'].replace(pattern, '', regex=True)

    def remove_cedilla(self):
        self.data['text'] = self.data['text'].replace("ţ", "ț", regex=True).replace("ş", "ș", regex=True).replace("Ţ", "Ț", regex=True).replace("Ş", "Ș", regex=True)

    def serialized_tokenized_data(self):
        final_texts = []
        for text in self.tokenized_data:
            final_texts.append(
                ' '.join([token.text if token.ent_type_ not in self.ner_labels else '$NE$' for token in text]))
        self.data['text'] = final_texts
    def get_balanced_title_and_content_dataset(self):
        '''
        :return: formated dataset with columns 'text' and 'label'
        '''
        self.add_first_sentence_to_title()
        self.remove_nan()
        self.balance_data()
        self.remove_cedilla()
        self.remove_multiple_spaces()
        self.remove_html_tags()
        self.remove_stop_words()
        self.remove_punctuation()
        return self.data

    def get_balanced_title_only_dataset_removing_rows_with_empty_title(self):
        '''
        :return: formated dataset with columns 'text' and 'label'
        '''
        self.keep_only_title()
        self.remove_nan()
        self.balance_data()
        self.remove_cedilla()
        self.remove_multiple_spaces()
        self.remove_html_tags()
        self.remove_stop_words()
        self.remove_punctuation()
        return self.data
