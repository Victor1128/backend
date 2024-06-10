import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from data_loader import DataLoader


class Model:

    def __init__(self, model_path):
        self.attention_masks_test = None
        self.text_test = None
        self.tokenized_texts = None
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        self.initialize_model()

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2, resume_download=None)

    def tokenize_texts(self, texts):
        return self.tokenizer.batch_encode_plus(texts, padding='longest', truncation=True, max_length=128,
                                                return_tensors="pt")

    def clean_input(self, data):
        data_loader = DataLoader()
        data_loader.load_custom_data(data)
        if 'content' in data.columns:
            data_loader.add_content_to_title(1000)
        data_loader.remove_cedilla()
        data_loader.remove_html_tags()
        data_loader.remove_new_lines()
        data_loader.remove_multiple_spaces()
        return data_loader.data['text'].tolist()

    def load_data(self, df):
        test_texts = self.clean_input(df)
        self.tokenized_texts = self.tokenize_texts(test_texts)
        self.text_test = self.tokenized_texts['input_ids']
        self.attention_masks_test = self.tokenized_texts['attention_mask']

    def predict(self):
        predict_batch_size = 32
        predictions = []
        for i in range(0, len(self.text_test), predict_batch_size):
            print(i)
            text_test_batch = self.text_test[i:i + predict_batch_size]
            attention_masks_test_batch = self.attention_masks_test[i:i + predict_batch_size]

            # text_test_batch = text_test_batch.to(device)
            # attention_masks_test_batch = attention_masks_test_batch.to(device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(text_test_batch, attention_mask=attention_masks_test_batch)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy().tolist()
            predictions.extend(probabilities)

        return predictions
