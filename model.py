import torch


class Model:

    def __init__(self, model_type, model_path):
        self.attention_masks_test = None
        self.text_test = None
        self.tokenized_texts = None
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        self.model_type = model_type
        self.initialize_model()

    def initialize_model(self):
        if self.model_type == 'robert':
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
            self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
            # self.load()

    def tokenize_texts(self, texts):
        return self.tokenizer.batch_encode_plus(texts, padding='longest', truncation=True, max_length=128,
                                                return_tensors="pt")

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

    def clean_input(self, texts):
        return texts

    def fit(self, test_texts):
        test_texts = self.clean_input(test_texts)
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

            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                outputs = self.model(text_test_batch, attention_mask=attention_masks_test_batch)

            # Get model predictions
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy().tolist()
            print(probabilities)
            predictions.extend(probabilities)

        return predictions

    def get(self):
        return 'It worked'


