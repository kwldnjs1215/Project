pipeline_path = '/content/drive/MyDrive/Project/기만탐지모델/text_classification_pipeline.pkl'

with open(pipeline_path, 'rb') as f:
    pipe = pickle.load(f)

new_data = pd.read_csv('/content/drive/MyDrive/Project/기만탐지모델/text_data.csv')

new_statements = new_data['text'].iloc[0]

import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

print(new_statements)

sentences = sent_tokenize(new_statements)

len(sentences)

preds = pipe(sentences)

preds_df = pd.DataFrame(preds)

preds_df['adjusted_score'] = (preds_df['score'] - 0.5)

preds_df['adjusted_score'] = preds_df.apply(
    lambda row: -row['adjusted_score'] if row['label'] == 'LABEL_0' else row['adjusted_score'],
    axis=1
)

preds_df.insert(0, 'sentences', sentences)

preds_df.to_csv('/content/drive/MyDrive/Project/기만탐지모델/preds_table.csv', index=False)
