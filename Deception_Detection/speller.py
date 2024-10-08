# -*- coding: utf-8 -*-
"""speller.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KHBSmNHL4zih53WJTBBkFGpyLnm9pNO0
"""

from google.colab import drive
drive.mount('/content/drive')

'''
import pandas as pd

whisper_data = pd.read_csv('/content/drive/MyDrive/Project/기만탐지모델/whisper_text.csv')
whisper_text = whisper_data['text'].astype(str).iloc[0]

file_path = '/content/drive/MyDrive/Project/기만탐지모델/original_text.txt'

with open(file_path, 'r') as file:
    content = file.read()

from nltk.tokenize import sent_tokenize

# 문장 단위로 나누기
whisper_sentences = sent_tokenize(whisper_text)
original_sentences = sent_tokenize(content)
'''

import pandas as pd

data = pd.read_csv('/content/drive/MyDrive/Project/기만탐지모델/문장비교.csv')

# 데이터프레임 생성
match_df = pd.DataFrame(data)

match_df = match_df.fillna('')

# 문장 일치 여부를 확인하는 함수
def check_if_sentences_match(row):
    return row['whisper'].strip() == row['original'].strip()

# 'match' 열을 추가하여 문장 일치 여부를 기록
match_df['match'] = match_df.apply(check_if_sentences_match, axis=1)

match_df['match'].value_counts()

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

similarity_df = pd.DataFrame(data)

similarity_df.rename(columns={'original': 'correct sentence'}, inplace=True)

# 'correct sentence' 컬럼을 맨 앞으로 이동
columns = ['correct sentence'] + [col for col in similarity_df.columns if col != 'correct sentence']
similarity_df = similarity_df[columns]

similarity_df = similarity_df.fillna('')

# 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

"""## Whisper"""

# 문장 임베딩을 계산하여 유사도 측정
def calculate_embedding_similarity(row):
    sentences = [row['correct sentence'].strip(), row['whisper'].strip()]
    embeddings = model.encode(sentences)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# 유사도 점수를 데이터프레임에 추가
similarity_df['similarity (whisper)'] = similarity_df.apply(calculate_embedding_similarity, axis=1)

similarity_df

import seaborn as sns
import matplotlib.pyplot as plt

# 유사도 점수의 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
sns.histplot(similarity_df['similarity (whisper)'], bins=10, kde=True, color='blue')
plt.title('Distribution of Similarity (whisper) Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 유사도가 0.9 이상인 항목의 개수와 전체 항목의 개수
count_above_threshold = (similarity_df['similarity (whisper)'] >= 0.9).sum()
total_count = len(similarity_df)

# 비율 계산
whisper_percentage_above_threshold = (count_above_threshold / total_count) * 100

# 결과 출력
print(f"0.9 이상인 유사도의 비율: {whisper_percentage_above_threshold:.2f}%")

sentences = similarity_df['whisper']

"""## BERT"""

import re
import nltk
from transformers import pipeline

# NLTK 패키지 다운로드 (처음 사용하는 경우 필요)
nltk.download('punkt')

# BERT 기반의 교정 모델 로드
corrector = pipeline("text2text-generation", model="facebook/bart-large")

# 원본 텍스트와 교정된 텍스트를 담을 리스트
bert_data = []

def bert_correct_spelling(text):
    max_length = 512  # BART 모델의 max_length는 1024로 설정되지만, 안전하게 512로 설정

    # 텍스트를 더 작은 조각으로 나누는 함수
    def split_text(text, max_length):
        words = text.split()
        for i in range(0, len(words), max_length):
            yield ' '.join(words[i:i + max_length])

    corrected_text = ""
    for chunk in split_text(text, max_length):
        # 교정 결과를 가져옵니다.
        result = corrector(chunk, max_length=max_length, num_beams=5, early_stopping=True)
        corrected_text += result[0]['generated_text'] + " "

    return corrected_text.strip()

for sentence in sentences:
    # 맞춤법 교정
    corrected_sentence = bert_correct_spelling(sentence)

    # data에 추가
    bert_data.append({'BERT': corrected_sentence})

# 새로운 DataFrame 생성
bert_df = pd.DataFrame(bert_data)

similarity_df['BERT'] = bert_df['BERT']

# 문장 임베딩을 계산하여 유사도 측정
def calculate_embedding_similarity(row):
    sentences = [row['correct sentence'].strip(), row['BERT'].strip()]
    embeddings = model.encode(sentences)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# 유사도 점수를 데이터프레임에 추가
similarity_df['similarity (BERT)'] = similarity_df.apply(calculate_embedding_similarity, axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

# 유사도 점수의 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
sns.histplot(similarity_df['similarity (BERT)'], bins=10, kde=True, color='blue')
plt.title('Distribution of Similarity (BERT) Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 유사도가 0.9 이상인 항목의 개수와 전체 항목의 개수
count_above_threshold = (similarity_df['similarity (BERT)'] >= 0.9).sum()
total_count = len(similarity_df)

# 비율 계산
bert_percentage_above_threshold = (count_above_threshold / total_count) * 100

# 결과 출력
print(f"0.9 이상인 유사도의 비율: {bert_percentage_above_threshold:.2f}%")

"""## SymSpell"""

!pip install symspellpy

from symspellpy import SymSpell, Verbosity
import os

# SymSpell 초기화
symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# 사전 파일의 경로를 설정합니다. 아래 링크에서 직접 다운로드한 후, 로컬 경로를 지정합니다.
dictionary_path = '/content/drive/MyDrive/Project/기만탐지모델/en-80k.txt'
symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# 오타 교정 함수 정의
def symspell_correct_spelling(text):
    # 문장을 단어 단위로 분리합니다.
    words = text.split()
    corrected_words = []
    for word in words:
        # 각 단어의 교정 제안 받기
        suggestions = symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_word = suggestions[0].term if suggestions else word
        corrected_words.append(corrected_word)
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# 원본 텍스트와 교정된 텍스트를 담을 리스트
symspell_data = []

# 문장 단위로 나누기
for sentence in sentences:
    # 맞춤법 교정
    corrected_sentence = symspell_correct_spelling(sentence)

    # data에 추가
    symspell_data.append({'SymSpell': corrected_sentence})

# 새로운 DataFrame 생성
symspell_df = pd.DataFrame(symspell_data)

similarity_df['SymSpell'] = symspell_df['SymSpell']

# 문장 임베딩을 계산하여 유사도 측정
def calculate_embedding_similarity(row):
    sentences = [row['correct sentence'].strip(), row['SymSpell'].strip()]
    embeddings = model.encode(sentences)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# 유사도 점수를 데이터프레임에 추가
similarity_df['similarity (SymSpell)'] = similarity_df.apply(calculate_embedding_similarity, axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

# 유사도 점수의 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
sns.histplot(similarity_df['similarity (SymSpell)'], bins=10, kde=True, color='blue')
plt.title('Distribution of Similarity (SymSpell) Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 유사도가 0.9 이상인 항목의 개수와 전체 항목의 개수
count_above_threshold = (similarity_df['similarity (SymSpell)'] >= 0.9).sum()
total_count = len(similarity_df)

# 비율 계산
symspell_percentage_above_threshold = (count_above_threshold / total_count) * 100

# 결과 출력
print(f"0.9 이상인 유사도의 비율: {symspell_percentage_above_threshold:.2f}%")

"""## TextBlob"""

!pip install TextBlob

from textblob import TextBlob

# 원본 텍스트와 교정된 텍스트를 담을 리스트
textblob_data = []

for sentence in sentences:

    # TextBlob 객체 생성
    blob = TextBlob(sentence)

    # 오타 교정
    corrected_sentence = blob.correct()

    # data에 추가
    textblob_data.append({'TextBlob': str(corrected_sentence)})

# 새로운 DataFrame 생성
textblob_data_df = pd.DataFrame(textblob_data)

similarity_df['TextBlob'] = textblob_data_df['TextBlob']

# 문장 임베딩을 계산하여 유사도 측정
def calculate_embedding_similarity(row):
    sentences = [row['correct sentence'].strip(), row['TextBlob'].strip()]
    embeddings = model.encode(sentences)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# 유사도 점수를 데이터프레임에 추가
similarity_df['similarity (TextBlob)'] = similarity_df.apply(calculate_embedding_similarity, axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

# 유사도 점수의 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
sns.histplot(similarity_df['similarity (TextBlob)'], bins=10, kde=True, color='blue')
plt.title('Distribution of Similarity (TextBlob) Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 유사도가 0.9 이상인 항목의 개수와 전체 항목의 개수
count_above_threshold = (similarity_df['similarity (TextBlob)'] >= 0.9).sum()
total_count = len(similarity_df)

# 비율 계산
textblob_percentage_above_threshold = (count_above_threshold / total_count) * 100

# 결과 출력
print(f"0.9 이상인 유사도의 비율: {textblob_percentage_above_threshold:.2f}%")

"""## T5 Seq2Seq"""

!pip install transformers torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

# 모델과 토크나이저 로드
model_name = 't5-small'  # 작은 크기의 T5 모델을 사용
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 원본 텍스트와 교정된 텍스트를 담을 리스트
seq2seq_data = []

for sentence in sentences:
    # 입력 텍스트를 토큰화
    input_text = f"Fix grammatical errors in the following {sentence}"

    inputs = tokenizer(input_text, return_tensors='pt')

    # 모델에 입력 텍스트를 전달하여 출력 생성
    outputs = model.generate(inputs['input_ids'], max_length=40, num_beams=4, early_stopping=True)

    # 출력 텍스트 디코딩
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # data에 추가
    seq2seq_data.append({'Seq2Seq': output_text})

# 새로운 DataFrame 생성
seq2seq_data_df = pd.DataFrame(seq2seq_data)

similarity_df['Seq2Seq'] = seq2seq_data_df['Seq2Seq']

from sentence_transformers import SentenceTransformer, util
import pandas as pd

# SentenceTransformer 모델 로드
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 문장 임베딩을 계산하여 유사도 측정
def calculate_embedding_similarity(row):
    sentences = [row['correct sentence'].strip(), row['Seq2Seq'].strip()]
    embeddings = embedding_model.encode(sentences)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# 유사도 점수를 데이터프레임에 추가
similarity_df['similarity (Seq2Seq)'] = similarity_df.apply(calculate_embedding_similarity, axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

# 유사도 점수의 분포를 히스토그램으로 시각화
plt.figure(figsize=(10, 6))
sns.histplot(similarity_df['similarity (Seq2Seq)'], bins=10, kde=True, color='blue')
plt.title('Distribution of Similarity (Seq2Seq) Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 유사도가 0.9 이상인 항목의 개수와 전체 항목의 개수
count_above_threshold = (similarity_df['similarity (Seq2Seq)'] >= 0.9).sum()
total_count = len(similarity_df)

# 비율 계산
seq2seq_qpercentage_above_threshold = (count_above_threshold / total_count) * 100

# 결과 출력
print(f"0.9 이상인 유사도의 비율: {seq2seq_qpercentage_above_threshold:.2f}%")

# 결과를 새로운 DataFrame으로 생성
percentage_df = pd.DataFrame({
    'whisper': [whisper_percentage_above_threshold],
    'bert' : [bert_percentage_above_threshold],
    'symspell' : [symspell_percentage_above_threshold],
    'textblob' : [textblob_percentage_above_threshold]
    #'seq2seq' : [seq2seq_qpercentage_above_threshold]
})

percentage_df

# 바 그래프 그리기
plt.figure(figsize=(10, 6))  # 그래프의 크기 설정
sns.barplot(percentage_df)

# y축 레이블 설정
plt.ylabel('percentage above threshold (>0.9)')

# x축 레이블 설정
plt.xlabel('Spelling Correction Algorithm')

# 그래프 제목 설정
plt.title('Comparison of Methods for Deception Detection')

# 그래프를 보여줌
plt.show()
