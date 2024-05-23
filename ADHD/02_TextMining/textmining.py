import pandas as pd
import re
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
font_path = 'C:/Windows/Fonts/malgun.ttf'

###########################
########## Konly ##########
###########################
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# text 불러오기
text_file = 'Project\ADHD\02_TextMining\kor_url_text.txt'

with open(text_file, 'r', encoding='utf-8') as file:
    text_data = file.read()

# 일반 명사 추출 함수
def extract_nouns(text):

    cleaned_text = re.sub(r'[^\w\s]', '', text)

    kkma = Kkma()
    pos_tags = kkma.pos(cleaned_text)

    nouns = []
    for word, pos in pos_tags:
        # 일반 명사 추출 한 글자 단어 제외
        if pos == 'NNG' and re.match(r'^[가-힣]{2,}$', word):
            nouns.append(word)
    
    # 명사별 빈도수 데이터프레임화
    nouns_series = pd.Series(nouns)
    nouns_freq_df = nouns_series.value_counts().reset_index()
    nouns_freq_df.columns = ['Nouns', 'Frequency']

    return nouns_freq_df

nouns_freq_df = extract_nouns(text_data)

# 증상별 그룹화 함수
def grouping(text_data):
    # 중요 단어 선택
    attention_importance_word = ['시간', '계획', '습관', '관리',
                                 '체크', '반복', '구체적', '정리',
                                 '지속', '보상', '일정', '체계', '세분화', '규칙', '효율', '매뉴얼',
                                 '계획', '칭찬', '루틴', '순서', '우선순위', '메모']

    hyper_activity_importance_word = ['음악', '명상', '운동',
                                      '신체', '연주', '휴식', '클래식',
                                      '다이어리', '수면']

    impulsivity_importance_word = ['사회적', '표현', '심리', '감정',
                                   '이야기', '억제', '노래', '말하기']
    
    # 내용을 저장할 새로운 리스트 생성
    selected_sentences_attention = []; words_attention = [];
    selected_sentences_hyper = []; words_hyper = [];
    selected_sentences_impulsivity = []; words_impulsivity = [];
    
    # 줄바꿈을 없애고 "."로 스플릿하여 문장들을 리스트에 저장
    sentences = text_data.replace('\n', '').split('.')
    sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]

    # 데이터프레임으로 변환
    df = pd.DataFrame({'sentence': sentences})

    # attention_importance_word에 대한 그룹화
    for word in attention_importance_word:
        # 단어가 포함된 문장 선택
        selected = df[df['sentence'].str.contains(word)]
        selected_sentences_attention.extend(selected['sentence'])
        words_attention.extend([word] * len(selected))

    # hyper_activity_importance_word에 대한 그룹화
    for word in hyper_activity_importance_word:
        # 단어가 포함된 문장 선택
        selected = df[df['sentence'].str.contains(word)]
        selected_sentences_hyper.extend(selected['sentence'])
        words_hyper.extend([word] * len(selected))

    # impulsivity_importance_word에 대한 그룹화
    for word in impulsivity_importance_word:
        # 단어가 포함된 문장 선택
        selected = df[df['sentence'].str.contains(word)]
        selected_sentences_impulsivity.extend(selected['sentence'])
        words_impulsivity.extend([word] * len(selected))

    # 새로운 리스트를 DataFrame으로 결합
    result_df_attention = pd.DataFrame({'word': words_attention, 'sentence': selected_sentences_attention})
    result_df_hyper = pd.DataFrame({'word': words_hyper, 'sentence': selected_sentences_hyper})
    result_df_impulsivity = pd.DataFrame({'word': words_impulsivity, 'sentence': selected_sentences_impulsivity})

    # word를 기준으로 그룹화하여 sentence를 모두 합치기
    grouped_df_attention = result_df_attention.groupby('word', sort=False)['sentence'].apply(' '.join).reset_index()
    grouped_df_hyper = result_df_hyper.groupby('word', sort=False)['sentence'].apply(' '.join).reset_index()
    grouped_df_impulsivity = result_df_impulsivity.groupby('word', sort=False)['sentence'].apply(' '.join).reset_index()

    return attention_importance_word, hyper_activity_importance_word, impulsivity_importance_word, grouped_df_attention, grouped_df_hyper, grouped_df_impulsivity

attention_importance_word, hyper_activity_importance_word, impulsivity_importance_word, grouped_df_attention, grouped_df_hyper, grouped_df_impulsivity = grouping(text_data)


####################### 부주의 #######################

# 제외할 단어들
excluded_words = {"연구", "프로그램", "모듈"}

for word in attention_importance_word:
    input_text = word

    input_columns = grouped_df_attention[grouped_df_attention['word'] == word]['sentence'].values
    if len(input_columns) == 0:
        print(f"No sentences found for word: {word}\n")
        continue

    input_sentences = ' '.join(input_columns)

    input_sentences = input_sentences.replace('\n', '').split('.')
    input_sentences = [sentence.strip() + '.' for sentence in input_sentences if sentence.strip()]

    # 제외할 단어가 포함된 문장은 필터링
    filtered_sentences = [sentence for sentence in input_sentences if not any(excluded_word in sentence for excluded_word in excluded_words)]

    # 제외 후 문장이 없는 경우 건너뜀
    if not filtered_sentences:
        print(f"No sentences left after filtering for word: {word}\n")
        continue

    # TF 벡터화 객체를 생성
    vectorizer = CountVectorizer()

    # 입력 텍스트와 필터링된 문장들을 합쳐서 벡터화
    texts = [input_text] + filtered_sentences
    tf_matrix = vectorizer.fit_transform(texts)

    # 입력 텍스트의 TF 벡터
    input_tf_vector = tf_matrix[0]

    # 각 문장의 TF 벡터와 입력 텍스트의 TF 벡터 간의 코사인 유사도를 계산
    similarity_scores = []
    for i in range(1, len(texts)):
        sentence_tf_vector = tf_matrix[i]
        similarity_score = cosine_similarity(input_tf_vector, sentence_tf_vector)[0][0]
        similarity_scores.append(similarity_score)

    # 가장 유사도가 높은 상위 3개 문장의 인덱스
    top_3_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:3]
    top_3_sentences = [filtered_sentences[idx] for idx in top_3_indices]

    print(word, top_3_sentences)
    

####################### 과잉행동 #######################

# 제외할 단어들
excluded_words = {"연구", "프로그램", "모듈"}

for word in hyper_activity_importance_word:
    input_text = word

    input_columns = grouped_df_hyper[grouped_df_hyper['word'] == word]['sentence'].values
    if len(input_columns) == 0:
        print(f"No sentences found for word: {word}\n")
        continue

    input_sentences = ' '.join(input_columns)

    input_sentences = input_sentences.replace('\n', '').split('.')
    input_sentences = [sentence.strip() + '.' for sentence in input_sentences if sentence.strip()]

    # 제외할 단어가 포함된 문장은 필터링
    filtered_sentences = [sentence for sentence in input_sentences if not any(excluded_word in sentence for excluded_word in excluded_words)]

    # 제외 후 문장이 없는 경우 건너뜀
    if not filtered_sentences:
        print(f"No sentences left after filtering for word: {word}\n")
        continue

    # TF 벡터화 객체를 생성
    vectorizer = CountVectorizer()

    # 입력 텍스트와 필터링된 문장들을 합쳐서 벡터화
    texts = [input_text] + filtered_sentences
    tf_matrix = vectorizer.fit_transform(texts)

    # 입력 텍스트의 TF 벡터
    input_tf_vector = tf_matrix[0]

    # 각 문장의 TF 벡터와 입력 텍스트의 TF 벡터 간의 코사인 유사도를 계산
    similarity_scores = []
    for i in range(1, len(texts)):
        sentence_tf_vector = tf_matrix[i]
        similarity_score = cosine_similarity(input_tf_vector, sentence_tf_vector)[0][0]
        similarity_scores.append(similarity_score)

    # 가장 유사도가 높은 상위 3개 문장의 인덱스
    top_3_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:3]
    top_3_sentences = [filtered_sentences[idx] for idx in top_3_indices]

    print(word, top_3_sentences)


####################### 충동성 #######################

# 제외할 단어들
excluded_words = {"연구", "프로그램", "모듈"}

for word in impulsivity_importance_word:
    input_text = word

    input_columns = grouped_df_impulsivity[grouped_df_impulsivity['word'] == word]['sentence'].values
    if len(input_columns) == 0:
        print(f"No sentences found for word: {word}\n")
        continue

    input_sentences = ' '.join(input_columns)

    input_sentences = input_sentences.replace('\n', '').split('.')
    input_sentences = [sentence.strip() + '.' for sentence in input_sentences if sentence.strip()]

    # 제외할 단어가 포함된 문장은 필터링
    filtered_sentences = [sentence for sentence in input_sentences if not any(excluded_word in sentence for excluded_word in excluded_words)]

    # 제외 후 문장이 없는 경우 건너뜀
    if not filtered_sentences:
        print(f"No sentences left after filtering for word: {word}\n")
        continue

    # TF 벡터화 객체를 생성
    vectorizer = CountVectorizer()

    # 입력 텍스트와 필터링된 문장들을 합쳐서 벡터화
    texts = [input_text] + filtered_sentences
    tf_matrix = vectorizer.fit_transform(texts)

    # 입력 텍스트의 TF 벡터
    input_tf_vector = tf_matrix[0]

    # 각 문장의 TF 벡터와 입력 텍스트의 TF 벡터 간의 코사인 유사도를 계산
    similarity_scores = []
    for i in range(1, len(texts)):
        sentence_tf_vector = tf_matrix[i]
        similarity_score = cosine_similarity(input_tf_vector, sentence_tf_vector)[0][0]
        similarity_scores.append(similarity_score)

    # 가장 유사도가 높은 상위 3개 문장의 인덱스
    top_3_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:3]
    top_3_sentences = [filtered_sentences[idx] for idx in top_3_indices]

    print(word, top_3_sentences)

###########################
######## WordCloud ########
###########################
from wordcloud import WordCloud

choice_words = ['시간', '계획', '습관', '관리',
                '체크', '반복', '구체적', '정리',
                '지속', '보상', '일정', '체계', '세분화', '규칙', '효율', '매뉴얼',
                '계획', '칭찬', '루틴', '순서', '우선순위', '메모', '명상', '운동',
                '신체', '연주', '휴식', '클래식',
                '다이어리', '수면', '사회적', '표현', '심리', '감정',
                '이야기', '억제', '노래', '말하기']

vis_nouns_df = nouns_freq_df[nouns_freq_df['Nouns'].isin(choice_words)]

# nouns_freq_df를 딕셔너리로 변환
word_freq_dict = dict(zip(vis_nouns_df['Nouns'], vis_nouns_df['Frequency']))

# 워드클라우드 생성
wordcloud = WordCloud(font_path=font_path, background_color='white').generate_from_frequencies(word_freq_dict)


# 워드클라우드 시각화
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()








