# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:37:12 2024

@author: itwill
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

####### 10대 미만 스마트폰 보급률 #######
df = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/1. 10대미만 스마트폰 보급율.csv')

# '유형' 열에서 '키즈폰' 행 제거
df = df[df['유형'] != '키즈폰']

# '응답' 열에서 '응답자수 (명)'과 '2대 이상 (%)' 행 제거
df = df[~df['응답'].isin(['응답자수 (명)', '2대 이상 (%)'])]

# '연도'와 '응답'으로 피벗 테이블 생성
df_grouped = df.pivot(index='연도', columns='응답', values='만10대미만')

# 그래프 그리기
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_grouped, palette='coolwarm', markers=True, dashes=False)
plt.title('연도별 스마트폰 보급율 (만 10대 미만)')
plt.xlabel('연도')
plt.ylabel('비율 (%)')
plt.legend(title='응답')
plt.rcParams['font.family'] = 'Malgun Gothic' # 사용할 글꼴 
plt.rcParams['axes.unicode_minus'] = False # 음수 부호 지원
plt.grid(True)
plt.show()


####### 시별 어린이 센터 비율 #######
df2 = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/2. 시별 어린이센터 비율.csv')


# 'ratio' 열 기준으로 내림차순 정렬
df2_sorted = df2.sort_values(by='ratio', ascending=False)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(data=df2_sorted, x='ratio', y='도시', hue='구분', palette='coolwarm', linewidth=1.5)  # 막대 굵기 설정
plt.title('시별 어린이 센터 비율', fontsize=15)
plt.xlabel('비율', fontsize=12)  # x축 레이블을 '비율'로 변경
plt.ylabel('도시', fontsize=12)
plt.xticks(rotation=90)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.legend(title='구분', loc='best')
plt.show()


###### 구분이 '남부'인 데이터 필터링 ######
# 구분이 '남부'인 데이터 필터링
df_south = df2[df2['구분'] == '남부']

df_south = df_south.sort_values(by='ratio', ascending=False)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(data=df_south, x='ratio', y='도시', palette='coolwarm')
plt.title('경기 남부 어린이 센터 비율', fontsize=15)
plt.xlabel('ratio', fontsize=12)
plt.ylabel('도시', fontsize=12)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.xticks(rotation=0)
plt.show()

###### 구분이 '북부'인 데이터 필터링 ######
# 구분이 '북부'인 데이터 필터링
df_south = df2[df2['구분'] == '북부']

df_south = df_south.sort_values(by='ratio', ascending=True)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(data=df_south, x='ratio', y='도시', palette='coolwarm')
plt.title('경기 북부 어린이 센터 비율', fontsize=15)
plt.xlabel('ratio', fontsize=12)
plt.ylabel('도시', fontsize=12)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.xticks(rotation=0)
plt.show()




###### 시별 어린이 센터 현황 ######
df3 = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/3. 시별 어린이 센터 현황.csv', encoding='cp949')

df3

# 시각화
# 구분별로 데이터 수 집계
grouped = df3.groupby('구분').size().reset_index(name='count')

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='구분', y='count', palette='coolwarm', width= 0.2)
plt.title('경기도 북부/남부 별 어린이 시설 수', fontsize=15)
plt.xlabel('구분', fontsize=12)
plt.ylabel('어린이 시설 수', fontsize=12)
plt.xticks(rotation=0)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.show()



###### 시별 신혼부부 현황 ######
df4 = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/4. 경기도_신혼부부_수.csv', encoding = 'cp949')

df4
df4['구분'] = df4['지역'].apply(lambda x: '북부' if '고양' in x or '의정부' in x or '파주시' in x or '양주' in x or '구리' in x or '남양주' in x or '동두천' in x or '포천' in x or '가평' in x or'연천' in x else '남부')

df4
###### 구분이 '북부'인 데이터 필터링 ######
# 구분이 '북부'인 데이터 필터링
df_south4 = df4[df4['구분'] == '북부']

df_south4 = df_south4.sort_values(by='X2022', ascending=False)


plt.figure(figsize=(12, 8))
sns.barplot(data=df_south4, x='X2022', y='지역', palette='coolwarm')
plt.title('경기 북부 신혼부부 현황', fontsize=15)
plt.xlabel('신혼부부', fontsize=12)
plt.ylabel('도시', fontsize=12)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.xticks(rotation=0)
plt.show()

'''
# 꺾은선 그래프 그리기
plt.figure(figsize=(12, 8))

# 각 연도별 데이터를 꺾은선 그래프로 표시
plt.plot(df4['지역'], df4['X2018'], marker='o', label='2018')
plt.plot(df4['지역'], df4['X2019'], marker='o', label='2019')
plt.plot(df4['지역'], df4['X2020'], marker='o', label='2020')
plt.plot(df4['지역'], df4['X2021'], marker='o', label='2021')
plt.plot(df4['지역'], df4['X2022'], marker='o', label='2022')

plt.title('지역별 연도별 데이터', fontsize=15)
plt.xlabel('지역', fontsize=12)
plt.ylabel('값', fontsize=12)
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()
'''
###### 북부 9세 미만 어린이 현황 ######
df5 = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/9세 미만_경기도_남북구분_시별 총합.csv')
df5['2023'] = df5['2023'].str.replace(',', '').astype(int)

df_south5 = df5[df5['구분'] == '북부']

df_south5 = df_south5.sort_values(by='2023', ascending=False)
df_south5


plt.figure(figsize=(12, 8))
sns.barplot(data=df_south5, x='2023', y='도시', palette='coolwarm')
plt.title('경기 북부 9세 미만 어린이 현황', fontsize=15)
plt.xlabel('어린이 수', fontsize=12)
plt.ylabel('도시', fontsize=12)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.xticks(rotation=0)
plt.show()


###### 10대 미만 어플 사용율 ######
df6 = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/6. 10대미만 어플리케이션 사용율.csv') 
df6 = df6[df6['만10대미만'] != 0]
df6 = df6[(df6['유형'] == '게임 (%)') | 
         (df6['유형'] == '방송/동영상(OTT 서비스 등) (%)') |
         (df6['유형'] == '인스턴트메신저(카카오톡, 라인, 페이스북 등의 SNS 자체 메신저 등) (%)') |
         (df6['유형'] == '교육(인터넷 강의, 사전/번역, 학교 등) (%)')]

pivot_df = df6.pivot(index='연도', columns='유형', values='만10대미만')

# 꺾은선 그래프 그리기
plt.figure(figsize=(12, 8))


# 모든 유형에 대해 꺾은선 그래프 그리기
for col in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[col], marker='o', label=col)

plt.title('연도별 10대 미만 어플 사용 현황', fontsize=15)
plt.xlabel('연도', fontsize=12)
plt.ylabel('값', fontsize=12)
plt.xticks(pivot_df.index)  # 연도를 x축에 표시
plt.legend(loc='best', fontsize=10)
plt.grid(True)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 사용할 한글 글꼴 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 지원
plt.show()


######  ######
df7 = pd.read_csv(r'C:/Users/itwill/Desktop/문화공모전/7. 한국문화정보원_어린이를_위한_공연정보_시각화.csv') 
df7 = df7.sort_values(by='number', ascending=False)

# 막대 그래프 그리기
plt.figure(figsize=(12, 8))
sns.barplot(data=df7, x='category', y='number', palette='coolwarm',  width=0.5)
plt.title('카테고리 별 어린이 공연정보', fontsize=15)
plt.xlabel('카테고리', fontsize=12)
plt.ylabel('숫자', fontsize=12)
plt.xticks(rotation=0)

# 막대 위에 숫자 표시하기
for index, value in enumerate(df7['number']):
    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=10)

plt.show()
