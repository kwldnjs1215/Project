# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:55:20 2024

@author: minjeong
"""

file1 = r'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터원본/파생변수 추가해주세요.csv'
file2 = r'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터원본/어린이_전체_총인구수.csv'
file3 = r'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터원본/시군구별_신혼부부_수_2022.csv'

import pandas as pd 

museum = pd.read_csv(file1)
pop = pd.read_csv(file2, encoding= 'EUC-KR')
married = pd.read_csv(file3, encoding= 'EUC-KR')
museum.info() # 도
# 도 # 군/구


#############국립 박물관

museum['군/구'] = museum['군/구'].str.split(' ',1).str[0]
museum['군/구'] 
museum = museum[museum['군/구'] != '공원순환로']
museum[museum['군/구']=='공원순환로'] = '달석구'
museum.loc[museum['군/구'] == '영월군,읍', '군/구'] = '영월군'
museum[museum['FCLTY_NM'] == '국제현대미술관']['군/구']
museum[museum['군/구']=='영월군,읍']
#############신혼부부 쌍(22년12월) 전처리
married.info()

married = married[married['행정구역별(2)'] != '소계']
married.columns = ['도','군/구','쌍']
married.shape #  (261, 3)
# 대한민국 시군구 개수 261개

#############어린이 수 (22년12월 0~9세) 전처리

pop.info()
pop.행정구역
pop = pop[['행정구역','2022년_계_총인구수','2022년_계_0~9세']]

pop[['도','군/구']] = pop['행정구역'].str.split(' ',1, expand = True)
pop.drop('행정구역', axis = 1, inplace = True)

pop['군/구'] = pop['군/구'].str.split(' ',1).str[0]
pop['군/구']  # 292
pop.dropna(axis =0 , inplace = True)
pop.isnull().sum()
# ~출장소 삭제 
pop[pop['도']=='동해출장소'] # 0 
pop = pop[(pop['도'] !='북부출장소')|(pop['도'] !='동부출장소')]

pop['군/구'].value_counts(ascending = False).head(20)
pop.도 =pop['도'].replace('강원특별자치도','강원도')

pop.도

museum.shape

pop.to_csv('C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/전국_인구수(어린이및전체).csv', index = False)
married.to_csv('C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/전국_신혼부부수.csv', index = False)
museum.to_csv('C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/국립박물관.csv', index = False)

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:05:03 2024

@author: minjeong
"""

import pandas as pd
file1 = 'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/전국_인구수(어린이및전체).csv'
file2 = 'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/전국_신혼부부수.csv'
file3 = 'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/국립박물관.csv'


married = pd.read_csv(file2)
pop = pd.read_csv(file1)
museum = pd.read_csv(file3)


museum[museum['FCLTY_NM'] == '국제현대미술관']['군/구']

pop.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 291 entries, 0 to 290
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   총인구수       291 non-null    object
 1   어린이수       291 non-null    object
 2   도  291 non-null    object
 3   군/구  271 non-null    object
dtypes: object(4)
memory usage: 9.2+ KB
'''
pop.dropna(axis =0 , inplace = True)
pop
married.info() # not null 

museum['region'] = museum['도'] + ' ' + museum['군/구']
married['region'] = married['도'] + ' ' + married['군/구']
pop['region'] = pop['도'] + ' ' + pop['군/구']


museum['region']
married['region'] 
pop['region']

married = married.rename(columns={'쌍': '신혼부부_쌍'})
pop = pop.rename(columns={'2022년_계_총인구수': '지역인구수', '2022년_계_0~9세':'어린이수'})
married = married.drop(['도','군/구'], axis = 1)
pop = pop.drop(['도','군/구'], axis = 1)


pop.region.nunique() # 239
married.region.nunique() # 261

merge_df = pd.merge(pop, married, on='region',how = 'left' )
merge_df.columns
merge_df.info()


merge_df.isnull().sum() 
'''
지역인구수      0
어린이수       0
region     0
신혼부부_쌍    56
dtype: int64
'''

merge_df = merge_df.fillna(0)

def remove_comma(x):
    if isinstance(x, str):
        return int(x.replace(',', ''))
    else:
        return x

merge_df['어린이수'] = merge_df['어린이수'].apply(remove_comma)
merge_df['신혼부부_쌍'] = merge_df['신혼부부_쌍'].apply(remove_comma)
merge_df['지역인구수'] = merge_df['지역인구수'].apply(remove_comma)


merge_df['어린이친화지수'] = (merge_df['어린이수']+merge_df['신혼부부_쌍'])/merge_df['지역인구수']

merge_df = merge_df.loc[:,['region','어린이수','신혼부부_쌍','지역인구수','어린이친화지수']]
merge_df.to_csv('C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/merged_df.csv', index = False)


'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 271 entries, 0 to 270
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   지역인구수   271 non-null    int64  
 1   어린이수    271 non-null    int64  
 2   region  271 non-null    object 
 3   신혼부부_쌍  271 non-null    int64  
 4   파생변수    260 non-null    float64
dtypes: float64(1), int64(3), object(1)
memory usage: 12.7+ KB
'''

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:42:49 2024

@author: minjeong
"""
import pandas as pd

file1 =  r'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/merged_df.csv'
file2 = r'C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/국립박물관.csv'
merge_df = pd.read_csv(file1)
museum = pd.read_csv(file2)

museum['region'] = museum['도'] + ' ' + museum['군/구']

merge_df.columns

merge_df.shape
museum.shape
museum_merged = pd.merge(museum, merge_df, on='region',how = 'left' )
museum_merged.columns
museum_merged = museum_merged.loc[:,['region','FCLTY_NM', '도', '군/구', 'FCLTY_LA',
       'FCLTY_LO','어린이수', '신혼부부_쌍', '지역인구수', '어린이친화지수']]

museum_merged[museum_merged['FCLTY_NM'] == '국제현대미술관']['군/구']


museum_merged.to_csv('C:/Users/minjeong/OneDrive/문화_공모전/파생변수 만들기/데이터전처리/국립박물관(어린이친화지수_민정).csv', encoding='euc-kr', index = False)

museum_merged.shape
museum.tail(5)
museum_merged.tail(5)

merge_df['region'].value_counts()



