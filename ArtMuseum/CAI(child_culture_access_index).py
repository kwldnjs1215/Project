import pandas as pd
import re


pd.set_option('display.max_rows', 10)  
pd.set_option('display.max_columns', 10)  

path = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\derived_variable'

####################################################################################################

# 미술관 데이터 로드
df = pd.read_csv(r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\museum_data\museum_list_data.csv', encoding='utf-8')
                 
df

# 면적 데이터 로드
area = pd.read_csv(path + '\도시지역면적_시도_시_군_구__20240708155815.csv', encoding='utf-8')

# 필요한 컬럼만 선택
area = area[['지역', '소재지(시군구)별', '2022']]
area.columns = ['CTPRVN_NM', 'SIGNGU_NM', '면적']

# 공백 제거
area['SIGNGU_NM'] = area['SIGNGU_NM'].apply(lambda x: x.strip())
area

# 미술관 데이터와 면적 데이터를 병합
merged_df = pd.merge(df, area, on=['CTPRVN_NM', 'SIGNGU_NM'], how='left')

# 면적 데이터가 누락된 행 확인
missing_area = merged_df[merged_df['면적'].isnull()]
missing_facilities = missing_area['FCLTY_NM']

# 누락된 시설을 제외한 데이터로 필터링
filtered_merge = merged_df[~merged_df['FCLTY_NM'].isin(missing_facilities)]
merged_df = filtered_merge.copy()

# 면적 데이터를 정수형으로 변환
merged_df['면적'] = merged_df['면적'].str.replace(',', '').astype(int)

merged_df

####################################################################################################

# 첫 번째 문화 시설 데이터 로드
culture = pd.read_csv(path + '/전국_아이랑_문화시설_위치(2023).csv')
culture2 = pd.read_csv(path + '/전국_가족_유아_동반_가능_문화시설_위치_데이터(2023).csv')

# 첫 번째 데이터셋 처리
culture['SIGNGU_NM'] = culture['SIGNGU_NM'].str.split().str[0]
culture_filtered = culture[culture['CTPRVN_NM'] != '서울특별시']
culture_filtered = culture_filtered[culture_filtered['FCLTY_NM'].str.contains('미술관')]
culture_cnt = culture_filtered.groupby(['CTPRVN_NM', 'SIGNGU_NM']).size().reset_index(name='kids_culture_cnt')
culture_cnt

# 두 번째 데이터셋 처리
culture2['SIGNGU_NM'] = culture2['SIGNGU_NM'].str.split().str[0]
culture2_filtered = culture2[(culture2['CTGRY_THREE_NM'] == '대형미술관') | (culture2['CTGRY_THREE_NM'] == '미술관')]
culture2_filtered = culture2_filtered[culture2_filtered['CTPRVN_NM'] != '서울특별시']
culture2_cnt = culture2_filtered.groupby(['CTPRVN_NM', 'SIGNGU_NM']).size().reset_index(name='kids_culture_cnt')
culture2_cnt

# 두 결과를 병합
culture_merge = pd.merge(culture_cnt, culture2_cnt, on=['CTPRVN_NM', 'SIGNGU_NM'], how='outer', suffixes=('_culture', '_culture2'))

# NaN 값을 0으로 채움
culture_merge = culture_merge.fillna(0)

# 두 개수를 합산
culture_merge['어린이 미술관 수(어린이 문화시설 접근성 지수)'] = culture_merge['kids_culture_cnt_culture'] + culture_merge['kids_culture_cnt_culture2']

# 필요한 컬럼만 선택
fn_culture_df = culture_merge[['CTPRVN_NM', 'SIGNGU_NM', '어린이 미술관 수(어린이 문화시설 접근성 지수)']].copy()
fn_culture_df['어린이 미술관 수(어린이 문화시설 접근성 지수)'] = fn_culture_df['어린이 미술관 수(어린이 문화시설 접근성 지수)'].astype(int)
fn_culture_df.columns = ['CTPRVN_NM', 'SIGNGU_NM', '어린이 미술관 수(어린이 문화시설 접근성 지수)']

# 미술관 데이터에 어린이 미술관 수 데이터 병합
merged_df = pd.merge(merged_df, fn_culture_df, on=['CTPRVN_NM', 'SIGNGU_NM'], how='left')
merged_df = merged_df.fillna(0)
merged_df

####################################################################################################

# 공연 정보 데이터 로드
event = pd.read_csv(path + '\한국문화정보원_어린이를_위한_공연정보.csv')

# 'grade' 컬럼에 결측치가 있는 행 제거
event.dropna(subset=['grade'], inplace=True)

# 중학생 이상, 고등학생 이상, 청소년 등 대상의 공연을 제외한 데이터 필터링
df_grade = event[~(event['grade'].str.contains('중학생') & event['grade'].str.contains('이상'))]
df_grade = df_grade[~(df_grade['grade'].str.contains('고등학생') & df_grade['grade'].str.contains('이상'))]
df_grade = df_grade[~(df_grade['grade'].str.contains('청소년'))]
df_grade = df_grade[~(df_grade['grade'].str.contains('중·고등'))]
df_grade = df_grade[~(df_grade['grade'].str.contains('일반인'))]
df_grade = df_grade[~(df_grade['grade'].str.contains('\d{2,}(?:세|이상)'))]

# '미술' 관련 공연만 필터링
df_art = df_grade[df_grade['category'].str.contains('미술')].copy()

# 시도명과 시군구명을 분리하여 새로운 컬럼 생성
df_art['CTPRVN_NM'] = df_art['address2'].str.split().str[0]
df_art['SIGNGU_NM'] = df_art['address2'].str.split().str[1]

# 시도별, 시군구별 공연 수 계산
art_cnt = df_art.groupby(['CTPRVN_NM', 'SIGNGU_NM']).size().reset_index(name='어린이 미술관련 공연 수(어린이 문화시설 접근성 지수)')
art_cnt.columns = ['CTPRVN_NM', 'SIGNGU_NM', '어린이 미술관련 공연 수(어린이 문화시설 접근성 지수)']
art_cnt

# 기존 데이터에 공연 수 데이터 병합
merged_df = pd.merge(merged_df, art_cnt, on=['CTPRVN_NM', 'SIGNGU_NM'], how='left')
merged_df = merged_df.fillna(0)
merged_df

####################################################################################################

# 학교 정보 데이터 로드
school  = pd.read_csv(path + '\학교기본정보_2022년12월31일기준.csv', encoding='cp949')

# 시도명과 시군구명을 분리하여 새로운 컬럼 생성
school['CTPRVN_NM'] = school['도로명주소'].str.split().str[0]
school['SIGNGU_NM'] = school['도로명주소'].str.split().str[1]

# 시도별, 시군구별 초등학교 수 계산
school_cnt = school.groupby(['CTPRVN_NM', 'SIGNGU_NM']).size().reset_index(name='초등학교 수(어린이 문화시설 접근성 지수)')
school_cnt.columns = ['CTPRVN_NM', 'SIGNGU_NM', '초등학교 수(어린이 문화시설 접근성 지수)']

# 기존 데이터에 초등학교 수 데이터 병합
merged_df = pd.merge(merged_df, school_cnt, on=['CTPRVN_NM', 'SIGNGU_NM'], how='left')
merged_df = merged_df.fillna(0)
merged_df

####################################################################################################

# 유치원 수 데이터 로드
kindergarten = pd.read_csv(path + '\유치원_수_시도_시_군_구__20240709141437.csv', encoding='cp949')
kindergarten = kindergarten[['지역', '행정구역별', '2022']]
kindergarten.columns = ['CTPRVN_NM', 'SIGNGU_NM', '유치원 수(어린이 문화시설 접근성 지수)']

# 기존 데이터에 유치원 수 데이터 병합
merged_df = pd.merge(merged_df, kindergarten, on=['CTPRVN_NM', 'SIGNGU_NM'], how='left')
merged_df = merged_df.fillna(0)
merged_df

####################################################################################################

# 어린이 문화시설 접근성 지수 계산
merged_df['어린이 문화시설 접근성 지수'] = ((merged_df['어린이 미술관 수(어린이 문화시설 접근성 지수)'] +
                                      merged_df['어린이 미술관련 공연 수(어린이 문화시설 접근성 지수)'] +
                                      merged_df['초등학교 수(어린이 문화시설 접근성 지수)'] +
                                      merged_df['유치원 수(어린이 문화시설 접근성 지수)'])
                                      / merged_df['면적'])

# 결과 데이터에 결측치가 있을 경우 0으로 채움
fn_df = merged_df.fillna(0)
fn_df

# 결과를 CSV 파일로 저장
fn_df.to_csv('child_culture_access_index.csv', index=False, encoding='euc-kr')
