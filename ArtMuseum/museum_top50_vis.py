import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# CSV 파일 불러오기
file_name = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\museum_data\museum_list_top50_vis.csv'
museum_data = pd.read_csv(file_name, encoding='utf-8')

# VIEWNG_NMPR_CO(방문객 수) 기준 내림차순으로 정렬하고 상위 50개 선택
top_50_museums = museum_data.sort_values(by='VIEWNG_NMPR_CO', ascending=False).head(50)

# Matplotlib에서 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 시스템에 맞게 폰트 경로 조정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 그라디언트 스타일의 색상 팔레트를 사용하여 Seaborn으로 시각화
plt.figure(figsize=(14, 10))
sns.barplot(x='VIEWNG_NMPR_CO', y='FCLTY_NM', data=top_50_museums, palette='coolwarm')  # 'coolwarm' 팔레트 사용
plt.xlabel('방문객 수', fontsize=14)  # x축 레이블 한글 설정 (방문객 수)
plt.ylabel('', fontsize=14)  # y축 레이블 한글 설정 (미술관 이름)
plt.title('방문객 수 Top 50 미술관', fontsize=16)  # 그래프 제목 한글 설정 (방문객 수 Top 50 미술관)
plt.tight_layout()
plt.show()
