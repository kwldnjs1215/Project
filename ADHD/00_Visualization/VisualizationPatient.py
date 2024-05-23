import pandas as pd

adhd = pd.read_csv(r'C:\Users\User\Downloads\Project\ADHD\ADHD_2018-2022.csv')


# 20~30대 연도별 환자수 시각화 (matplotlib)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

age_df = adhd.loc[(adhd["연령구분"] == "20~29세") | (adhd["연령구분"] == "30~39세")]
count_df = age_df.groupby(['진료년도', '연령구분'])['환자수'].sum()
count_df = count_df.groupby([count_df.index.get_level_values('진료년도').astype(str), count_df.index.get_level_values('연령구분')]).sum().unstack()
count_df.plot(ylabel='환자수', marker='o', cmap='coolwarm', alpha = 0.4)
plt.legend(["20대", "30대"], title="연령 구분")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

# 20~30대 연도별 환자수 시각화 (seaborn)
import seaborn as sn
age_df = adhd.loc[(adhd["연령구분"] == "20~29세") | (adhd["연령구분"] == "30~39세")]
count_df = age_df.groupby(['진료년도', '연령구분'])['환자수'].sum().reset_index()
count_df['연령구분'] = count_df['연령구분'].map({'20~29세': '20대', '30~39세':'30대'})
count_df['진료년도'] = count_df['진료년도'].astype(str)

sn.relplot(count_df, x="진료년도", y="환자수", hue="연령구분", kind="line", palette="coolwarm", marker='o')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 10~50대 연도별 환자수 시각화(seaborn)
adhd_df = adhd.groupby(['진료년도', '연령구분'])['환자수'].sum().reset_index()
adhd_df['진료년도'] = adhd_df['진료년도'].astype(str)
age_groups_to_drop = ['0~9세', '90~99세', '80~89세', '70~79세', '60~69세', '100세 이상']
adhd_df = adhd_df[~adhd_df['연령구분'].isin(age_groups_to_drop)]
adhd_df['연령구분'] = adhd_df['연령구분'].map({'10~19세': '10대', '20~29세':'20대', '30~39세':'30대', '40~49세': '40대', '50~59세': '50대'})

sn.relplot(adhd_df, x="진료년도", y="환자수", hue="연령구분", kind="line", palette="vlag", marker='o')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 20~50대 연도별 환자수 시각화
adhd_df = adhd.groupby(['진료년도', '연령구분'])['환자수'].sum().reset_index()
adhd_df['진료년도'] = adhd_df['진료년도'].astype(str)
age_groups_to_drop = ['10~19세', '0~9세', '90~99세', '80~89세', '70~79세', '60~69세', '100세 이상']
adhd_df = adhd_df[~adhd_df['연령구분'].isin(age_groups_to_drop)]
adhd_df['연령구분'] = adhd_df['연령구분'].map({'20~29세':'20대', '30~39세':'30대', '40~49세': '40대', '50~59세': '50대'})

sn.relplot(adhd_df, x="진료년도", y="환자수", hue="연령구분", kind="line", palette="vlag", marker='o')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 20~30대 성별에 따른 ADHD 환자수 비교
sex_df = adhd.loc[(adhd["연령구분"] == "20~29세") | (adhd["연령구분"] == "30~39세")].groupby(['연령구분', '성별'])['환자수'].sum().reset_index()
sn.barplot(sex_df, x="연령구분", y="환자수", hue="성별", palette="vlag")

# 20~50대 성별에 따른 ADHD 환자수 비교
sex_df = adhd.groupby(['연령구분', '성별'])['환자수'].sum().reset_index()
age_groups_to_drop = ['10~19세', '0~9세', '90~99세', '80~89세', '70~79세', '60~69세', '100세 이상']
sex_df = sex_df[~sex_df['연령구분'].isin(age_groups_to_drop)]
sn.barplot(sex_df, x="연령구분", y="환자수", hue="성별", palette="vlag")
