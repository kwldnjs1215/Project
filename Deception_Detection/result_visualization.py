import seaborn as sns
import matplotlib.pyplot as plt

# 음성 모델
preds_df.to_csv('/content/drive/MyDrive/Project/기만탐지모델/preds_table.csv', index=False)

# 선 그래프 그리기
sns.lineplot(x=preds_df.index, y=preds_df['adjusted_score'])

# y축 이름 설정
plt.ylabel('Adjusted Score')

# x축 이름 제거
plt.xlabel('')

# 그래프 제목 설정
plt.title('Voice-based Deception Detection Results')

# 그래프를 보여줌
plt.show()

########################################################################

# 영상 모델
import seaborn as sns
import matplotlib.pyplot as plt

face_detect_df = pd.read_csv('/content/drive/MyDrive/Project/기만탐지모델/data_with_truth_scores.csv')

face_detect_df['adjusted_score'] = (((face_detect_df['Truth Score'] - 50)*2) - 50) * -1

# 선 그래프 그리기
sns.lineplot(x=face_detect_df.index, y=face_detect_df['adjusted_score'])

# y축 이름 설정
plt.ylabel('Adjusted Score')

# x축 이름 제거
plt.xlabel('')

# 그래프 제목 설정
plt.title('Video-based Deception Detection Results')

# 그래프를 보여줌
plt.show()
