import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 데이터 로드
a = pd.read_csv('/Users/louis_wiz/Downloads/modify_variable_list.csv', encoding='utf-8')
a.info()

# 특징 데이터와 타겟 변수 설정
ax = a.iloc[:, 6:].drop('population', axis=1)  # 인구수 열을 제외한 데이터
ay = a['VIEWNG_NMPR_CO'] / a['population']  # 인구 대비 관람객 수 비율

# 박스 플롯 생성
sns.boxplot(ay)
plt.title('상자 그림')
plt.show()

# 이상치 계산
IQR = ay.quantile(0.75) - ay.quantile(0.25)
ay.quantile(0.75) + 1.5 * IQR

# 상관 분석
b = pd.concat([ax, ay], axis=1)
b.rename(columns={0: 'popularity'}, inplace=True)  # 타겟 변수 이름 변경
b = b.loc[b['popularity'] >= 0.4409011584319073, :]  # 특정 임계값 이상인 데이터만 선택
b.info()

# 모델 학습을 위한 독립 변수와 종속 변수 설정
X = b.drop('popularity', axis=1)
X = X.iloc[:, 12:]  # 마지막 12개 열만 선택
y = b['popularity']
X.info()

# 데이터셋을 훈련셋과 테스트셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 데이터프레임으로 변환
X_train = pd.DataFrame(X_train, columns=[f'x{i+1}' for i in range(X_train.shape[1])])
X_test = pd.DataFrame(X_test, columns=[f'x{i+1}' for i in range(X_test.shape[1])])

# 상수 항 추가
X_train_with_const = sm.add_constant(X_train)
X_test_with_const = sm.add_constant(X_test)

# 훈련 데이터에서 상관 행렬 계산
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr()

# 상관 행렬 히트맵 시각화
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('상관 행렬')
plt.show()
