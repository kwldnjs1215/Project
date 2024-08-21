import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#########################################
#### Loading Questionnaire Data Form ####
#########################################

adhd = pd.read_csv(r'C:\Users\User\Downloads\Project\ADHD\성인_ADHD_설문지.csv')

#########################################
########### Data Preprocessing ##########
#########################################
columns_rename = ['시간', '전화번호', '성별', '연령']

for i in range(1, 19):
    q = f"Q{i}"
    columns_rename.append(q)

adhd.columns = columns_rename

adhd_df = adhd.drop(['성별', '연령', '시간', '전화번호'], axis=1)

#########################################
############# ADHD score ################
#########################################
def ADHD_score(adhd_df):
    adhd_risk = []
    adhd_score = ''
    for i in range(len(adhd_df)):
        cnt = 0
        for j in range(18):
            if adhd_df.iloc[i, j] == "전혀 그렇지 않다":
                adhd_df.iloc[i, j] = 0
            elif adhd_df.iloc[i, j] == "드물게 그렇다":
                adhd_df.iloc[i, j] = 1
            elif adhd_df.iloc[i, j] == "가끔 그렇다":
                adhd_df.iloc[i, j] = 2
            elif adhd_df.iloc[i, j] == "보통 그렇다":
                adhd_df.iloc[i, j] = 3
            elif adhd_df.iloc[i, j] == "매우 자주 그렇다":
                adhd_df.iloc[i, j] = 4

        # 가중치 부여
        for k in range(18):
            if k in [0, 1, 2, 8, 11, 15, 17]:
                if adhd_df.iloc[i, k] in [2, 3, 4]:
                    cnt += 1
            else:
                if adhd_df.iloc[i, k] in [3, 4]:
                    cnt += 1

        # ADHD 위험군 판단
        if cnt >= 4:
            adhd_score = 1
        else:
            adhd_score = 0
        adhd_risk.append(adhd_score)

    adhd_df['ADHD 위험군'] = adhd_risk

    # 모든 열을 정수형으로 변환
    adhd_df = adhd_df.astype(int)
    return adhd_df

adhd_df = ADHD_score(adhd_df)

##############################################
########## Logistogistic Regression ##########
##############################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

def Logistic_model(adhd_df):
    
    X = adhd_df.drop(columns=['ADHD 위험군'])
    y = adhd_df['ADHD 위험군']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 이항분류
    model = LogisticRegression()

    # 모델 학습
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # 모델 성능 확인
    accuracy = accuracy_score(y_test, y_pred)
    
    
    # 회귀계수 확인
    coefficients = model.coef_[0]
    sorted_indices = sorted(range(len(coefficients)), key=lambda i: abs(coefficients[i]), reverse=True)
    
    for i in sorted_indices:
        feature = X.columns[i]
        coef = coefficients[i]
        result = f"{feature}: {coef}"
        
    
    # ROC curve 시각화
    y_scores = model.predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))
    for i in range(len(y_test.unique())):
        fpr, tpr, _ = roc_curve(y_test, y_scores[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve class {i} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
        
    return model

model = Logistic_model(adhd_df)

# 모델 pickle 객체로 저장
path = r'C:\Users\User\Downloads\Project\ADHD\01_ADHD_ML'

with open(path + '\logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

####################################
########## Normality Test ##########
####################################

# Shapiro-Wilk Test
from scipy import stats

for column in adhd_df.columns:
    if column.startswith("Q"):
        statistic = stats.shapiro(adhd_df[column])[0]
        statistic = "{:.3e}".format(statistic)
        pvalue = stats.shapiro(adhd_df[column])[1]
        formatted_pvalue = "{:.3e}".format(pvalue)
        test_result = f"{column} and ADHD risk group: statistic = {statistic}, ", f"pvalue = {formatted_pvalue}"


# Q-Q plot 시각화
adhd_df = adhd_df.astype('float')

num_cols = adhd_df.columns.str.startswith("Q").sum()
fig, axes = plt.subplots(3, 6, figsize=(12, 8))

index = 0
for column in adhd_df.columns:
    if column.startswith("Q"):
        i, j = divmod(index, 6)  # 서브플롯의 행과 열 인덱스 계산
        stats.probplot(adhd_df[column], plot=axes[i, j])
        axes[i, j].set_title(f'Q-Q plot for {column}')
        index += 1

plt.tight_layout()


###########################################
########## Homoscedasticity Test ##########
###########################################

# Levene Test
from scipy.stats import levene

for column in adhd_df.columns:
    if column.startswith("Q"):
        statistic, p_value = levene(adhd_df[column], adhd_df['ADHD 위험군'], center='median')

        statistic = "{:.3e}".format(statistic)
        p_value = "{:.3e}".format(p_value)

        test_result = f"{column} and ADHD risk group : Statistic = {statistic}, P-value: {p_value}"


###########################################
########## Correlation Analysis ###########
###########################################

# 모든 컬럼 간의 Spearman Rank Correlation
spearman_corr = adhd_df.corr(method='spearman')

# ‘ADHD 위험군’ 컬럼과 다른 모든 컬럼간의 Spearman Rank Correlation
spearman_corrwith = adhd_df.corrwith(adhd_df['ADHD 위험군'], method='spearman')


######################################
########## Factor Analysis ###########
######################################

from factor_analyzer import FactorAnalyzer


X = adhd_df.drop(columns=['ADHD 위험군'])

fa = FactorAnalyzer(n_factors=3, method='minres', rotation='promax')

fa.fit(X)

data = fa.loadings_

sorted_indices = np.argsort(np.abs(data), axis=0)[::-1]


# 요인 분석 시각화 (Bar Plot) : 1번 요인

loadings_first_factor = np.abs(data[:, 0])

variables = [f"Var{i+1}" for i in range(len(loadings_first_factor))]

plt.figure(figsize=(10, 6))
plt.barh(variables, loadings_first_factor, color='skyblue')
plt.xlabel('로딩 값 (Loading Value)')
plt.ylabel('변수 (Variable)')
plt.title('부주의 관련 문항')
plt.grid(axis='x')
plt.show()

# 요인 분석 시각화 (Bar Plot) : 2번 요인

loadings_first_factor = np.abs(data[:, 1])

variables = [f"Var{i+1}" for i in range(len(loadings_first_factor))]

plt.figure(figsize=(10, 6))
plt.barh(variables, loadings_first_factor, color='skyblue')
plt.xlabel('로딩 값 (Loading Value)')
plt.ylabel('변수 (Variable)')
plt.title('과잉행동 관련 문항')
plt.grid(axis='x')
plt.show()

# 요인 분석 시각화 (Bar Plot) : 3번 요인

loadings_first_factor = np.abs(data[:, 2])

variables = [f"Var{i+1}" for i in range(len(loadings_first_factor))]

plt.figure(figsize=(10, 6))
plt.barh(variables, loadings_first_factor, color='skyblue')
plt.xlabel('로딩 값 (Loading Value)')
plt.ylabel('변수 (Variable)')
plt.title('충동성 관련 문항')
plt.grid(axis='x')
plt.show()


###################################
########## Decision Tree ##########
###################################
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)

X = adhd_df.drop(columns=['ADHD 위험군'])
y = adhd_df['ADHD 위험군']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)


# 의사결정나무 시각화
feature_names = X.columns

plt.figure(figsize=(10, 10))
plot_tree(clf, filled=True, feature_names=feature_names)
plt.show()


########################################
########## Class Distribution ##########
########################################
from matplotlib_venn import venn3

symptoms_dic = {"attention_defict_symptoms" : [adhd_df.columns[0], adhd_df.columns[1], adhd_df.columns[2], adhd_df.columns[3], adhd_df.columns[6], adhd_df.columns[7],
                                                   adhd_df.columns[8], adhd_df.columns[9], adhd_df.columns[10]], 
                "hyperactivity_symptoms" : [adhd_df.columns[4], adhd_df.columns[5], adhd_df.columns[11], 
                                                adhd_df.columns[12], adhd_df.columns[13]],
                "impulsivity_symptoms" : [adhd_df.columns[14], adhd_df.columns[15], 
                                              adhd_df.columns[16], adhd_df.columns[17]]}

weights = ["Q1", "Q2", "Q3", "Q9", "Q12", "Q16", "Q18"]

count_df = pd.DataFrame(0, index=adhd_df.index, columns=["attention", "hyperactivity", "impulsivity"])
    
for key, value in symptoms_dic.items():
    category = key.split('_')[0]
    for col in value:
        if col in weights:
            count_df[category] += adhd_df[col].isin([2, 3, 4])
        else:
            count_df[category] += adhd_df[col].isin([3, 4])

venn_df = pd.DataFrame(0, index=adhd_df.index, columns=["attention", "hyperactivity", "impulsivity"])

venn_df['attention'] = (count_df['attention'] >= 3).astype(int)
venn_df['hyperactivity'] = (count_df['hyperactivity'] >= 2).astype(int)
venn_df['impulsivity'] = (count_df['impulsivity'] >= 2).astype(int)


# Venn Diagram 시각화
plt.figure(figsize=(12, 12))

venn_diagram  = venn3(subsets=(len(venn_df[(venn_df['attention'] == 1) & (venn_df['hyperactivity'] == 0) & (venn_df['impulsivity'] == 0)]),
               len(venn_df[(venn_df['attention'] == 0) & (venn_df['hyperactivity'] == 1) & (venn_df['impulsivity'] == 0)]),
               len(venn_df[(venn_df['attention'] == 1) & (venn_df['hyperactivity'] == 1) & (venn_df['impulsivity'] == 0)]),
               len(venn_df[(venn_df['attention'] == 0) & (venn_df['hyperactivity'] == 0) & (venn_df['impulsivity'] == 1)]),
               len(venn_df[(venn_df['attention'] == 1) & (venn_df['hyperactivity'] == 0) & (venn_df['impulsivity'] == 1)]),
               len(venn_df[(venn_df['attention'] == 0) & (venn_df['hyperactivity'] == 1) & (venn_df['impulsivity'] == 1)]),
               len(venn_df[(venn_df['attention'] == 1) & (venn_df['hyperactivity'] == 1) & (venn_df['impulsivity'] == 1)])),
      set_labels=('Attention', 'Hyperactivity', 'Impulsivity'))

for label in venn_diagram.set_labels:
    label.set_fontsize(20) 

for text in venn_diagram.subset_labels:
    text.set_fontsize(16)

plt.title("Venn Diagram of ADHD Symptoms", fontsize=30)
plt.show()

