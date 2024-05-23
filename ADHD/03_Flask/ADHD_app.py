##############################################
########### ADHD Assessement Model ###########  
##############################################

import pandas as pd
import pickle

path = r'C:\Users\User\Downloads\Project\ADHD\01_ADHD_ML'

with open(path + '\logistic_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

def modeling(scores, loaded_model):
    
    z = scores
    z = pd.DataFrame(z).T
    
    r = loaded_model.predict(z)
    
    return z, r
    

#########################################################
######### Function to calculate ADHD risk score #########
#########################################################

def calculate_adhd_score(scores):
    cnt = 0
    for k in range(18):
        if k in [0, 1, 2, 8, 11, 15, 17]:
            if scores[k] in [2, 3, 4]:
                cnt += 1
        else:
            if scores[k] in [3, 4]:
                cnt += 1
    
    if cnt >= 4:
        return 1
    else:
        return 0

############################
########## Flask ###########
############################
from flask import Flask, render_template, request
import oracledb

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/assessment")
def assessment():
    return render_template('assessment.html')

@app.route("/results", methods=["POST"])
def results():    
    # 폼에서 q1부터 q18까지의 응답을 정수로 변환하여 리스트로 저장
    scores = [int(request.form[f"q{i}"]) for i in range(1, 19)]
    
    # OracleDB 반영
    conn = oracledb.connect(dsn='127.0.0.1:1521/XE',
                            user='c##scott',
                            password='tiger')
        
    cursor = conn.cursor()

    query = f"""insert into adhd_table values ({scores[0]}, {scores[1]}, {scores[2]}, {scores[3]}, 
            {scores[4]}, {scores[5]}, {scores[6]}, {scores[7]}, {scores[8]}, {scores[9]}, {scores[10]}, {scores[11]},
            {scores[12]}, {scores[13]}, {scores[14]}, {scores[15]}, {scores[16]}, {scores[17]})"""
        
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    
    # 새로운 사용자 ADHD 판단 예측
    z,r = modeling(scores, loaded_model)
    
    # 증상 분류 (부주의, 과잉행동, 충동성)
    attention_defict_symptoms = [
        z.columns[0], z.columns[1], z.columns[2], z.columns[3], z.columns[6], z.columns[7],
        z.columns[8], z.columns[9], z.columns[10]
    ]
    hyperactivity_symptoms = [
        z.columns[4], z.columns[5], z.columns[11], z.columns[12], z.columns[13]
    ]
    impulsivity_symptoms = [
        z.columns[14], z.columns[15], z.columns[16], z.columns[17]
    ]

    # 환자별 특정 증상에 해당하는 데이터 개수 계산 함수
    def count_data_with_condition_per_patient(z, group_columns):
        counts_per_patient = []
        for i in range(len(z)):
            cnt = 0
            for col in group_columns:  
                if col in [0, 1, 2, 8, 11, 15, 17]:
                    if z.loc[i][col] in [2, 3, 4]:
                        cnt += 1
                else:
                    if z.loc[i][col] in [3, 4]:
                        cnt += 1
            counts_per_patient.append(cnt)
        return counts_per_patient

    attention_deficit_count_per_patient = count_data_with_condition_per_patient(z, attention_defict_symptoms)
    hyperactivity_count_per_patient = count_data_with_condition_per_patient(z, hyperactivity_symptoms)
    impulsivity_count_per_patient = count_data_with_condition_per_patient(z, impulsivity_symptoms)

    primary_1_class = ""; primary_2_class = ""; primary_3_class = "";
    final_attention = "";   final_hyperactivity= "";   final_impulsivity="";

    
    # 임계점 : ADHD 위험군이지만 어떠한 증상도 나타나지 않은 환자들이 생기지 않도록 최대값으로 설정
    
    class_ = []
    # 주의력 결핍 증상이 의심될 경우 추천 사항 설정    
    if attention_deficit_count_per_patient[0] >=3:
        class1 = "부주의함"
        class_.append(class1)
        #primary_1_class = "현재의 정황으로 미루어 보아, 약간의 부주의함이 엿보입니다."
        final_attention = ["체크리스트 작성법 공부 및 체크리스트 작성", "집중모드 기능 사용", 
                           "자신의 주요 작업 공간 사진 찍은 후 항상 유지할 수 있도록 도움 주기",
                           "루틴 짜주기 40분 집중 - 20분 휴식"]
        
    # 과잉행동 증상이 의심될 경우 추천 사항 설정
    if hyperactivity_count_per_patient[0] >=2:
        class2 = "과잉행동"
        class_.append(class2)
        #primary_2_class = "현재 정황상 과잉행동 증상이 엿보입니다."
        final_hyperactivity = ["음악감상", "명상 및 요가", "ASMR", "운동"]
        
    # 충동성 증상이 의심될 경우 추천 사항 설정
    if impulsivity_count_per_patient[0] >=2:
        class3 = "충동성"
        class_.append(class3)
        #primary_3_class = "현재 정황상 충동성 증상이 엿보입니다."
        final_impulsivity = ["감정일기 쓰기", "스피칭 연습", "심호흡 연습", "분노 조절하는 법 연습"]
    
    
    class_ = ', '.join(class_)
    
    # ADHD 위험 점수 계산
    adhd_risk = calculate_adhd_score(scores)
    
    # 결과를 HTML 템플릿에 렌더링하여 반환
    return render_template('results.html', adhd_risk=adhd_risk, 
                           final_attention=final_attention,
                           final_hyperactivity=final_hyperactivity,
                           final_impulsivity=final_impulsivity,
                           primary_1_class=primary_1_class,
                           primary_2_class=primary_2_class,
                           primary_3_class=primary_3_class,
                           class_=class_)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
