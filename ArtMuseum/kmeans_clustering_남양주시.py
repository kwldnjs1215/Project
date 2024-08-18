# 기존 설정
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)

# 데이터 경로 설정
data_path = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\data'


# 남양주 공원 데이터 불러오기
park = pd.read_csv(data_path + '\국내_문화체육관광_분야_국립도군립_및_도시내_공원_데이터_2023.csv')
park_region = park[(park['CTPRVN_NM'] == '경기도') & (park['SIGNGU_NM'].str.contains('남양주시'))]
park_info = park_region[['POI_NM', 'LC_LA', 'LC_LO']]
park_info.insert(0, 'Type', '공원')
park_info

# 남양주 키즈카페 데이터 불러오기
kids = pd.read_csv(data_path + '\전국_어린이_대상_문화공간(키즈카페)_위치(2022).csv')
kids_region = kids[(kids['CTPRVN_KLANG_NM'] == '경기도') & (kids['SIGNGU_KLANG_NM'].str.contains('남양주시'))]
kids_info = kids_region[['FCLTY_NM', 'FCLTY_LA', 'FCLTY_LO']]
kids_info.insert(0, 'Type', '키즈카페')
kids_info

# 남양주 초등학교 데이터 불러오기
school = pd.read_csv(data_path + '\전국초중등학교위치표준데이터.csv', encoding='euc-kr')
school = school[school['학교급구분'] == '초등학교']
school_region = school[school['소재지지번주소'].str.contains('남양주시')]
school_info = school_region[['학교명', '위도', '경도']]
school_info.insert(0, 'Type', '초등학교')
school_info

# 남양주 주차장 데이터 불러오기
parking = pd.read_csv(data_path + '\전국주차장정보표준데이터.csv', encoding='euc-kr')
parking_region = parking[(parking['소재지지번주소'].str.contains('남양주시')) | (parking['소재지도로명주소'].str.contains('남양주시'))]
parking_info = parking_region[['주차장명', '위도', '경도']]
parking_info.insert(0, 'Type', '주차장')
parking_info = parking_info.dropna()
parking_info

# 남양주 어린이집 데이터 불러오기
kid=pd.read_csv(data_path + '\어린이집기본정보조회(정기)-기준일(20240531).csv', encoding='UTF-8')
kid_active=kid[(kid['운영현황']=='정상') | (kid['운영현황']=='재게')]
kid_info=kid_active[['어린이집명', '위도','경도']]
kid_info.insert(0,'Type','어린이집')
kid_info

# 남양주 유치원 불러오기
kid1=pd.read_csv(data_path + '\유치원현황.csv', encoding='euc-kr')
kid1_region = kid1[(kid1['정제지번주소'].str.contains('남양주시')) | (kid1['정제도로명주소'].str.contains('남양주시'))]
kid1_info=kid1_region[['유치원명','정제WGS84위도','정제WGS84경도']]
kid1_info.insert(0,'Type','유치원')
kid1_info=kid1_info.dropna()
kid1_info.isnull().sum()

# 남양주 근처 문화 시설
playground=pd.read_csv(data_path + '\전국_아이랑_문화시설_위치(2023).csv', encoding='UTF-8')
playground_region=playground[(playground['SIGNGU_NM']=='남양주시') & (playground['MLSFC_NM']=='구/군급/재단도서관')]
playground_info=playground_region[[ 'FCLTY_NM', 'FCLTY_LA', 'FCLTY_LO']]
playground_info.insert(0,'Type','문화시설')


# 남양주시 좌표 데이터 병합
point_df = pd.concat([park_info.rename(columns={'POI_NM': 'NM', 'LC_LA': 'LA', 'LC_LO': 'LO'}),
                     kids_info.rename(columns={'FCLTY_NM': 'NM', 'FCLTY_LA': 'LA', 'FCLTY_LO': 'LO'}),
                     school_info.rename(columns={'학교명': 'NM', '위도': 'LA', '경도': 'LO'}),
                     parking_info.rename(columns={'주차장명': 'NM', '위도': 'LA', '경도': 'LO'}),
                     kid_info.rename(columns={'어린이집명': 'NM', '위도': 'LA', '경도': 'LO'}),
                     kid1_info.rename(columns={'유치원명': 'NM', '정제WGS84위도': 'LA', '정제WGS84경도': 'LO'}),
                     playground_info.rename(columns={'FCLTY_NM': 'NM', 'FCLTY_LA': 'LA', 'FCLTY_LO': 'LO'})],
                     ignore_index=True)


exclude_list = ['다산도담어린이집', '시립행복한푸르지오어린이집', '시립다산자이폴라리스어린이집', '시립남양주하나어린이집']
point_df = point_df[~point_df['NM'].isin(exclude_list)]


# K-means 클러스터링을 위한 좌표 데이터 준비
X = point_df[['LA', 'LO']]

# 데이터 표준화 (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 엘보우 플랏
sse = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.xticks(range(1, 15))
plt.grid(True)
plt.show()

# 최적의 클러스터 개수 설정
optimal_k = 9

# K-means 클러스터링 수행
kmeans = KMeans(n_clusters=optimal_k, random_state=123)
kmeans.fit(X_scaled)
clusters = kmeans.labels_
point_df['Cluster'] = clusters

# 실루엣 계수 계산
silhouette_avg = silhouette_score(X_scaled, clusters)
silhouette_values = silhouette_samples(X_scaled, clusters)
print(f"실루엣 계수: {silhouette_avg}")

'''
실루엣 계수: 0.5731420040889402
'''

# 군집별 실루엣 계수 평균 값 계산 및 출력
cluster_silhouette_avg_values = []

for i in range(optimal_k):
    cluster_silhouette_avg = silhouette_values[clusters == i].mean()
    cluster_silhouette_avg_values.append((i + 1, cluster_silhouette_avg))

# 내림차순으로 정렬
cluster_silhouette_avg_values.sort(key=lambda x: x[1], reverse=True)

print("군집별 실루엣 계수 평균 값 (내림차순 정렬):")
for cluster, silhouette_avg in cluster_silhouette_avg_values:
    print(f"Cluster {cluster}: {silhouette_avg:.4f}")
    
'''
Cluster 7: 0.9248
Cluster 6: 0.7843
Cluster 5: 0.6354
Cluster 1: 0.5615
Cluster 3: 0.5327
Cluster 4: 0.5239
Cluster 2: 0.5110
Cluster 8: 0.4996
Cluster 9: 0.4475
'''
    
    
# 군집별 실루엣 계수 시각화
fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10
for i in range(optimal_k):
    ith_cluster_silhouette_values = silhouette_values[clusters == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))  # 클러스터 번호를 1부터 시작
    y_lower = y_upper + 10

ax.set_title("Silhouette plot for the various clusters.")
ax.set_xlabel("Silhouette coefficient values")
ax.set_ylabel("Cluster label")

ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.set_yticks([])  # Clear the yaxis labels / ticks
ax.set_xticks(np.arange(-0.1, 1.1, 0.1))

plt.show()


# 클러스터 중심점 출력 (원래 스케일로 변환하여 출력)
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
print("Cluster centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i+1}: Latitude = {centroid[0]}, Longitude = {centroid[1]}")




############################################################################
# 시각화
plt.figure(figsize=(10, 8))

# 클러스터별로 색상 맵 설정 (6가지 색상)
cluster_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown','pink','cyan','yellow']

# 클러스터링된 클러스터들을 다른 색상으로 표시
for cluster in range(optimal_k):
    cluster_data = X[clusters == cluster]
    plt.scatter(cluster_data['LO'], cluster_data['LA'], color=cluster_colors[cluster], label=f'Cluster {cluster+1}')

# 클러스터 중심점 표시 (원래 스케일로 변환하여 표시)
plt.scatter(centroids[:, 1], centroids[:, 0], marker='x', c='black', s=100, label='Centroids')



plt.title('K-means 클러스터링을 통한 남양주 최적 입지 선정')
plt.xlabel('경도 (LO)')
plt.ylabel('위도 (LA)')

# 범례 설정
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', markerscale=0.5)

plt.show()

############################################################################

# 추가할 특정 좌표들
specific_points = [
    (37.6161048, 127.1524814), # 경기 남양주시 다산순환로 50 (모카가든)
    (37.5533599, 127.2366799), # 경기 남양주시 화도읍 경춘로2110번길 8 (모란미술관)
    (37.6326724, 127.351459), # 경기 남양주시 화도읍 북한강로 1344 (서호미술관)
    (37.579984, 127.3311847), # 경기 남양주시 조안면 북한강로626번길 82-20 (이규택석조미술관)
    (37.5533599, 127.2366799) # 경기 남양주시 와부읍 경강로926번길 30 한강뮤지엄 (한강뮤지엄)
]


# 각 특정 좌표들에 대해 가장 가까운 클러스터 중심점 찾기
for point in specific_points:
    min_distance = np.inf
    nearest_cluster = None
    
    for i, centroid in enumerate(centroids):
        # 맨해튼 거리 계산
        distance = np.abs(point[0] - centroid[0]) + np.abs(point[1] - centroid[1])
        
        if distance < min_distance:
            min_distance = distance
            nearest_cluster = i + 1  # 클러스터 번호는 1부터 시작하므로 인덱스에 +1을 해줍니다
    
    # 거리를 km 단위로 변환
    min_distance_km = min_distance * 111  # 약 111 km per degree
    print(f"Point {point}: Nearest Cluster = {nearest_cluster}, Distance = {min_distance_km:.3f} km")

'''
Point (37.6161048, 127.1524814): Nearest Cluster = 1, Distance = 1.300 km
Point (37.5533599, 127.2366799): Nearest Cluster = 6, Distance = 5.393 km
Point (37.6326724, 127.351459): Nearest Cluster = 3, Distance = 7.422 km
Point (37.579984, 127.3311847): Nearest Cluster = 8, Distance = 8.059 km
Point (37.5533599, 127.2366799): Nearest Cluster = 6, Distance = 5.393 km
'''
    

############################################################################
# 시각화: 실제 좌표와 클러스터 중심점 비교
plt.figure(figsize=(10, 8))

# 클러스터별로 색상 맵 설정 (6가지 색상)
cluster_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown','pink','cyan','yellow']

# 클러스터링된 클러스터들을 다른 색상으로 표시
for cluster in range(optimal_k):
    cluster_data = X[clusters == cluster]
    plt.scatter(cluster_data['LO'], cluster_data['LA'], color=cluster_colors[cluster], label=f'Cluster {cluster+1}')

# 클러스터 중심점 표시 (원래 스케일로 변환하여 표시)
plt.scatter(centroids[:, 1], centroids[:, 0], marker='x', c='black', s=100, label='Centroids')

for i, point in enumerate(specific_points):
    plt.scatter(point[1], point[0], color='red', marker='*', s=200, label='Specific Point' if i == 0 else None)
# 각 특정 좌표와 가장 가까운 클러스터 중심점 연결선 그리기
for point in specific_points:
    min_distance = np.inf
    nearest_cluster = None
    
    for i, centroid in enumerate(centroids):
        # 맨해튼 거리 계산
        distance = np.abs(point[0] - centroid[0]) + np.abs(point[1] - centroid[1])
        
        if distance < min_distance:
            min_distance = distance
            nearest_cluster = i + 1  # 클러스터 번호는 1부터 시작하므로 인덱스에 +1을 해줍니다
    
    # 가장 가까운 클러스터 중심점의 좌표
    nearest_centroid = centroids[nearest_cluster - 1]
    
    # 연결선 그리기
    plt.plot([point[1], nearest_centroid[1]], [point[0], nearest_centroid[0]], linestyle='--', color='gray')
plt.title('남양주시 최적 입지 선정 좌표와 실제 미술관 비교')
plt.xlabel('경도 (LO)')
plt.ylabel('위도 (LA)')

# 범례 설정
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', markerscale=0.5)

plt.show()

############################################################################
# 필요한 열만 선택
data = point_df[['LA', 'LO', 'Type']]

# K-means를 위해 위도 경도를 기준으로 군집화를 진행
kmeans = KMeans(n_clusters=9, random_state=123)
data['Cluster'] = kmeans.fit_predict(data[['LA', 'LO']])

# 데이터를 시각화하기 위해 군집별로 색을 지정
cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
data['Color'] = data['Cluster'].map(cluster_colors)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='LO', y='LA', hue='Type', palette='tab10', data=data, s=100, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], marker='x', s=200, c='black', label='Centroids')
plt.title('남양주 K-means Clustering 주변 상권')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()
