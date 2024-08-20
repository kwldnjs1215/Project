import pandas as pd
import requests

# Overpass API를 사용하여 태그가 있는 모든 요소를 가져오는 함수
def get_all_tags_osm(latitude, longitude, radius=2000):  # 2km 반경 설정
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{latitude},{longitude})["amenity"];
      way(around:{radius},{latitude},{longitude})["amenity"];
      relation(around:{radius},{latitude},{longitude})["amenity"];
      node(around:{radius},{latitude},{longitude})["leisure"="park"];
      way(around:{radius},{latitude},{longitude})["leisure"="park"];
      relation(around:{radius},{latitude},{longitude})["leisure"="park"];
    );
    out tags;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data['elements']

# CSV 파일 경로 설정
file_name = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\museum_data\museum_list_data_top10.csv'

# CSV 파일 로드 (인코딩 설정)
museum_data = pd.read_csv(file_name, encoding='utf-8')

# 모든 amenity 태그 개수를 저장할 딕셔너리 생성
all_amenity_counts = {index: {} for index in museum_data.index}

# 상위 10개의 박물관에 대해 모든 태그를 검색
for index, row in museum_data.iterrows():
    latitude = row['FCLTY_LA']  # 위도
    longitude = row['FCLTY_LO']  # 경도
    nearby_places = get_all_tags_osm(latitude, longitude)

    # 각 amenity 하위 카테고리의 빈도를 계산
    amenity_counts = all_amenity_counts[index]
    for place in nearby_places:
        if 'tags' in place and 'amenity' in place['tags']:
            amenity = place['tags']['amenity']
            if amenity in amenity_counts:
                amenity_counts[amenity] += 1
            else:
                amenity_counts[amenity] = 1
        elif 'tags' in place and 'leisure' in place['tags'] and place['tags']['leisure'] == 'park':
            # 'leisure' 카테고리에서 'park' 추가
            amenity = 'park'
            if amenity in amenity_counts:
                amenity_counts[amenity] += 1
            else:
                amenity_counts[amenity] = 1

# 모든 고유한 amenity 하위 카테고리 가져오기
all_amenities = set(amenity for amenity_counts in all_amenity_counts.values() for amenity in amenity_counts)

# 모든 amenity 하위 카테고리에 대한 열 추가 및 초기 값을 0으로 설정
for amenity in all_amenities:
    museum_data[f'amenity.{amenity}'] = 0

# 계산된 빈도를 데이터프레임에 채우기
for index, amenity_counts in all_amenity_counts.items():
    for amenity, count in amenity_counts.items():
        museum_data.at[index, f'amenity.{amenity}'] = count

# 결과를 CSV 파일로 저장 (euc-kr 인코딩 사용)
output_file_name = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\overpass_api_map\top10\amenity_counts_top10.csv'
museum_data.to_csv(output_file_name, index=False, encoding='euc-kr')

# 각 amenity 열의 합계를 계산
amenity_sums = museum_data.filter(regex=r'^amenity\.', axis=1).sum()
