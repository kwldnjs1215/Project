import pandas as pd
import requests

# Overpass API를 사용하여 태그가 있는 모든 요소를 가져오는 함수
def get_all_tags_osm(latitude, longitude, radius=1000):  # 1km 반경
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node(around:{radius},{latitude},{longitude})["amenity"];
      way(around:{radius},{latitude},{longitude})["amenity"];
      relation(around:{radius},{latitude},{longitude})["amenity"];
    );
    out tags;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data['elements']

file_name = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\museum_data\museum_list_data.csv'

# 감지된 인코딩으로 CSV 파일을 로드
museum_data = pd.read_csv(file_name, encoding='utf-8')

# 모든 amenity 태그의 개수를 저장할 딕셔너리
all_amenity_counts = {index: {} for index in museum_data.index}

# 각 박물관에 대한 모든 태그 검색
for index, row in museum_data.iterrows():
    latitude = row['FCLTY_LA']
    longitude = row['FCLTY_LO']
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

# 모든 고유한 amenity 하위 카테고리를 가져옴
all_amenities = set(amenity for amenity_counts in all_amenity_counts.values() for amenity in amenity_counts)

# 모든 amenity 하위 카테고리에 대한 열을 추가하고 초기값을 0으로 설정
for amenity in all_amenities:
    museum_data[f'amenity.{amenity}'] = 0

# 개수 채우기
for index, amenity_counts in all_amenity_counts.items():
    for amenity, count in amenity_counts.items():
        museum_data.at[index, f'amenity.{amenity}'] = count

# utf-8 인코딩으로 CSV 파일 저장
output_file_name = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\overpass_api_map\all\national_art_museum_with_amenity_counts.csv'
museum_data.to_csv(output_file_name, index=False, encoding='euc-kr')
