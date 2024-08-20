import pandas as pd
import requests

# Function to get all elements with tags using Overpass API
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

# Load the CSV file with the detected encoding
museum_data = pd.read_csv(file_name, encoding='utf-8')

# Dictionary to hold all amenity tag counts
all_amenity_counts = {index: {} for index in museum_data.index}

# Search for all tags for each museum
for index, row in museum_data.iterrows():
    latitude = row['FCLTY_LA']
    longitude = row['FCLTY_LO']
    nearby_places = get_all_tags_osm(latitude, longitude)

    # Count the frequency of each amenity subcategory
    amenity_counts = all_amenity_counts[index]
    for place in nearby_places:
        if 'tags' in place and 'amenity' in place['tags']:
            amenity = place['tags']['amenity']
            if amenity in amenity_counts:
                amenity_counts[amenity] += 1
            else:
                amenity_counts[amenity] = 1

# Get all unique amenity subcategories
all_amenities = set(amenity for amenity_counts in all_amenity_counts.values() for amenity in amenity_counts)

# Add columns for all amenity subcategories and set initial values to 0
for amenity in all_amenities:
    museum_data[f'amenity.{amenity}'] = 0

# Fill in the counts
for index, amenity_counts in all_amenity_counts.items():
    for amenity, count in amenity_counts.items():
        museum_data.at[index, f'amenity.{amenity}'] = count

# Save csv with utf-8 encoding
output_file_name = r'C:\Users\User\Desktop\02_공모전\6월_2024_문화_디지털혁신_및_문화데이터_활용_공모전\overpass_api_map\all\national_art_museum_with_amenity_counts.csv'
museum_data.to_csv(output_file_name, index=False, encoding='euc-kr')
