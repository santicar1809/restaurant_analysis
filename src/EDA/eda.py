# 
# # Part 1
# 
# Subway is looking to raise growth equity capital to further expand in Brazil. Our fund is looking to conduct outside-in diligence on the company to assess its current state and further opportunities in order to decide whether to participate in the process.
# The fund would like you to evaluate the company and provide data-driven insights and recommendations. The base analysis should include the following:
# 
#     ● Current % coverage of Brazilian population & additional locations needed to cover additional 2% of the population
# 
#     ● Overlap with Mcdonald’s locations (i.e., % of locations with overlap within a 1-, 5-, 10-km radius)
# 
# In addition to the location analysis, you may choose to use additional data points or analyses to further your recommendations. Here are a few ideas of signals that we often look at, please note these are simply ideas of optional additions to your final analysis.
# 
#     ● Social media sentiment
#     ● Employee sentiment
#     ● Pricing / cost signals
#     ● International opportunities
#     ● Any additional factors you think that we should consider
# 
# Here are some sources to get you started:
# 
#     ● Databases provided in the “Data” folder along with this file. 
#     ● https://www.ibge.gov.br

# 
# ## Import libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import chardet
import re
import plotly.graph_objects as go
import folium
from folium import plugins
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import os
import ast
from transformers import pipeline
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup


# 
# Load Datasets
# 


subw_data = pd.read_csv('./files/datasets/input/Subway_Locations.csv')
mc_data = pd.read_csv('./files/datasets/input/Mcdonalds_Locations.csv')

# 
# City Population
# (https://www.ibge.gov.br/estatisticas/downloads-estatisticas.html?caminho=Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/Agregados_por_Municipio_csv/)


# Detect encoding
with open("pob_brasil_city.csv", "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")


pob_city = pd.read_csv('./files/datasets/input/pob_brasil_city.csv',sep=";",encoding="ISO-8859-1")

# 
# ## Preprocessing data


# Function to format data columns
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    s2 = s1.replace('-','')
    s3 = s2.replace(',','')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s3).lower()

def columns_transformer(data):
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data



# Mapping of accented characters to unaccented counterparts
accented_to_unaccented = {
    'á': 'a', 'ã': 'a', 'à': 'a', 'â': 'a',
    'é': 'e', 'ê': 'e',
    'í': 'i',
    'ó': 'o', 'õ': 'o', 'ô': 'o',
    'ú': 'u',
    'ç': 'c',
    'Á': 'A', 'Ã': 'A', 'À': 'A', 'Â': 'A',
    'É': 'E', 'Ê': 'E',
    'Í': 'I',
    'Ó': 'O', 'Õ': 'O', 'Ô': 'O',
    'Ú': 'U',
    'Ç': 'C'
}

# Function to replace accented characters
def replace_accented(text):
    s1 = re.sub(r'[áãàâéêíóõôúçÁÃÀÂÉÊÍÓÕÔÚÇ]', 
                  lambda match: accented_to_unaccented[match.group(0)], 
                  text)
    s2 = s1.replace(' ','-')
    s3 = s2.replace("'","")
    s4 = s3.replace("-d-","-d")
    return s4.lower()


# Column transform to snake_case
subw_data = columns_transformer(subw_data)
pob_city = columns_transformer(pob_city)
mc_data = columns_transformer(mc_data)

# 
# ### Subway Locations


subw_data.info()


subw_data.drop(['unnamed:_0'],axis=1,inplace=True)


subw_data.duplicated().sum()


subw_data.isna().sum()


subw_data['city']=subw_data['city'].apply(replace_accented)


# Change tel column to boolean
subw_data['tel'].fillna(False)
def tel(data):
    if data!=False:
        return True
subw_data['tel']=subw_data['tel'].apply(tel)


# We change the city names that does not match
subw_data['city']=subw_data['city'].replace('cabo-de-sto-agostinho','cabo-de-santo-agostinho')
subw_data['city']=subw_data['city'].replace('igarussu','igarassu')
subw_data['city']=subw_data['city'].replace('monte-camelo','monte-carmelo')
subw_data['city']=subw_data['city'].replace('paraupebas','parauapebas')
subw_data['city']=subw_data['city'].replace('embu','embu-das-artes')


subw_data.head(5)

# 
# ### McDonalds Locations


mc_data.info()


mc_data.isna().sum()


mc_data.duplicated().sum()


mc_data['coordinates'] = mc_data['coordinates'].apply(ast.literal_eval)
mc_data['longitude'] = mc_data['coordinates'].apply(lambda x: x['longitude'])
mc_data['latitude'] = mc_data['coordinates'].apply(lambda x: x['latitude'])
mc_data.drop(['coordinates'],axis=1,inplace=True)


mc_data = mc_data.reset_index(drop=True).dropna()


mc_data.drop(['unnamed:_0'],axis=1,inplace=True)


mc_data['city']=mc_data['city'].apply(replace_accented)

# 
# ### City Population


info_rep=pob_city.info()


# We keep columns with the name of the city and the population number. Besides that, we calculate the percentage
pob_city_acot = pob_city[['nm_mun','quantidade_de_moradores']]
pob_city_acot['percentage_pop'] = 100*pob_city_acot['quantidade_de_moradores']/sum(pob_city_acot['quantidade_de_moradores'])


describe_rep=pob_city_acot.describe()


# We change the accents and the format of the cities
pob_city_acot['nm_mun']=pob_city_acot['nm_mun'].apply(replace_accented)

# 
# ## EDA

# 
# ### Percentage of the population covered per city


#Groupby the cities to plot data
subw_data_city = subw_data.groupby(['city'])['code'].count().reset_index()
subw_data_city_pop = subw_data_city.merge(pob_city_acot,how='left',left_on='city',right_on='nm_mun')
total=pd.DataFrame({'city':'total','percentage_pop':[subw_data_city_pop['percentage_pop'].sum()]})
subw_data_city_pop=pd.concat([subw_data_city_pop,total])
subw_data_city_top=subw_data_city_pop.sort_values(by='percentage_pop',ascending=False).head(10)


subw_data_city_top=subw_data_city_top.drop(['code'],axis=1)


# Barplot of the population percentage per city and the total
fig,ax=plt.subplots()
highlight_city='total'
colors = ['red' if city == highlight_city else 'blue' for city in subw_data_city_top['city']]
bars=ax.bar(x=subw_data_city_top['city'],height=subw_data_city_top['percentage_pop'],color=colors)
ax.set_xticklabels(subw_data_city_top['city'], rotation=45, ha='right')
ax.bar_label(bars, fmt='%.2f%%')

fig.savefig('./files/modeling_output/figures/pop_city.png')

# 
# ### Percentage of the population to be covered


# Function to calculate the accumulated percentage
def acum_per(data):
    result=[]
    acum=0
    for i in data:
        acum += i
        result.append(acum)
        return result


# Groupby the cities that subway does not have restaurant 
subw_data_not_city = subw_data_city.merge(pob_city_acot,how='right',left_on='city',right_on='nm_mun')
subw_data_not_city=subw_data_not_city[subw_data_not_city['city'].isna()]
subw_data_not_city.sort_values(by='percentage_pop',ascending=False,inplace=True)
subw_data_not_city['acum_per']=subw_data_not_city['percentage_pop'].cumsum()
total=pd.DataFrame({'nm_mun':'total','percentage_pop':[subw_data_not_city['percentage_pop'].sum()]})
subw_data_not_city_pop=pd.concat([subw_data_not_city,total])
subw_data_not_city_pop_top=subw_data_not_city_pop.sort_values(by='percentage_pop',ascending=False).head(10)


subw_data_not_city_pop_top=subw_data_not_city_pop_top.drop(['city','code'],axis=1)


# Plot of the population percentage where subway does not have restaurant
fig1,ax1=plt.subplots()
highlight_city='total'
colors = ['red' if city == highlight_city else 'blue' for city in subw_data_not_city_pop_top['nm_mun']]
bars=ax1.bar(x=subw_data_not_city_pop_top['nm_mun'],height=subw_data_not_city_pop_top['percentage_pop'],color=colors)
ax1.set_xticklabels(subw_data_not_city_pop_top['nm_mun'], rotation=45, ha='right')
ax1.bar_label(bars, fmt='%.2f%%')
ax1.set_title('Population top 10 cities without restaurant')
fig1.savefig('./files/modeling_output/figures/pop_percentage.png')

# 
# ### Percentage Accumulated


# Percentage accummulated of the top 10 cities where Subway does not have restaurant
subw_data_not_city_2 = subw_data_not_city[subw_data_not_city['acum_per']<2.008]


fig2,ax2=plt.subplots()
bars=ax2.bar(x=subw_data_not_city_2['nm_mun'],height=subw_data_not_city_2['acum_per'])
ax2.set_xticklabels(subw_data_not_city_2['nm_mun'], rotation=45, ha='right')
ax2.bar_label(bars, fmt='%.2f%%')

fig2.savefig('./files/modeling_output/figures/percentage_acum.png')

# Based on the analysis made, currently subway is covering **61.31%** of Brasil Population and to get to an additional 2% of the population, Subway has to open new branches in the cities shown in the bar chart. This cities were sorted from the highest population percentage to the least population percentage, so when the accumulated percentage reaches 2%, the list of cities are taken to open new restaurants.  
# The list of cities is the following:  
# campos-dos-goytacazes, caucaia, santarem, mossoro, maracanau, arapiraca, mage, viamao, marica, castanhal, alvorada, ferraz-de-vasconcelos, mesquita, sao-caetano-do-sul, linhares, abaetetuba, caxias, camaragibe.  
# In total it would be **18** within the cities.

# 
# Map of the branches


# Function to extract coordenates
def extract_coordinates(url):
    match = re.search(r"destination=(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if match:
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        return latitude, longitude
    return None, None

subw_data[['latitude', 'longitude']] = subw_data['maps__link'].apply(lambda x: pd.Series(extract_coordinates(x)))


subw_data.to_csv('./files/datasets/output/subw_data.csv',index=False)


# The dataset of the coordinates is merged with the population percentage
subw_data_geo = subw_data.merge(pob_city_acot,how='inner',left_on='city',right_on='nm_mun')



subw_data_city_pop['city']='Brasil, '+subw_data_city_pop['city']
subw_data_not_city_2['nm_mun']='Brasil, '+subw_data_not_city_2['nm_mun']


# Function that finds the location based on the city
geolocator = Nominatim(user_agent="city_geocoder",timeout=10)
def get_coordinates(city):
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
# The function is applied to the dataset with subway restaurant
subw_data_city_pop[['latitude', 'longitude']] = subw_data_city_pop['city'].apply(lambda city: pd.Series(get_coordinates(city)))



# The function is applied to the dataset filtered without subway restaurant
subw_data_not_city_2[['latitude', 'longitude']] = subw_data_not_city_2['nm_mun'].apply(lambda city: pd.Series(get_coordinates(city)))


#subw_data_city_pop.to_csv(output_path+'subw_data_city_pop.csv',index=False)


subw_data_city_pop.reset_index(drop=True,inplace=True)


subw_data_city_pop.dropna(inplace=True)


# Initialize map
m = folium.Map(
    location=[-14.2350, -51.9253],  # center around Africa
    zoom_start=4,  # dezoom
    tiles='cartodb positron'  # background style
)

for idx, row in subw_data_geo.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['name']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}', 
        tooltip=row['city'] 
    ).add_to(m)
for idx, row in subw_data_city_pop.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        color='red',
        fill=True,
        radius=row['percentage_pop']*8,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['city']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}<br>Population:{row['percentage_pop']}', 
        tooltip=row['city'] 
    ).add_to(m)
for idx, row in subw_data_not_city_2.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        color='yellow',
        fill=True,
        radius=row['percentage_pop']*100,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['nm_mun']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}<br>Population:{row['percentage_pop']}', 
        tooltip=row['nm_mun']
    ).add_to(m)


legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
        <b>Map</b><br>
        <ul>
            <li><span style="background-color: yellow; width: 20px; height: 20px; display: inline-block;"></span> City without restaurant</li>
            <li><span style="background-color: red; width: 20px; height: 20px; display: inline-block;"></span> City with restaurant</li>
            <li><span style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></span> Restaurant</li>
        </ul>
    </div>
"""


m.get_root().html.add_child(folium.Element(legend_html))

m.save("./files/modeling_output/figures/mapa_bubble.html")

# 
# ### Overlap with McDonalds


# Calculation of the locations overlaped 
subway=subw_data[['name','latitude','longitude']]
mcdonalds=mc_data[['name','latitude','longitude']]

subway_gdf = gpd.GeoDataFrame(subway, 
                               geometry=gpd.points_from_xy(subway.longitude, subway.latitude))

mcdonalds_gdf = gpd.GeoDataFrame(mcdonalds, 
                                  geometry=gpd.points_from_xy(mcdonalds.longitude, mcdonalds.latitude))

def calculate_overlap(subway_gdf, mcdonalds_gdf, radius_km):
    overlaps = 0
    total_subways = len(subway_gdf)
    
    for _, subway in subway_gdf.iterrows():
        subway_point = subway['geometry']
        
        # Check McDonald's locations within the radius
        for _, mcdonald in mcdonalds_gdf.iterrows():
            mcdonald_point = mcdonald['geometry']
            
            # Calculate the distance in km between Subway and McDonald's locations
            distance = geodesic((subway_point.y, subway_point.x), (mcdonald_point.y, mcdonald_point.x)).km
            
            if distance <= radius_km:
                overlaps += 1
                break  # Once overlap is found, no need to check further McDonald's locations
    
    # Calculate the percentage of Subway locations that overlap
    overlap_percentage = (overlaps / total_subways) * 100
    return overlap_percentage

overlap_1km = calculate_overlap(subway_gdf, mcdonalds_gdf, 1)
overlap_5km = calculate_overlap(subway_gdf, mcdonalds_gdf, 5)
overlap_10km = calculate_overlap(subway_gdf, mcdonalds_gdf, 10)

print(f"Percentage overlap with McDonald's within 1 km: {overlap_1km}%")
print(f"Percentage overlap with McDonald's within 5 km: {overlap_5km}%")
print(f"Percentage overlap with McDonald's within 10 km: {overlap_10km}%")

if overlap_1km > 50:
    print("High competition in 1 km zones. Consider avoiding these areas.")
else:
    print("Few Subway locations overlap with McDonald's within 1 km. Opportunity for growth.")

if overlap_5km > 50:
    print("High competition in 5 km zones. Consider targeting less competitive areas.")

if overlap_10km > 50:
    print("High competition in 10 km zones. Focus on underserved locations.")

# Plot the Subway and McDonald's locations
fig3,ax3=plt.subplots(figsize=(10, 6))
ax3.plot(subway_gdf,color='blue', marker='o', label='Subway')
ax3.plot(mcdonalds_gdf,color='red', marker='x', label="McDonald's")
ax3.set_title("Subway vs McDonald's Locations")
ax3.legend()
fig3.tight_layout()
fig3.savefig('./files/modeling_output/figures/mc_loc.png')



# Initialize map
m = folium.Map(
    location=[-14.2350, -51.9253],  # center around Africa
    zoom_start=4,  # dezoom
    tiles='cartodb positron'  # background style
)

for idx, row in subw_data_geo.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['name']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}', 
        tooltip=row['city'] 
    ).add_to(m)
for idx, row in mcdonalds.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        color='red',
        fill=True,
        radius=2,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['name']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}<br>', 
        tooltip=row['name'] 
    ).add_to(m)



legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
        <b>Map</b><br>
        <ul>
            <li><span style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></span> Subway</li>
            <li><span style="background-color: red; width: 20px; height: 20px; display: inline-block;"></span> McDonalds</li>
        </ul>
    </div>
"""

# Añadir la leyenda al mapa
m.get_root().html.add_child(folium.Element(legend_html))

m.save("./files/modeling_output/figures/overlap_map.html")

# 
# - Percentage overlap with McDonald's within 1 km: 41.6%
# - Percentage overlap with McDonald's within 5 km: 72.12307692307692%
# - Percentage overlap with McDonald's within 10 km: 78.0923076923077%
# - Few Subway locations overlap with McDonald's within 1 km. Opportunity for growth.
# - High competition in 5 km zones. Consider targeting less competitive areas.
# - High competition in 10 km zones. Focus on underserved locations.

# 
# ### Social Sentiment

# 
# Sentiment analysis is made making a web scrapping of the links of each restaurant to get the rating.


def rating_sentiment(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for failed requests
    soup = BeautifulSoup(response.text, 'lxml')
    # Extracting all rating values
    ratings = [float(rating.text.strip()) for rating in soup.find_all('strong', class_='rating-value')]

    # Calculate the mean
    mean_rating = np.mean(ratings)

    return mean_rating



subw_data_sentiment=subw_data_geo.sort_values(by='percentage_pop',ascending=False).head(700)


subw_data_sentiment['rating']=subw_data_sentiment['restaurant__link'].apply(rating_sentiment)


subw_data_sentiment_gr=subw_data_sentiment.groupby(['city'],as_index=False)['rating'].mean()


subw_data_sentiment_gr[['latitude', 'longitude']] = subw_data_sentiment_gr['city'].apply(lambda city: pd.Series(get_coordinates(city)))


# Initialize map
m = folium.Map(
    location=[-14.2350, -51.9253],  # center around Africa
    zoom_start=4,  # dezoom
    tiles='cartodb positron'  # background style
)

for idx, row in subw_data_sentiment_gr.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['rating']*2,
        color='blue',
        fill=True,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['city']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}', 
        tooltip=row['city'] 
    ).add_to(m)


m.save("./files/modeling_output/figures/map_sentiment.html")


subw_data_sentiment_gr=subw_data_sentiment_gr[subw_data_sentiment_gr['city']!='cuiaba']
total_subw=pd.DataFrame({'city':'Total','rating':[subw_data_sentiment_gr['rating'].mean()]})
subw_data_sentiment_gr=pd.concat([subw_data_sentiment_gr,total_subw])


with pd.ExcelWriter('./files/modeling_output/reports/output.xlsx', engine='openpyxl') as writer:
    subw_data_sentiment_gr.to_excel(writer,sheet_name='subw_data_sentiment_gr',index=False)
    subw_data_city_top.to_excel(writer,sheet_name='subw_data_city_top',index=False)
    subw_data_not_city_pop_top.to_excel(writer,sheet_name='subw_data_not_city_pop_top',index=False)
    subw_data_geo.to_excel(writer,sheet_name='subw_data_geo',index=False)
    subway_gdf.to_excel(writer,sheet_name='subway_gdf',index=False)
    mcdonalds_gdf.to_excel(writer,sheet_name='mcdonalds_gdf',index=False)
    subw_data_city_pop.to_excel(writer,sheet_name='subw_data_city_pop',index=False)
    subw_data_not_city_2.to_excel(writer,sheet_name='subw_data_not_city_2',index=False)
    


fig5,ax5=plt.subplots(figsize=(10,7))
colors = ['red' if city == 'Total' else 'blue' for city in subw_data_sentiment_gr['city']]
bars=ax5.bar(subw_data_sentiment_gr['city'],subw_data_sentiment_gr['rating'],color=colors)
for bar in bars:
    height = bar.get_height()  #
    ax5.text(
        bar.get_x() + bar.get_width() / 2, 
        height + 0,  
        f'{height:.1f}',  
        ha='center', va='bottom', fontsize=10, color='black'
    )
ax5.set_xticklabels(subw_data_sentiment_gr['city'],rotation=90)
plt.tight_layout()
fig5.savefig('./files/modeling_output/figures/sentiment_top_bar.png')


# 
# In conclusion, we see the main cities with an average of 4.9 of rating wich is excelent performance.

# 
# ## Summary

# 
# - Currently the brand is geting to 61 % of brazil, however, building restaurants in other cities with the highest population is needed to get to the additional 2% of brazil's population.
# 
# - To stay away from the competence, Subway could build restaurants in the north-east and east of brazil in order to get good sales and don't overlap with McDonalds.
# 
# - Lastly, Subway must keep the performance of their main restaurants, therefore restaurants in cities such as Sao Paulo, Brasilia, Belo Horizonte, Rio de Janeiro and so on, will get more sales and bring satisfaction tu the customers.


