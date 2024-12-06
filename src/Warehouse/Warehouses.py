# Warehouses
# # Description
'''Imaginary Shipping Company (distribution business) currently has 21 locations in Brazil (see Warehouse_Locations.xlsx file in data folder). 
Each warehouse location can supply kitchen equipment and disposables for restaurants in a 300 Km radius. 
How many additional warehouse locations does the company need to service 100% of Subway locations, and where should these locations be? 
Optimize the locations so that the company can build the minimum number of new warehouses to cover the restaurant locations that are currently not serviced.'''

import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
import folium
import os

warehouse=pd.read_excel('./files/datasets/input/Warehouse_Locations.xlsx',sheet_name='Sheet1')
subway=pd.read_csv('./files/datasets/output/subw_data.csv')

if os.path.exists('./files/modeling_output/figures/'):
    os.mkdir('./files/modeling_output/figures/')

output_path='./files/modeling_output/figures/'
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    s2 = s1.replace('-','')
    s3 = s2.replace(',','')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s3).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data


warehouse = columns_transformer(warehouse)


warehouse_data=warehouse[['latitude','longitude']].values
subway_data=subway[['latitude','longitude']].values
radius=300

# Calculate the uncovered
uncovered=0
for restaurant in subway_data:
        if not any(geodesic(restaurant, warehouse).km <= radius for warehouse in warehouse_data):
            uncovered+=1

# Check if all restaurants are covered by the warehouses
def all_restaurants_covered(restaurants, warehouses, radius):
    for restaurant in restaurants:
        if not any(geodesic(restaurant, warehouse).km <= radius for warehouse in warehouses):
            return False
    return True
def optimize_warehouses(restaurants, initial_warehouses, radius):
    # Combine restaurants and warehouses as numpy arrays
    uncovered_restaurants = restaurants.copy()
    all_warehouses = np.array(initial_warehouses)
    
    # Start with the current warehouses
    num_new_warehouses = 0

    while True:
        # Check if all restaurants are covered
        if all_restaurants_covered(uncovered_restaurants, all_warehouses, radius):
            break

        # Increment number of warehouses to add
        num_new_warehouses += 1

        # Cluster the uncovered restaurants
        kmeans = KMeans(n_clusters=num_new_warehouses, random_state=42)
        uncovered_clusters = kmeans.fit(uncovered_restaurants)

        # Add new warehouses at cluster centroids
        new_warehouses = kmeans.cluster_centers_
        all_warehouses = np.vstack((initial_warehouses, new_warehouses))

    return all_warehouses, num_new_warehouses,new_warehouses


# Run the optimization
optimized_warehouses, num_additional_warehouses,new_warehouses = optimize_warehouses(
    restaurants=subway_data,
    initial_warehouses=warehouse_data,
    radius=radius
)
new_warehouses_df=pd.DataFrame(new_warehouses,columns=['latitude','longitude'])

new_warehouses_df.to_excel('./files/modeling_output/reports/new_warehouses.xlsx',index=False)
# Calculate the uncovered new
uncovered_new=0
for restaurant in subway_data:
        if not any(geodesic(restaurant, warehouse).km <= radius for warehouse in optimized_warehouses):
            uncovered_new+=1

coverage=100-(100*uncovered/subway_data.shape[0])
coverage_new=100-(100*uncovered_new/subway_data.shape[0])

coverage_df=pd.DataFrame({'Level':['Existent Warehouses','New Warehouses'],'Percentage of coverage':[coverage,coverage_new]})

fig1,ax1=plt.subplots(figsize=(5,5))

bars=ax1.bar(coverage_df['Level'],coverage_df['Percentage of coverage'],color='red',width=0.2)
ax1.set_ylabel('% Percentage of coverage')


for bar in bars:
    height = bar.get_height()  #
    ax1.text(
        bar.get_x() + bar.get_width() / 2, 
        height + 1,  
        f'{height:.1f}%',  
        ha='center', va='bottom', fontsize=10, color='black'
    )
plt.tight_layout()
fig1.savefig(output_path+'coverage.png')

# Plotting existing and new warehouses
fig,ax=plt.subplots(figsize=(12, 8))

# Existing warehouses
ax.scatter(warehouse['longitude'], warehouse['latitude'], color='blue', label='Existing Warehouses', s=100)

# Restaurants
ax.scatter(subway['longitude'], subway['latitude'], color='red', label='Restaurants', s=50)

# New warehouses
ax.scatter(optimized_warehouses[len(warehouse_data):, 1], 
            optimized_warehouses[len(warehouse_data):, 0], 
            color='green', label='New Warehouses', s=100, marker='x')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Optimized Warehouse Locations')
ax.legend()
fig.tight_layout()
fig.savefig(output_path+'new_warehouses.png')

# Initialize map with old warehouses
m = folium.Map(
    location=[-14.2350, -51.9253],  # center around Africa
    zoom_start=4,  # dezoom
    tiles='cartodb positron'  # background style
)

for idx, row in subway.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['city']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}', 
        tooltip=row['city'] 
    ).add_to(m)
for idx, row in warehouse.iterrows():
    folium.Circle(
        location=[row['latitude'], row['longitude']],
        color='red',
        fill=True,
        radius=300000,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['nm_mun']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}', 
        tooltip=row['nm_mun'] 
    ).add_to(m)

legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
        <b>Map</b><br>
        <ul>
            <li><span style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></span> Restaurant</li>
            <li><span style="background-color: red; width: 20px; height: 20px; display: inline-block;"></span> Warehouse</li>
        </ul>
    </div>
"""

# Añadir la leyenda al mapa
m.get_root().html.add_child(folium.Element(legend_html))

m.save(output_path+"warehouse_map.html")


# Initialize map with new warehouses
m = folium.Map(
    location=[-14.2350, -51.9253],  # center around Africa
    zoom_start=4,  # dezoom
    tiles='cartodb positron'  # background style
)

for idx, row in subway.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b>{row['city']}</b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}', 
        tooltip=row['city'] 
    ).add_to(m)
for idx, row in new_warehouses_df.iterrows():
    folium.Circle(
        location=[row['latitude'],row['longitude']],
        color='yellow',
        fill=True,
        radius=300000,
        fill_opacity=0.5,
        weight=1,
        popup=f'<b><br>Lat: {row['latitude']}<br>Lon: {row['longitude']}'
    ).add_to(m)
legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
        <b>Mapa de Ciudades</b><br>
        <ul>
            <li><span style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></span> Restaurant</li>
            <li><span style="background-color: yellow; width: 20px; height: 20px; display: inline-block;"></span> Warehouse</li>
        </ul>
    </div>
"""

# Añadir la leyenda al mapa
m.get_root().html.add_child(folium.Element(legend_html))

m.save(output_path+"warehouse_map_new.html")
