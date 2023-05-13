import re
import pandas as pd
import streamlit as st
from geopy.distance import distance as geo_distance
import pyarrow
import openrouteservice
from streamlit_searchbox import st_searchbox
import requests
import random
import math
import base64
import plotly.graph_objects as go

def is_coordinate(input_string):
    coordinate_pattern = r'^-?\d+(\.\d+)?\s+-?\d+(\.\d+)?$'
    return bool(re.match(coordinate_pattern, input_string))

# Caching the geocoding results for better performance
@st.cache_data
def geocode_location(location):
    return geolocator.geocode(location)

def midpoint_coordinates(coords):
    if len(coords) > 0:
        index = len(coords) // 2
        return coords[index]
    else:
        return None

def display_route(location_route, x, locations, loc_df, distance_matrix):
    api_key = st.secrets['ORS_API_KEY']  
    client = openrouteservice.Client(key=api_key)
    num_locations = len(locations)
    route = [0]
    current_place = 0

    location_route_with_coordinates = []
    for loc in location_route:
        if isinstance(loc, str):
            location = geolocator.geocode(loc)
            if location:
                location_route_with_coordinates.append((location.latitude, location.longitude))
            else:
                location_route_with_coordinates.append(None)
        else:
            location_route_with_coordinates.append(loc)

    st.write('\n')

    rows = []
    distance_total = 0
    initial_loc = ''  # starting point
    location_route_names = []  # list of final route place names in order

    for i, loc in enumerate(location_route_with_coordinates[:-1]):
        next_loc = location_route_with_coordinates[i + 1]

        # Get the actual distance between two locations based on road network using openrouteservice
        route_data = client.directions(coordinates=[(loc[1], loc[0]), (next_loc[1], next_loc[0])], profile='driving-car',
                                       format='geojson', radiuses=-1)
        distance = route_data['features'][0]['properties']['segments'][0]['distance'] / 1000
        distance_text = f"{distance:.2f} km"

        a = loc_df[loc_df['Coordinates'] == loc]['Place_Name'].reset_index(drop=True)[0]
        b = loc_df[loc_df['Coordinates'] == next_loc]['Place_Name'].reset_index(drop=True)[0]

        if i == 0:
            location_route_names.append(a.replace(' ', '+') + '/')
            initial_loc = (a.replace(' ', '+')) + '/'
        else:
            location_route_names.append(a.replace(' ', '+') + '/')

        distance_total += distance
        rows.append((a, b, distance))

    distance_total = int(round(distance_total, 0))
    st.write('\n')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Distance", '{} km'.format(distance_total))
        
    df = pd.DataFrame(rows, columns=["From", "To", "Distance (km)"]).reset_index(drop=True)
    df['Distance (km)']=round(df['Distance (km)'],1)
    
    # display route with distance
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='lightblue',
                    align='left'),
        cells=dict(values=[df["From"], df["To"], df["Distance (km)"]],
                   fill_color='white',
                   align='left'))
    ])

    st.plotly_chart(fig)        
    # Create a download button for the DataFrame
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="optimal_route.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    location_route_names.append(initial_loc)

    return location_route_names
    
def tsp_solver(data_model, iterations=1000, temperature=10000, cooling_rate=0.95):
    def distance(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    num_locations = data_model['num_locations']
    locations = [(lat, lng) for lat, lng in data_model['locations']]

    # Randomly generate a starting solution
    current_solution = list(range(num_locations))
    random.shuffle(current_solution)

    # Compute the distance of the starting solution
    current_distance = 0
    for i in range(num_locations):
        current_distance += distance(locations[current_solution[i-1]], locations[current_solution[i]])

    # Initialize the best solution as the starting solution
    best_solution = current_solution
    best_distance = current_distance

    # Simulated Annealing algorithm
    for i in range(iterations):
        # Compute the temperature for this iteration
        current_temperature = temperature * (cooling_rate ** i)

        # Generate a new solution by swapping two random locations
        new_solution = current_solution.copy()
        j, k = random.sample(range(num_locations), 2)
        new_solution[j], new_solution[k] = new_solution[k], new_solution[j]

        # Compute the distance of the new solution
        new_distance = 0
        for i in range(num_locations):
            new_distance += distance(locations[new_solution[i-1]], locations[new_solution[i]])

        # Decide whether to accept the new solution
        delta = new_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / current_temperature):
            current_solution = new_solution
            current_distance = new_distance

        # Update the best solution if the current solution is better
        if current_distance < best_distance:
            best_solution = current_solution
            best_distance = current_distance

    # Convert the solution to the required format
    x = {}
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                if (i, j) in x:
                    continue
                if (j, i) in x:
                    continue
                if (i == 0 and j == num_locations - 1) or (i == num_locations - 1 and j == 0):
                    x[i, j] = 1
                    x[j, i] = 1
                elif i < j:
                    x[i, j] = 1
                    x[j, i] = 0
                else:
                    x[i, j] = 0
                    x[j, i] = 1

    # Create the optimal route
    optimal_route = []
    start_index = best_solution.index(0)
    for i in range(num_locations):
        optimal_route.append(best_solution[(start_index+i)%num_locations])
    optimal_route.append(0)
    
    # Return the optimal route
    location_route = [locations[i] for i in optimal_route]
    return location_route, x

def parse_locations_input(locations_input, geolocator):
    locations = []
    input_list = locations_input.split('\n')

    for loc in input_list:
        loc = loc.strip()
        if is_coordinate(loc):
            lat, lng = map(float, loc.split())
            locations.append((lat, lng))
        else:
            location = geocode_location(loc)
            if location:
                locations.append((location.latitude, location.longitude))

    st.write(locations)            
    return locations, input_list

# Caching the distance matrix calculation for better performance
@st.cache_data
def compute_distance_matrix(locations):
    # use openrouteservice distance for more accurate roadroute distance (but high compute time)
#     api_key = "API_KEY"  # Replace this with your actual API key
#     client = openrouteservice.Client(key=api_key)

#     num_locations = len(locations)
#     distance_matrix = []

#     for origin in locations:
#         origin_distances = []
#         for destination in locations:
#             if origin == destination:
#                 origin_distances.append(0)
#             else:
#                 # Get the distance between the origin and destination using the road network
#                 coords = [(origin[1], origin[0]), (destination[1], destination[0])]
#                 route = client.directions(coordinates=coords, profile='driving-car', format='geojson',radiuses=-1)
#                 distance = route['features'][0]['properties']['segments'][0]['distance']
#                 origin_distances.append(distance)
#         distance_matrix.append(origin_distances)
    
    # using geopy geo_distance for lesser compute time
    num_locations = len(locations)
    distance_matrix = [[0] * num_locations for i in range(num_locations)]
    for i in range(num_locations):
        for j in range(i, num_locations):
            distance = geo_distance(locations[i], locations[j]).km
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
            
    return distance_matrix

def create_data_model(locations):
    data = {}
    num_locations = len(locations)
    data['locations']=locations
    data['num_locations'] = num_locations
    data['depot'] = 0

    distance_matrix = compute_distance_matrix(locations)
    data['distance_matrix'] = distance_matrix

    return data

def autocomplete_placenames(word):    
    import requests
    api_key = st.secrets['GMAPS_API_KEY']
    input_text = word # text input for autocomplete
    url = f'https://maps.googleapis.com/maps/api/place/autocomplete/json?input={input_text}&key={api_key}'
    place_ids,place_names=[],[]

    response = requests.get(url)
    resp_json_payload = response.json()
    for prediction in resp_json_payload['predictions']:
        place_names.append(prediction['description'])
    return place_names if word else []

def geocode_address(address):
    url = 'https://maps.googleapis.com/maps/api/geocode/json' 
    params = {
        'address': address,
        'key': st.secrets['GMAPS_API_KEY'] 
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            latitude = data['results'][0]['geometry']['location']['lat']
            longitude = data['results'][0]['geometry']['location']['lng']
            return address, latitude,longitude
        else:
            st.write(f'Geocode was not successful for the following reason: {data["status"]}')
    else:
        st.write(f'')

def main():
    st.title("Interactive Travel Route Planner")

    # st.session_state is a feature in Streamlit that allows you to store and persist data across reruns of your Streamlit app. 
    if 'selected_values' not in st.session_state:
        st.session_state.selected_values = []

    selected_value = st_searchbox(
            autocomplete_placenames,clearable=True)        

    if st.button('Add Location'):
        if selected_value:
            st.session_state.selected_values.append([geocode_address(selected_value)])

    location_names=[x[0][0] for x in st.session_state.selected_values if x is not None] # address names
    locations=[(x[0][1],x[0][2]) for x in st.session_state.selected_values if x is not None] # coordinates        
    st.text_area('',location_names) 
    
    loc_df=pd.DataFrame({'Coordinates':locations,'Place_Name':location_names})

    if st.button("Calculate Optimal Route"):
        if locations:
                data_model = create_data_model(locations)
                solution, x = tsp_solver(data_model)

                if solution:
                    distance_matrix = compute_distance_matrix(locations)
                    location_route_names=display_route(solution, x, locations, loc_df, distance_matrix)
                    gmap_search='https://www.google.com/maps/dir/+'
                    gmap_places=gmap_search+''.join(location_route_names)
                    st.write('\n')
                    st.write('[Google Maps Link with Optimal Route added]({})'.format(gmap_places))
                else:
                    st.error("No solution found.")
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')    
    st.write('\n')    
    st.write('#### **About**')
    st.info(
     """
                Created with GPT-4 by:
                [Parthasarathy Ramamoorthy](https://www.linkedin.com/in/parthasarathyr97/) (Data Scientist @ Walmart Global Tech)
            """)
    
if __name__ == "__main__":
    main()
