import re
import pandas as pd
from docplex.mp.model import Model
import streamlit as st
from geopy.distance import distance as geo_distance
from geopy.distance import geodesic
import requests
import random
import math


def display_route(location_route, x, locations, loc_df, distance_matrix):
    num_locations = len(locations)
    route = [0]
    current_place = 0

    location_route_with_coordinates = []
    for loc in location_route:
        if isinstance(loc, str):
            location = loc_df[loc_df['Place_Name'] == loc]['Coordinates'].values[0]
            if location:
                location_route_with_coordinates.append(location)
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

        # Calculate the geodesic distance between two locations
        distance = geodesic(loc, next_loc).kilometers
        distance_km_text = f"{distance:.2f} km"
        distance_mi_text = f"{distance*0.621371:.2f} mi"

        a = loc_df[loc_df['Coordinates'] == loc]['Place_Name'].reset_index(drop=True)[0]
        b = loc_df[loc_df['Coordinates'] == next_loc]['Place_Name'].reset_index(drop=True)[0]
        
        if i == 0:
            location_route_names.append(a.replace(' ', '+') + '/')
            initial_loc = (a.replace(' ', '+')) + '/'
        else:
            location_route_names.append(a.replace(' ', '+') + '/')

        distance_total += distance
        rows.append((a, b, distance_km_text, distance_mi_text))

    distance_total = int(round(distance_total, 0))
    st.write('\n')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Optimal Geodesic Distance", '{} km'.format(distance_total))
        
    df = pd.DataFrame(rows, columns=["From", "To", "Distance (km)","Distance (mi)"]).reset_index(drop=True)
    
    st.dataframe(df)  # display route with distance
    location_route_names.append(initial_loc)
    return location_route_names    
    
def tsp_solver(data_model, iterations=1000, temperature=10000, cooling_rate=0.95):
    def distance(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    num_locations = data_model['num_locations']
    locations = [(float(lat), float(lng)) for lat, lng in data_model['locations']]

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

# Caching the distance matrix calculation for better performance
@st.cache_data
def compute_distance_matrix(locations):    
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

def geocode_address(address):
    url = f'https://photon.komoot.io/api/?q={address}'
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if results['features']:
            first_result = results['features'][0]
            latitude = first_result['geometry']['coordinates'][1]
            longitude = first_result['geometry']['coordinates'][0]
            return address, latitude, longitude
        else:
            print(f'Geocode was not successful. No results found for address: {address}')
    else:
        print('Failed to get a response from the geocoding API.')
        
def main():
    st.title("Interactive Travel Route Planner")

    default_locations = [['Houston'],['Austin'],['Dallas']]
    existing_locations = '\n'.join([x[0] for x in default_locations])
    selected_value = st.text_area("Enter Locations:", value=existing_locations)

    if st.button("Calculate Optimal Route"):
        lines = selected_value.split('\n')
        values = [geocode_address(line) for line in lines if line.strip()]    
        location_names=[x[0] for x in values if x is not None] # address names
        locations=[(x[1],x[2]) for x in values if x is not None] # coordinates        
        loc_df = pd.DataFrame({'Coordinates': locations, 'Place_Name': location_names})    
        
        if locations:
                data_model = create_data_model(locations)
                solution, x = tsp_solver(data_model)

                if solution:
                    distance_matrix = compute_distance_matrix(locations)
                    location_route_names = display_route(solution, x, locations, loc_df, distance_matrix)
                    gmap_search = 'https://www.google.com/maps/dir/+'
                    gmap_places = gmap_search + ''.join(location_route_names)
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
    st.write('#### **About**')
    st.info(
     """
                Created with GPT-4 by:
                [Parthasarathy Ramamoorthy](https://www.linkedin.com/in/parthasarathyr97/) (Data Scientist @ Walmart Global Tech)
            """)
    
if __name__ == "__main__":
    main()
