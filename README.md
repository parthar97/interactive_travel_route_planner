## Travel Route Planner

The Travel Route Planner is a Streamlit application that helps users optimize their travel routes based on distance. It allows users to enter a list of locations and then calculates the optimal route using the Simulated Annealing algorithm. The application provides detailed information about the route, including optimal route and distance.

### Features

- Enter a list of locations as input
- Calculate the optimal route using the Simulated Annealing algorithm
- Display detailed information about the route, including distance
- Download the route information as a CSV file
- Refresh the application to start a new optimization

### Installation
1. Clone the repository: `git clone https://github.com/parthar97/interactive_travel_route_planner.git`
2. Navigate to the project directory: `cd interactive_travel_route_planner`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

### Usage

1. Enter the list of locations or coordinates in the provided text area.
2. Click the "Add Location" button to add a location to the list.
3. Click the "Calculate Optimal Route" button to start the optimization process.
4. View the optimal route on the plotly chart with route distance.
5. Use the download button to save the route information as a CSV file.
6. Link to Google Maps with Optimal Route Locations added 
7. Click the "Refresh" button to reset the application and start a new optimization.

### Technologies Used

- Python
- Streamlit: for building the user interface
- Plotly: for visualizing the route as a table
- Openrouteservice: for calculating distances and directions between locations
- Geopy: for geocoding and distance calculations
- Simulated Annealing algorithm: for solving the Traveling Salesman Problem (TSP)

### Contributions

Contributions to the Travel Route Planner project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

### License

This project is licensed under the [MIT License](LICENSE).
