## Interactive Travel Route Planner

Interactive Travel Route Planner is a Python-based web application built with Streamlit, designed to calculate the optimal travel route between multiple locations. It utilizes various libraries such as Pandas for data manipulation, Geopy for calculating geographical distances, and Docplex for solving the Traveling Salesman Problem.

### Features

- Interactive User Interface: Built with Streamlit, providing a user-friendly interface.
- Optimal Route Calculation: Solves the Traveling Salesman Problem to find the shortest possible route that visits a given set of locations.
- Geocoding: Converts location names to geographical coordinates.
- Distance Calculation: Calculates geographical distances between locations using Geopy.
- Google Maps Integration: Generates a link to view the optimal route on Google Maps.

### Installation
To run the application, you need to have Python and Pip installed. If you don’t have Pip installed, you can download it here. Then, you can install the required libraries using the following command:
##### pip install pandas streamlit docplex geopy requests

To start the application, navigate to the directory containing the script in your terminal, and run:
##### streamlit run script_name.py

### Usage
1. Enter Locations: In the text area provided, enter the locations you wish to visit, each on a new line.
2. Calculate Optimal Route: Click the "Calculate Optimal Route" button to calculate the optimal travel route.
3. View Results: The optimal route, along with the distances between locations, will be displayed on the screen.
4. Google Maps Link: A link to view the optimal route on Google Maps is provided.

### Requirements
Python 3.6 or higher
Pandas
Streamlit
Docplex
Geopy
Requests

### Contributions
Contributions to the Travel Route Planner project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

### License
This project is licensed under the [MIT License](LICENSE).
