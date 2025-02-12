import xml.etree.ElementTree as ET
from apis.map4d import Map4dSDKClient
from scipy.interpolate import interp1d
import geopy.distance
import numpy as np
import pandas as pd
import math


class GPSHandler:
    def __init__(self, gpx_file):
        tree = ET.parse(gpx_file)
        root = tree.getroot()
        namespace = {'ns': 'http://www.topografix.com/GPX/1/0'}
        self.gps_data = []
        self.interpolated_gps_data = []
        self.corrected_gps_data_with_api = []
        self.gps_data_for_frames = []
        self.gps_data_with_geo_coding = []
        self.map_api_client = Map4dSDKClient()
        self.video_duration = float(root.find('ns:desc', namespace).text)
        # This should be calculated after interpolated and corrected with API
        self.path_length = 0
        for elem in root.iter():
            if 'trkpt' in elem.tag:
                lat = elem.get('lat')
                lon = elem.get('lon')
                time = elem.get('time')
                self.gps_data.append({
                    'lat': lat,
                    'lon': lon,
                    'time': time
                })
        self.orignal_gps_length = len(self.gps_data)
        print ("video duration : " + str(self.video_duration))
    
    def _calculate_distance(self, lat1 ,lon1, lat2, lon2):
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return geopy.distance.distance(coords_1, coords_2).meters
    
    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """
        Calculate the bearing from point A (lat1, lon1) to point B (lat2, lon2).
        """
        delta_lon = math.radians(lon2 - lon1)
        lat1, lat2 = math.radians(lat1), math.radians(lat2)

        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

        initial_bearing = math.atan2(x, y)
        return (math.degrees(initial_bearing) + 360) % 360

    @staticmethod
    def haversine_and_bearing(lat1, lon1, lat2, lon2):
        """
          Calculate the distance and bearing between two coordinates.
        """
        # Convert to radians
        EARTH_RADIUS = 6371 * 1000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        
        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = EARTH_RADIUS * c

        # Calculate bearing
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

        return distance, bearing
    
    @staticmethod
    def determine_side(poi_bearing, route_bearing):
        """
        Determine whether the POI is on the left or right side based on the bearing.
        """
        delta = (poi_bearing - route_bearing + 360) % 360
        return 0 if 0 <= delta <= 180 else 1

    
    @staticmethod
    def calculate_average_distance(distances):
        """
        Calculate the average distance.
        """
        return np.mean(distances)

    @staticmethod
    def calculate_new_position(lat, lon, bearing, distance):
        """
        Calculate new latitude and longitude based on starting point, bearing, and distance.
        Uses haversine formula to calculate the new position.
        """
        R = 6371000  # Radius of Earth in meters
        bearing = np.radians(bearing)

        lat1 = np.radians(lat)
        lon1 = np.radians(lon)

        lat2 = np.arcsin(np.sin(lat1) * np.cos(distance / R) + np.cos(lat1) * np.sin(distance / R) * np.cos(bearing))
        lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(lat1),
                                 np.cos(distance / R) - np.sin(lat1) * np.sin(lat2))

        return np.degrees(lat2), np.degrees(lon2)

    @staticmethod
    def find_min_bearing(poi_group):
        """
        Find the point with the minimum bearing in the group.
        """
        
        min_bearing_point = min(poi_group, key=lambda x: x[2])  # Assuming bearing is at index 2
        return min_bearing_point

    def interpolate_gps(
        self,
        num_points=30,
        threshold_distance=400
    ):
        df = pd.DataFrame(self.gps_data)
        df['time'] = pd.to_datetime(df['time'])
        times = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
        lat_interp = interp1d(times, df['lat'], kind='linear', fill_value="extrapolate")
        lon_interp = interp1d(times, df['lon'], kind='linear', fill_value="extrapolate")
        new_times = np.linspace(times.min(), times.max(), num_points)

        # Get interpolated values
        new_lat = lat_interp(new_times)
        new_lon = lon_interp(new_times)
        self.interpolated_gps_data = [
            {
                'lat': new_lat[0],
                'lon': new_lon[0],
                'time': new_times[0]
            }
        ]
        for i in range(1, len(new_lat)):
            last_gps = len(self.interpolated_gps_data) - 1
            lat_1 = self.interpolated_gps_data[last_gps]['lat']
            lon_1 = self.interpolated_gps_data[last_gps]['lon']
            lat_2 = new_lat[i]
            lon_2 = new_lon[i]
            time = new_times[i]
            coords_1 = (lat_1, lon_1)
            coords_2 = (lat_2, lon_2)
            distance = geopy.distance.distance(coords_1, coords_2).meters
            if distance < threshold_distance:
                self.interpolated_gps_data.append({
                    'lat': new_lat[i],
                    'lon': new_lon[i],
                    'time': time 
                })
        return self.interpolated_gps_data
    
    def correct_gps_with_route_and_geo_api(
        self,
        list_collect_street=[
            'Đ. số 5',
            'duong so 5',
            'Đường số 5',
            'Nguyễn Thị Thập',
            'nguyen thi thap'
        ],
    ):
        for i in range(1, len(self.interpolated_gps_data)):
            start_lat = self.interpolated_gps_data[i - 1]['lat']
            start_lon = self.interpolated_gps_data[i - 1]['lon']
            end_lat = self.interpolated_gps_data[i]['lat']
            end_lon = self.interpolated_gps_data[i]['lon']
            data_route = self.map_api_client.api_route(
                start_lat,
                start_lon,
                end_lat,
                end_lon
            )
            coords_route_list = self._from_steps_route_to_coordinates(
                data_route['result']['routes'][0]['legs'][0]['steps']
            )
            for coord in coords_route_list:
                # TODO :
                # Get coords and interpolate time
                geo_data = self.map_api_client.api_geo_coding(
                    coord['lat'],
                    coord['lon']
                )
                address_component = self._parse_address(
                    geo_data['result'][0]['addressComponents']
                )
                if 'street' in address_component.keys():
                    street_name_checked = address_component['street'] in list_collect_street
                    if street_name_checked:
                        coord['time'] = self.interpolated_gps_data[i]['time']
                        self.corrected_gps_data_with_api.append(coord)
        self._calculate_total_path_length()
        return self.corrected_gps_data_with_api

    def create_gps_for_frames(
        self
    ):
        gps_data_df = pd.DataFrame(self.corrected_gps_data_with_api)
        gps_data_numpy = gps_data_df[['lat', 'lon']].to_numpy()
        # Calculate the total distance along the path
        distances = np.sqrt(np.sum(np.diff(gps_data_numpy, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        # Calculate the cumulative distance
        cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

        # Generate 235 evenly spaced distances along the path
        even_distances = np.linspace(0, total_distance, self.orignal_gps_length)

        frames = np.zeros((self.orignal_gps_length, 2))
        for i in range(2):
            frames[:, i] = np.interp(even_distances, cumulative_distance, gps_data_numpy[:, i])

        df = pd.DataFrame(frames, columns=['lat', 'lon'])
        self.gps_data_for_frames = df.to_dict(orient='records')
        current_seconds = 0
        average_speed = (self.path_length / self.video_duration) * 1.1
        for i in range(1, len(self.gps_data_for_frames)):            
            current_distance = self._calculate_distance(
                self.gps_data_for_frames[i - 1]['lat'],
                self.gps_data_for_frames[i - 1]['lon'],
                self.gps_data_for_frames[i]['lat'],
                self.gps_data_for_frames[i]['lon']
            )
            self.gps_data_for_frames[i - 1]['time'] = current_seconds
            seconds = current_distance / average_speed 
            current_seconds = current_seconds + seconds
            self.gps_data_for_frames[i]['time'] = current_seconds
        return self.gps_data_for_frames
    
    def _calculate_total_path_length(self):
        for i in range(1, len(self.corrected_gps_data_with_api)):
            coords_one = self.corrected_gps_data_with_api[i-1]
            coords_two = self.corrected_gps_data_with_api[i]
            distance = geopy.distance.distance(
                (coords_one['lat'], coords_one['lon']),
                (coords_two['lat'], coords_two['lon'])
            ).meters
            self.path_length = self.path_length + distance
    
    def load_coordinates_from_csv(self, csv_file, coords_type):
        coords_df = pd.read_csv(csv_file)
        gps_list = []
        for i, row in coords_df.iterrows():    
            gps_list.append(row.to_dict())
        if coords_type == 2:
            self.corrected_gps_data_with_api = gps_list
            self._calculate_total_path_length()
            print('total length : ' + str(self.path_length))
        elif coords_type == 3:
            self.gps_data_for_frames = gps_list
        
    def _parse_address(self, address_component):
        address_data = address_component
        return_json = {}
        for component in address_data:
            component_type = component['types'][0]
            name = component['name']
            if 'housenumber' in component_type:
                return_json['housenumber'] = name
            if 'street' in component_type:
                return_json['street'] = name
            if 'admin_level_4' in component_type:
                return_json['admin_level_4'] = name
            if 'admin_level_3' in component_type:
                return_json['admin_level_3'] = name
            if 'admin_level_2' in component_type:
                return_json['admin_level_2'] = name
        return return_json
    
    def _from_steps_route_to_coordinates(self, steps):
        lat_lon_list = []
        for step in steps:
            start_lat = step['startLocation']['lat']
            start_lon = step['startLocation']['lng']
            end_lat = step['endLocation']['lat']
            end_lon = step['endLocation']['lng']
            lat_lon_list.append(
                {
                    'lat': start_lat,
                    'lon': start_lon
                }
            )
            lat_lon_list.append(
                {
                    'lat': end_lat,
                    'lon': end_lon
                }
            )
        return lat_lon_list
