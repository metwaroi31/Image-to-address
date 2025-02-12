import numpy as np
import math
import csv
from gps_utils.gps_handler import GPSHandler
from apis.map4d import Map4dSDKClient


class POIOptimizationProcessor:
    
    def __init__(
        self,
        gps_route_list
    ):
        self.map_api_client = Map4dSDKClient()
        self.gps_route_list = gps_route_list
        self.segment_bearings = self._calculate_segment_bearings()
        self.results_for_poi = []
        self.poi_list_with_distance = []

    def _calculate_segment_bearings(self):
        """
        Calculate the average bearing for each segment of the route.
        """
        segment_bearings = []
        for i in range(len(self.gps_route_list) - 1):
            lat1, lon1 = self.gps_route_list[i]["lat"], self.gps_route_list[i]["lon"]
            lat2, lon2 = self.gps_route_list[i + 1]["lat"], self.gps_route_list[i + 1]["lon"]
            segment_bearing = GPSHandler.calculate_bearing(lat1, lon1, lat2, lon2)
            segment_bearings.append(segment_bearing)
        return segment_bearings

    def _from_result_to_poi_gps(
        self,
        poi_dict_from_api
    ):
        lat = poi_dict_from_api['location']['lat']
        lon = poi_dict_from_api['location']['lng']
        return {
            'lat': lat,
            'lon': lon
        }

    def _is_right_of_line(
            self,
            route_prev_lat,
            route_prev_lon,
            route_current_lat,
            route_current_lon,
            poi_lat,
            poi_lon):
        # Convert latitudes and longitudes to radians
        rad_prev_lat, rad_prev_lon = math.radians(route_prev_lat), math.radians(route_prev_lon)
        rad_current_lat, rad_current_lon = math.radians(route_current_lat), math.radians(route_current_lon)
        rad_poi_lat, rad_poi_lon = math.radians(poi_lat), math.radians(poi_lon)
        
        # Calculate vectors
        vector_line = (math.cos(rad_current_lat) * math.cos(rad_current_lon) - math.cos(rad_prev_lat) * math.cos(rad_prev_lon),
                    math.cos(rad_current_lat) * math.sin(rad_current_lon) - math.cos(rad_prev_lat) * math.sin(rad_prev_lon))
        vector_point = (math.cos(rad_poi_lat) * math.cos(rad_poi_lon) - math.cos(rad_prev_lat) * math.cos(rad_prev_lon),
                        math.cos(rad_poi_lat) * math.sin(rad_poi_lon) - math.cos(rad_prev_lat) * math.sin(rad_prev_lon))
        
        # Calculate the cross product of the vectors
        cross_product = vector_line[0] * vector_point[1] - vector_line[1] * vector_point[0]
        
        # Determine if the point is on the right side of the line
        return cross_product < 0  # Point is on the right if the cross product is negative

    def choose_poi_for_optimizing(
        self
    ):
        # TODO:
        # Rewrite calculate bearing for GPS route 
        # Then decide bearing for POI GPS
        for i in range(1, len(self.gps_route_list)):
            prev_lat = self.gps_route_list[i-1]["lat"]
            prev_lon = self.gps_route_list[i-1]["lon"]
            current_lat = self.gps_route_list[i]["lat"]
            current_lon = self.gps_route_list[i]["lon"]
            result = self.map_api_client.api_search_nearby(
                current_lat,
                current_lon,
                radius=15
            )
            nearby_pois = result['result']
            distance_route, route_bearing = GPSHandler.haversine_and_bearing(
                    prev_lat,
                    prev_lon,
                    current_lat,
                    current_lon
                )
            poi_distance_from_source = {
                'source_lat': current_lat,
                'source_lon': current_lon,
                'time': self.gps_route_list[i]['time'],
                'route_bearing': route_bearing,
                'list_poi': []
            }

            for poi in nearby_pois:
                gps_poi = self._from_result_to_poi_gps(
                    poi
                )
                distance, poi_bearing = GPSHandler.haversine_and_bearing(
                    current_lat,
                    current_lon,
                    gps_poi['lat'],
                    gps_poi['lon']
                )
                if self._is_right_of_line(
                        prev_lat,
                        prev_lon,
                        current_lat,
                        current_lon,
                        gps_poi['lat'],
                        gps_poi['lon']
                    ) is False:
                    poi_distance_from_source['list_poi'].append({
                        'poi_lat': gps_poi['lat'],
                        'poi_lon': gps_poi['lon'],
                        'time': self.gps_route_list[i]['time'],
                        'distance': distance,
                        'poi_bearing': poi_bearing
                    })
            if len(poi_distance_from_source['list_poi']) == 0:
                # We have to fake poi gps info so that it will work when we choose poi for optimal
                poi_bearing = (route_bearing + 90) % 360
                optimal_lat, optimal_lon = GPSHandler.calculate_new_position(
                    current_lat, current_lon, poi_bearing, 15
                )
                poi_distance_from_source['list_poi'].append({
                    'poi_lat': optimal_lat,
                    'poi_lon': optimal_lon,
                    'time': self.gps_route_list[i]['time'],
                    'distance': 15,
                    'poi_bearing': poi_bearing
                })
            self.poi_list_with_distance.append(poi_distance_from_source)

    def find_optimal_position(
        self
    ):
        for poi_with_source in self.poi_list_with_distance:
            total_num_pois = len(poi_with_source['list_poi'])
            optimal_bearing = (poi_with_source['route_bearing'] + 90) % 360
            distance = 0
            for poi_info in poi_with_source['list_poi']:
                    distance = distance + poi_info['distance']
            average_distance = distance / total_num_pois
            # avg_bearing = total_bearing / total_num_pois
            optimal_lat, optimal_lon = GPSHandler.calculate_new_position(
                poi_with_source['source_lat'], poi_with_source['source_lon'], optimal_bearing, average_distance
            )
            self.results_for_poi.append({
                'lat': poi_with_source['source_lat'],
                'lon': poi_with_source['source_lon'],
                'time': poi_with_source['time'],
                'poi_lat': optimal_lat,
                'poi_lon': optimal_lon
            })
        return self.results_for_poi
    
    def optimize_poi_positions(self):
        """
        Optimize POI positions based on average distance and minimum bearing.
        """
        poi_groups = {}

        # Step 1: Read the input CSV and group POIs by name
        with open(self.input_csv, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                poi_name = row['poi_name']
                if not poi_name:
                    continue

                source_lat = float(row['source_lat'])
                source_lon = float(row['source_lon'])
                # POI by name
                # POI by tag/cate
                poi_lat = float(row['poi_lat'])
                poi_lon = float(row['poi_lon'])
                distance = float(row['distance'])
                bearing = float(row['bearing'])
                side = row['side'].lower()
              # In giá trị 'side'
                if side != 'right':
                    continue
                
                if poi_name not in poi_groups:
                    poi_groups[poi_name] = []

                poi_groups[poi_name].append((poi_lat, poi_lon, bearing, distance, source_lat, source_lon))

        # Step 2: Optimize positions for each group
        optimized_results = []
        for poi_name, group in poi_groups.items():
            distances = [g[3] for g in group]  # Extract distances
            average_distance = GPSHandler.calculate_average_distance(distances)

            # Find the POI with the minimum bearing
            min_bearing_point = GPSHandler.find_min_bearing(group)

            # Use the average distance and minimum bearing point to determine new position
            optimal_lat, optimal_lon = GPSHandler.calculate_new_position(
                min_bearing_point[0], min_bearing_point[1], min_bearing_point[2], average_distance
            )
            # Add optimized result for each original source point
            for _, _, _, _, source_lat, source_lon in group:
                optimized_results.append([source_lat, source_lon, poi_name, optimal_lat, optimal_lon])

        # Step 3: Write optimized results to a new CSV
        with open(self.output_csv_optimized, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            header = ["source_lat", "source_lon", "poi_name", "optimal_lat", "optimal_lon"]
            writer.writerow(header)
            writer.writerows(optimized_results)

        print(f"Optimized POI positions have been saved to: {self.output_csv_optimized}")