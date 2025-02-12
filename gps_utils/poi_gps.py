import csv
import folium
import math
import pandas as pd
from collections import defaultdict


class POIGPSProcessor:
    
    def __init__(self, poi_gps_list):
        self.poi_gps_list = poi_gps_list
        self.segment_bearings = self._calculate_segment_bearings()

    def _calculate_segment_bearings(self):
        """
        Calculate the average bearing for each segment of the route.
        """
        segment_bearings = []
        for i in range(len(self.poi_gps_list) - 1):
            lat1, lon1 = self.poi_gps_list[i]["lat"], self.poi_gps_list[i]["lon"]
            lat2, lon2 = self.poi_gps_list[i + 1]["lat"], self.poi_gps_list[i + 1]["lon"]
            segment_bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
            segment_bearings.append(segment_bearing)
        return segment_bearings

    # def 

    def calculate_new_gps(self):
        pass
    
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
        return "Right" if 0 <= delta <= 180 else "Left"



class POIProcessor:
    def __init__(self, poi_gps_list):
        # results
        self.poi_gps_list = poi_gps_list
        self.segment_bearings = self._calculate_segment_bearings()
        self.poi_groups = defaultdict(list)
        self.seen_records = set()
        self.data_to_write = []

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
        # TODO: Decide wheter to put the POI based on route_bearing
        # 90 degree is East
        # 180 degree is South
        # 270 degree is West
        # 360 degree is North
        delta = (poi_bearing - route_bearing + 360) % 360
        return "Right" if 0 <= delta <= 180 else "Left"

    def _calculate_segment_bearings(self):
        """
        Calculate the average bearing for each segment of the route.
        """
        segment_bearings = []
        for i in range(len(self.poi_gps_list) - 1):
            lat1, lon1 = self.poi_gps_list[i]["coordinate"]["lat"], self.poi_gps_list[i]["coordinate"]["lon"]
            lat2, lon2 = self.poi_gps_list[i + 1]["coordinate"]["lat"], self.poi_gps_list[i + 1]["coordinate"]["lon"]
            segment_bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
            segment_bearings.append(segment_bearing)
        return segment_bearings

    # def calculate_
    
    def process_pois(self):
        """
        Process Nearby Search data and calculate necessary information.

        """
        
        for i, result in enumerate(self.poi_gps_list):
            source_lat = result["coordinate"]["lat"]
            source_lon = result["coordinate"]["lon"]
            nearby_results = result["nearby"].get("result", [])

            # Use the average bearing of the current segment
            current_bearing = self.segment_bearings[i] if i < len(self.segment_bearings) else self.segment_bearings[-1]

            for poi in nearby_results:
                name = poi.get("name")
                location = poi.get("location", {})
                poi_lat = location.get("lat")
                poi_lon = location.get("lon")
                types = poi.get("types", [])

                # Calculate distance and bearing
                distance, poi_bearing = self.haversine_and_bearing(source_lat, source_lon, poi_lat, poi_lon)

                side = self.determine_side(poi_bearing, current_bearing)
                # TODO : rewrite this function to calculate average distance and average bearing

                record_key = (source_lat, source_lon, poi_lat, poi_lon, name)
                if record_key in self.seen_records:
                    continue

                self.seen_records.add(record_key)

                self.data_to_write.append([
                    source_lat, source_lon,
                    name, poi_lat, poi_lon,
                    distance, poi_bearing,
                    side, ",".join(types),  
                    None  
                ])

                # Update POI
                self.poi_groups[name].append((side, source_lat, source_lon))

        self._update_shared_details()

    def _update_shared_details(self):
        """
        Update shared POI group information.
        """
        for row in self.data_to_write:
            poi_name = row[2]
            shared_sources = self.poi_groups[poi_name]
            if len(shared_sources) > 1:
                # Đếm số lần trái/phải
                side_counts = {"Left": 0, "Right": 0}
                for side, _, _ in shared_sources:
                    side_counts[side] += 1
                row[7] = "Right" if side_counts["Right"] > side_counts["Left"] else "Left"

            row[-1] = "; ".join([f"{lat}, {lon}" for _, lat, lon in shared_sources])

    def write_to_csv(self, output_path):
        """
        Write processed results to a CSV file.
        """
        header = [
            "source_lat", "source_lon",
            "poi_name", "poi_lat", "poi_lon",
            "distance", "bearing",
            "side", "all_types", "shared_with"
        ]

        with open(output_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.data_to_write)


class RoutePOIVisualizer:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def visualize_route_and_poi(self, output_html):
        """
        Visualize the route and POIs from a CSV file on a map, including side information.
        """
        data = pd.read_csv(self.csv_path)

        required_columns = {'source_lat', 'source_lon', 'poi_name', 'poi_lat', 'poi_lon', 'side'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"File CSV cần các cột sau: {required_columns}")

        route_points = data[['source_lat', 'source_lon']].drop_duplicates().values.tolist()
        pois = data[['poi_name', 'poi_lat', 'poi_lon', 'side']].values.tolist()

        map_center = route_points[0] if route_points else [0, 0]
        my_map = folium.Map(location=map_center, zoom_start=15)

        # Plot route points on the map
        folium.PolyLine(
            locations=route_points,
            color='blue',
            weight=4,
            opacity=0.8,
            tooltip="Route"
        ).add_to(my_map)

        # Add markers for POIs
        for poi_name, poi_lat, poi_lon, side in pois:
            color = "green" if side == "Right" else "red"

            folium.Marker(
                location=[poi_lat, poi_lon],
                popup=f"POI: {poi_name}\nSide: {side}",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(my_map)

        my_map.save(output_html)
        print(f"Map has been saved to: {output_html}")
# Ví dụ sử dụng:
# results = [...]  # Dữ liệu đầu vào
# processor = POIProcessor(results)
# processor.process_pois()
# processor.write_to_csv("output.csv")

# visualizer = RoutePOIVisualizer("output.csv")
# visualizer.visualize_route_and_poi("route_and_poi_side_verification.html")
