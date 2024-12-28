import subprocess
import xml.etree.ElementTree as ET
import geopy.distance
from datetime import datetime
from PIL.ExifTags import TAGS

def _run_command_UNIX(commands):
    return subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def _extract_gps_data(video_file):
    # Run ExifTool command to extract GPS data
    commands = ['exiftool', '-ee', '-p', 'gpx.fmt', '-ext', 'mp4', '-w', 'video.gpx', video_file]
    result = _run_command_UNIX(commands)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return video_file.replace('.MP4', 'video.gpx', 1)


def _parse_gpx(gpx_file):
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    gps_data = []
    for elem in root.iter():
        if 'trkpt' in elem.tag:
            lat = elem.get('lat')
            lon = elem.get('lon')
            time = elem.get('time')
            gps_data.append({
                'lat': lat,
                'lon': lon,
                'time': time
            })
    return gps_data

def cut_frames(video_file):
    result = _extract_gps_data(video_file)
    gps_data = _parse_gpx(result)
    streetview_points = _get_choosen_gps_data_2(gps_data)
    print (len(gps_data))
    print (len(streetview_points))
    
    _extract_frame(video_file, streetview_points, gps_data)

def _extract_frame(video_file, streetview_points, gps_data):
    start_time = gps_data[0]['time']
    
    for point in streetview_points:
        datetime_point = point['time']
        date_format = "%Y-%m-%d %H:%M:%S"
        lat = point['lat']
        long = point['lon']
        
        # Convert the string to a datetime object
        gps_datetime = datetime.strptime(datetime_point, date_format)
        start_datetime = datetime.strptime(start_time, date_format)
        difference = gps_datetime - start_datetime
        # Format the difference as %H:%M:%S
        formatted_difference = str(difference)
        print (formatted_difference)
        commands_cut_frame = ['ffmpeg', '-i', video_file, '-ss', formatted_difference, '-vframes', '1', 'images/' + formatted_difference + '.jpg']
        _run_command_UNIX(commands_cut_frame)
        commands_write_metadata = ['exiftool', '-overwrite_original', '-Latitude=' + str(lat), '-Longitude=' + str(long), 'images/' + formatted_difference + '.jpg']
        _run_command_UNIX(commands_write_metadata)

def _get_choosen_gps_data(gps_data):
    # Desired interval in meters
    short_distance = 3
    streetview_points = [gps_data[0]]
    # Calculate distances and determine split points
    for i in range(1, len(gps_data)):
        lat_1 = streetview_points[-1]['lat']
        lon_1 = streetview_points[-1]['lon']
        lat_2 = gps_data[i]['lat']
        lon_2 = gps_data[i]['lon']
        gps_data[i]['order'] = i
        coords_1 = (lat_1, lon_1)
        coords_2 = (lat_2, lon_2)
        distance = geopy.distance.distance(coords_1, coords_2).meters
        if distance >= short_distance:
            streetview_points.append(gps_data[i])
    return streetview_points

def _get_choosen_gps_data_2(gps_data):
    # Desired interval in meters
    short_distance = 3
    streetview_points = []
    cumulative_distance = 0
    # Calculate distances and determine split points
    for i in range(1, len(gps_data)):
        lat_1 = gps_data[i-1]['lat']
        lon_1 = gps_data[i-1]['lon']
        lat_2 = gps_data[i]['lat']
        lon_2 = gps_data[i]['lon']
        coords_1 = (lat_1, lon_1)
        coords_2 = (lat_2, lon_2)
        distance = geopy.distance.distance(coords_1, coords_2).meters
        cumulative_distance = cumulative_distance + distance
        if distance >= short_distance:
            streetview_points.append(gps_data[i])
            cumulative_distance = 0
    return streetview_points
