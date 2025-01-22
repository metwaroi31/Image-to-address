import subprocess
import geopy.distance
from image_utils.read_metadata import (
    decimal_to_dms,
    format_dms
)
from gps_utils.gps_handler import GPSHandler
from datetime import datetime
from PIL.ExifTags import TAGS
import numpy as np
import pandas as pd


def _run_command_UNIX(commands):
    return subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def _extract_gps_data(video_file):
    # Run ExifTool command to extract GPS data
    commands = [
        'exiftool',
        '-ee',
        '-p',
        'gpx.fmt',
        '-ext',
        'mp4',
        '-w',
        'video.gpx',
        video_file
    ]
    result = _run_command_UNIX(commands)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return video_file.replace('.MP4', 'video.gpx', 1)

def cut_frames(video_file):
    result = _extract_gps_data(video_file)
    gps_handler = GPSHandler(result)
    gps_data = gps_handler.interpolate_gps(gps_data)
    streetview_points = gps_handler.correct_gps_with_route_and_geo_api()
    _extract_frame(video_file, streetview_points, gps_handler.gps_data)

def extract_single_frame(
    video_file,
    gps_image,
    difference
):
    formatted_difference = str(difference)
    commands_cut_frame = [
        'ffmpeg',
        '-i',
        video_file,
        '-ss',
        formatted_difference,
        '-vframes',
        '1',
        'images/' + formatted_difference + '.jpg'
    ]
    _run_command_UNIX(commands_cut_frame)
    lat_deg, lat_min, lat_sec = decimal_to_dms(float(gps_image['lat']))
    long_deg, long_min, long_sec = decimal_to_dms(float(gps_image['lon']))
    lat = format_dms(float(gps_image['lat']), lat_deg, lat_min, lat_sec)
    long = format_dms(float(gps_image['lon']), long_deg, long_min, long_sec)

    commands_write_metadata = [
            'exiftool',
            '-overwrite_original',
            '-Latitude=' + lat,
            '-Longitude=' + long,
            'images/' + formatted_difference + '.jpg'
    ]
    _run_command_UNIX(commands_write_metadata)
    return 'images/' + formatted_difference + '.jpg'

def _extract_frame(video_file, streetview_points, gps_data):
    start_time = gps_data[0]['time']
    
    for point in streetview_points:
        datetime_point = point['time']
        date_format = "%Y-%m-%d %H:%M:%S"
        lat_deg, lat_min, lat_sec = decimal_to_dms(float(point['lat']))
        long_deg, long_min, long_sec = decimal_to_dms(float(point['lon']))
        lat = format_dms(float(point['lat']), lat_deg, lat_min, lat_sec)
        long = format_dms(float(point['lon']), long_deg, long_min, long_sec)
        # Convert the string to a datetime object
        gps_datetime = datetime.strptime(datetime_point, date_format)
        start_datetime = datetime.strptime(start_time, date_format)
        difference = gps_datetime - start_datetime
        # Format the difference as %H:%M:%S
        formatted_difference = str(difference)
        print (formatted_difference)
        commands_cut_frame = [
            'ffmpeg',
            '-i',
            video_file,
            '-ss',
            formatted_difference,
            '-vframes',
            '1',
            'images/' + formatted_difference +
            '.jpg'
        ]
        _run_command_UNIX(commands_cut_frame)
        commands_write_metadata = [
            'exiftool',
            '-overwrite_original',
            '-Latitude=' + lat,
            '-Longitude=' + long,
            'images/' + formatted_difference + '.jpg'
        ]
        _run_command_UNIX(commands_write_metadata)
