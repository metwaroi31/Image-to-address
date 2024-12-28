import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import subprocess
import datetime

def _run_command_UNIX(commands):
    return subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def convert_to_decimal_degrees(dms):
    """Convert a tuple of (degrees, minutes, seconds) to decimal degrees."""
    if isinstance(dms, tuple) and len(dms) == 3:
        degrees, minutes, seconds = dms
        decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
        return round(decimal_degrees, 7)  # Round to 7 decimal places
    return 'N/A'

def read_exif_with_exifread(image_path):
    image = Image.open(image_path)
    exif_data = {}
    commands_exif_data = ['exiftool', '-json', image_path]
    result = _run_command_UNIX(commands_exif_data)
    exif_data = json.loads(result.stdout)[0]
    print (exif_data)
    exif_data['Width'] = image.width
    exif_data['Height'] = image.height
    
    exif_data['Latitude'] = convert_to_decimal_degrees(exif_data['Latitude'])
    exif_data['Longitude'] = convert_to_decimal_degrees(exif_data['Longitude'])
    return exif_data


# Example usage
def get_exif_data_video(image_path):
    try:
        image = Image.open(image_path)
        exif_data = {}
    
        exif_data['Width'] = image.width
        exif_data['Height'] = image.height

        if hasattr(image, '_getexif'):
            raw_exif = image._getexif()
            print (raw_exif)
            if raw_exif is not None:
                for tag_id, value in raw_exif.items():
                    print (tag_id, value)
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
        exif_data['Latitude'] = convert_to_decimal_degrees(exif_data['Latitude'])
        exif_data['Longitude'] = convert_to_decimal_degrees(exif_data['Longitude'])
        return exif_data
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None


def get_exif_data(image_path):
    """Extracts EXIF data from an image."""
    try:
        image = Image.open(image_path)
        exif_data = {}

        # Basic image info
        exif_data['File Name'] = os.path.basename(image_path)
        exif_data['Folder Path'] = os.path.dirname(image_path)
        exif_data['Date Created'] = datetime.datetime.fromtimestamp(os.path.getctime(image_path)).isoformat()
        exif_data['Date Modified'] = datetime.datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat()
        exif_data['Size (bytes)'] = os.path.getsize(image_path)
        exif_data['Dimensions'] = image.size
        exif_data['Width'] = image.width
        exif_data['Height'] = image.height
        exif_data['Format'] = image.format
        exif_data['Mode'] = image.mode
        
        # Extract EXIF data
        if hasattr(image, '_getexif'):
            raw_exif = image._getexif()
            if raw_exif is not None:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value

        # Extract GPS info if available
        if 'GPSInfo' in exif_data:
            gps_info = {}
            for key in exif_data['GPSInfo']:
                gps_tag = GPSTAGS.get(key, key)
                gps_info[gps_tag] = exif_data['GPSInfo'][key]
            exif_data['GPSInfo'] = gps_info
        
        # Optional EXIF fields
        exif_data['Camera Maker'] = exif_data.get('Make', 'N/A')
        exif_data['Camera Model'] = exif_data.get('Model', 'N/A')
        exif_data['F-stop'] = exif_data.get('FNumber', 'N/A')
        exif_data['Exposure Time'] = exif_data.get('ExposureTime', 'N/A')
        exif_data['ISO Speed'] = exif_data.get('ISOSpeedRatings', 'N/A')
        exif_data['Exposure Bias'] = exif_data.get('ExposureBiasValue', 'N/A')
        exif_data['Focal Length'] = exif_data.get('FocalLength', 'N/A')
        exif_data['Max Aperture'] = exif_data.get('MaxApertureValue', 'N/A')
        exif_data['Metering Mode'] = exif_data.get('MeteringMode', 'N/A')
        exif_data['Subject Distance'] = exif_data.get('SubjectDistance', 'N/A')
        exif_data['Flash Mode'] = exif_data.get('Flash', 'N/A')
        exif_data['35mm Focal Length'] = exif_data.get('FocalLengthIn35mmFilm', 'N/A')
        exif_data['Contrast'] = exif_data.get('Contrast', 'N/A')
        exif_data['Light Source'] = exif_data.get('LightSource', 'N/A')
        exif_data['Exposure Program'] = exif_data.get('ExposureProgram', 'N/A')
        exif_data['Saturation'] = exif_data.get('Saturation', 'N/A')
        exif_data['Sharpness'] = exif_data.get('Sharpness', 'N/A')
        exif_data['White Balance'] = exif_data.get('WhiteBalance', 'N/A')
        exif_data['Digital Zoom'] = exif_data.get('DigitalZoomRatio', 'N/A')
        exif_data['EXIF Version'] = exif_data.get('ExifVersion', 'N/A')
        
        # Extract GPS coordinates if available
        if 'GPSInfo' in exif_data:
            gps = exif_data['GPSInfo']
            exif_data['Latitude'] = convert_to_decimal_degrees(gps.get('GPSLatitude', 'N/A'))
            exif_data['Longitude'] = convert_to_decimal_degrees(gps.get('GPSLongitude', 'N/A'))
            exif_data['Altitude'] = gps.get('GPSAltitude', 'N/A')
        else:
            exif_data['Latitude'] = 'N/A'
            exif_data['Longitude'] = 'N/A'
            exif_data['Altitude'] = 'N/A'

        return exif_data

    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def write_metadata_to_csv(metadata_list, csv_file_path):
    """Writes the list of metadata dictionaries to a CSV file."""
    if not metadata_list:
        print("No metadata to write to CSV.")
        return

    # Extract keys for the CSV header from the first dictionary
    keys = metadata_list[0].keys()
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadata_list)
    
    print(f"Metadata written to {csv_file_path}")

def process_folder(folder_path):
    """Processes all image files in a folder and extracts metadata for each."""
    if not os.path.isdir(folder_path):
        print(f"Error: The path {folder_path} is not a valid directory.")
        return

    supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    metadata_list = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(folder_path, filename)
            print(f"\nExtracting metadata for {filename}")
            metadata = get_exif_data(image_path)
            if metadata:
                metadata_list.append(metadata)

    # Write all metadata to CSV
    if metadata_list:
        csv_file_path = os.path.join(folder_path, 'image_metadata.csv')
        write_metadata_to_csv(metadata_list, csv_file_path)

