import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from defisheye import Defisheye
import numpy as np
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# TODO : ?????
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def from_tensor_to_pixels(image_tensor, input_size=448, max_num=12):
    transform = T.ToPILImage()
    image = transform(image_tensor).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image_as_pixels(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def _rotate_image(image_file, image_file_name, angle):
    BASE_FOLDER_STANDARD_IMAGE = "images_input/"
    image = cv2.imread(image_file)
    (h, w) = image.shape[:2]

    # Define the center of the image
    center = (w // 2, h // 2)

    # Specify the rotation angle
    angle = angle  # Replace with your desired angle

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(BASE_FOLDER_STANDARD_IMAGE + image_file_name + ".jpg", rotated)
    return BASE_FOLDER_STANDARD_IMAGE + image_file_name + ".jpg"

def _defisheye_image(image_file, image_file_name, streetview):
    BASE_FOLDER_STANDARD_IMAGE = "images_input/"
    dtype = 'linear'
    format = 'circular'
    fov = streetview
    pfov = 90
    angle = 0
    padding = 1000
    obj = Defisheye(
        image_file,
        dtype=dtype,
        format=format,
        fov=fov,
        pfov=pfov,
        angle=angle,
        pad=padding
    )
    obj.convert(
        outfile=BASE_FOLDER_STANDARD_IMAGE + image_file_name + ".jpg"
    )
    return BASE_FOLDER_STANDARD_IMAGE + image_file_name + ".jpg"

def convert_streetview_to_normal_image(full_image_file_name, image_exif_data):
    width = int(image_exif_data["Width"])
    image_file_name = full_image_file_name.split('/')
    image_file_name = image_file_name[len(image_file_name) - 1]
    if width > 2705:
        full_image_frame = cv2.imread(full_image_file_name)
        xcenter = width // 2
        image_frame_half_first = full_image_frame[:,0:xcenter,:]
        image_frame_half_second = full_image_frame[:,xcenter:5760,:]
        cv2.imwrite("firsthalf.jpg", image_frame_half_first)
        cv2.imwrite("secondhalf.jpg", image_frame_half_second)
        output_file_first_half = _defisheye_image("firsthalf.jpg", image_file_name + "first", 360)
        output_file_first_half = _rotate_image(output_file_first_half, image_file_name + "first", -25)
        output_file_second_half = _defisheye_image("secondhalf.jpg", image_file_name + "second", 360)
        output_file_second_half = _rotate_image(output_file_second_half, image_file_name + "second", 25)
        return [output_file_first_half, output_file_second_half]
    else:
        output_file = _defisheye_image(full_image_file_name, image_file_name, 180)
        output_file = _rotate_image(output_file, image_file_name, 10)
        return [output_file]
