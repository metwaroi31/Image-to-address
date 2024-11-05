from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

def read_optical_character(img_path):
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`, 'vi'
    # to switch the language model in order
    ocr = PaddleOCR(use_angle_cls=True, lang='vi') # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw result
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return boxes, txts
