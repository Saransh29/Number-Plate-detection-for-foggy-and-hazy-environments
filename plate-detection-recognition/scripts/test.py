import io
import os
import cv2
from google.cloud import vision_v1p3beta1 as vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vnpr-gvision-67e0d187a3c6.json'

def recognise_license_plate(img_path):
    
    og_path = img_path
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img = cv2.resize(img,(800, int((height*800)/width)))

    # cv2.imshow('Original image',img)
    # cv2.waitKey(0)

    cv2.imwrite("output.jpg", img)
    img_path= "output.jpg"

    normalized_vertices = [(0.32996445894241333, 0.522243320941925), (0.724198579788208, 0.522243320941925), (0.724198579788208, 0.6761650443077087), (0.32996445894241333, 0.6761650443077087)]
    height, width = img.shape[:2]

    vertices = []
    for vertex in normalized_vertices:
        vertices.append((int(vertex[0]*width), int(vertex[1]*height)))

    cropped_image = img[vertices[0][1]:vertices[2][1], vertices[0][0]:vertices[2][0]]

    # height, width = cropped_image.shape[:2]
    # print(height, width)
    cv2.imwrite("crop.jpg", cropped_image)
    cv2.imshow('cropped',cropped_image)
    cv2.waitKey(0)

    gv_path = "crop.jpg"

    client = vision.ImageAnnotatorClient()
    with io.open(gv_path, 'rb') as image_file :
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print(texts)

    for(text) in texts:
        print(text.description)
        break
    


   

path = './data/temp/mahindra.jpeg'

recognise_license_plate(path)

