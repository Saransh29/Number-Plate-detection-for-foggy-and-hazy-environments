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

    cv2.waitKey(0)
    cv2.imwrite("output.jpg", img)
    img_path= "output.jpg"

    client = vision.ImageAnnotatorClient()
    with io.open(img_path, 'rb') as image_file :
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.crop_hints(image=image)

    response = client.object_localization(image=image)
    # print(response)

    lplate = response.localized_object_annotations
    lplate_vector=[]
    for (i) in lplate:
        if(i.name == 'License plate'):
            lplate_vector.append(i.bounding_poly)
    # print(lplate_vector)

    normalized_vertices = [(vertex.x, vertex.y) for vertex in lplate_vector[0].normalized_vertices]
    # print(normalized_vertices)

    height, width = img.shape[:2]

    vertices = []
    for vertex in normalized_vertices:
        vertices.append((int(vertex[0]*width), int(vertex[1]*height)))

    cropped_image = img[vertices[0][1]:vertices[2][1], vertices[0][0]:vertices[2][0]]

    cv2.imwrite("crop.jpg", cropped_image)
    cv2.imshow('cropped',cropped_image)
    cv2.waitKey(0)

    gv_path = "crop.jpg"

    with io.open(gv_path, 'rb') as image_file :
        content = image_file.read()
    
    image = vision.types.Image(content=content)
    response = client.object_localization(image=image)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print(texts)

    for(text) in texts:
        lplate_text = text.description
        # print(text.description)
        break
    print(lplate_text)

    testimage = cv2.rectangle(img, vertices[2], vertices[0], (0, 255, 0), 1)
    # cv2.imshow('done ',testimage)
    # cv2.waitKey(0)


    for text in texts:
            vertices = [(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices]
            cv2.putText(img, lplate_text, (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # cv2.rectangle(img, vertices[0], vertices[2], (0, 255, 0), 3)

            cv2.imshow('overlay_added ',img)
            cv2.waitKey(0)
            break

    cv2.imwrite('op.jpg', img)


path = 'E:/btp-8/Number-Plate-detection-for-foggy-and-hazy-environments/dataset/video_images/car-wbs-MH12FU1014_00000.png'

recognise_license_plate(path)

