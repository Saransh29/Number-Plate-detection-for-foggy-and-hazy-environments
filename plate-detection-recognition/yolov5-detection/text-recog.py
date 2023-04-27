import pytesseract
import os
import cv2
import re
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Saransh Bibiyan\AppData\Local\Tesseract-OCR'

src = "E:/btp-8/Number-Plate-detection-for-foggy-and-hazy-environments/plate-detection-recognition/test/exp16/crops"
# find images in the path


# if not detected
# strech the image
def strech(img):

    width = img.shape[1]*1.8 # strech width
    height = img.shape[0] # keep original height
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def recognize_plate(img):
   
    # grayscale region within bounding box
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)

    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # print(sorted_contours)

    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape


        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 15: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)

        
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]

        cv2.imshow("char" , roi)
        cv2.waitKey(0)

        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        try:
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except: 
            text = None
    if plate_num != None:
        print("License Plate #: ", plate_num)
    #cv2.imshow("Character's Segmented", im2)
    #cv2.waitKey(0)
    return plate_num
def main():
    
    # dir = os.getcwd()
    dir = src
    cnt=0

    for img_name in os.listdir(dir):

        if not img_name.endswith('.jpg'):
            continue
        cnt+=1
        if(cnt>2):
            break


        img = cv2.imread(os.path.join(src,img_name))
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        # resize image to three times as large as original for better readability
        resz = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('resz',resz)
        cv2.waitKey(0)

        recognize_plate(resz)

        plate = pytesseract.image_to_string(img, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        # # plate = pytesseract.image_to_string(img, lang='eng')
        print("2: Number plate is:", plate ," ", img_name)
            

        # custom_config = r'--oem 3 --psm 6'
        # pytesseract.image_to_string(img, config=custom_config)



if __name__ == '__main__':
    main()





