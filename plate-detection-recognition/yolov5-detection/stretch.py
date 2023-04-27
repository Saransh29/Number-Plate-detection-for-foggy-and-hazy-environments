import pytesseract
import os
import cv2
import re
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Saransh Bibiyan\AppData\Local\Tesseract-OCR'

src = "E:/btp-8/img-report"

# find images in the path


# if not detected
# strech the image
def strech(img):

    width = img.shape[1]*1.8 # strech width
    height = img.shape[0] # keep original height
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, None, fx = 4, fy = 2, interpolation = cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV, 21, 9)
    
    # th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)

    blur = cv2.medianBlur(thr, 3)

    roi = cv2.bitwise_not(blur)

    roi = cv2.medianBlur(roi, 5)

    cv2.imshow('roi',roi)
    cv2.waitKey(0)

    cv2.imwrite(r'E:/btp-8/img-report/baleno_crop_out_test.jpg',thr)

    return resized





def main():
    
    # dir = os.getcwd()
    dir = src
    cnt=0

    img1 = cv2.imread(r'E:/btp-8/img-report/full_process/baleno_crop_GAN.jpg')
    img2 = strech(img1)


    # for img_name in os.listdir(dir):
    
    #     if not img_name.endswith('.jpg'):
    #         continue
    #     cnt+=1
    #     # if(cnt>2):
    #     #     break
 
        
    #     img = cv2.imread(os.path.join(src,img_name))
    #     # cv2.imshow('img',img)
    #     # cv2.waitKey(0)

    #     # resize image to three times as large as original for better readability
    #     resz = cv2.resize(img, None, fx = 4, fy = 2, interpolation = cv2.INTER_CUBIC)
    #     cv2.imshow('resz',resz)
    #     cv2.waitKey(0)

    #     # recognize_plate(resz)

    #     # plate = pytesseract.image_to_string(img, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    #     # # # plate = pytesseract.image_to_string(img, lang='eng')
    #     # print("2: Number plate is:", plate ," ", img_name)
           

    #     # custom_config = r'--oem 3 --psm 6'
    #     # pytesseract.image_to_string(img, config=custom_config)



if __name__ == '__main__':
    main()





