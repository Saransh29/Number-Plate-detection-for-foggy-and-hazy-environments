import image_dehazer										# Load the library
import cv2

HazeImg = cv2.imread('house_input.jpg')						# read input image -- (**must be a color image**)
HazeCorrectedImg = image_dehazer.remove_haze(HazeImg)		# Remove Haze

cv2.imshow('input image', HazeImg);						# display the original hazy image
cv2.imshow('enhanced_image', HazeCorrectedImg);			# display the result
cv2.waitKey(0)											# hold the display window
