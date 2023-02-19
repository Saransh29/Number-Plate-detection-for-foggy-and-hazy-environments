import cv2
import numpy as np
import matplotlib.pyplot as plt

def addOverlay(natural_img):    
    fog_img = cv2.imread(r'C:\Users\yugan\OneDrive\Desktop\Gog overlay textures\07_PS_overlay_mist.jpg')

    # Resize the fog image to the same size as the natural image
    fog_img = cv2.resize(fog_img, (natural_img.shape[1], natural_img.shape[0]))

    # Adjust the opacity of the fog image
    alpha = 0.5
    fog_img = cv2.addWeighted(natural_img, alpha, fog_img, 1 - alpha, 0)
    return fog_img

def add_haze(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Create a new image with a blue tint
    tinted = np.zeros_like(img)
    tinted[:] = (220, 240, 255)

    # Blend the tinted image with the blurred image
    alpha = 0.4
    blended = cv2.addWeighted(img, 1-alpha, tinted, alpha, 0)
    return blended

def add_haze2(img):
    h, w, _ = img.shape

    # Randomly generate parameters for haze
    A = np.random.uniform(0.5, 0.9)
    phi = np.random.uniform(0.1, 0.3)
    theta = np.random.uniform(0.5, 0.7)
    # Generate a white image with same size as input image
    haze_img = np.ones_like(img) * 255
    # Generate random noise for haze
    noise = np.zeros_like(img)
    cv2.randn(noise, 0, 25)
    # Apply haze to each channel of image
    for i in range(3):
        # Add noise to channel
        noise_channel = noise[:,:,i].astype(np.int16)
        img_channel = img[:,:,i].astype(np.int16)
        img_channel += noise_channel

        # Apply haze to channel
        haze_channel = A * img_channel + (1 - A) * 255
        haze_channel = phi * haze_channel + (1 - phi) * img_channel
        haze_channel = theta * haze_channel + (1 - theta) * 255

        # Normalize channel and convert back to uint8
        haze_channel = (haze_channel - np.min(haze_channel)) / (np.max(haze_channel) - np.min(haze_channel)) * 255
        haze_img[:,:,i] = haze_channel.astype(np.uint8)

    return haze_img

def showImage(img1,img2,img3):
    img1 = plt.imread('image1.jpg')
    img2 = plt.imread('image2.jpg')
    img3 = plt.imread('image3.jpg')
    names = ['Image 1', 'Image 2', 'Image 3']

    # Display the images and their names using matplotlib
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    axs[0].imshow(img1,cv2.COLOR_BGR2RGB)
    axs[0].set_title(names[0])
    axs[1].imshow(img2,cv2.COLOR_BGR2RGB)
    axs[1].set_title(names[1])
    axs[2].imshow(img3,cv2.COLOR_BGR2RGB)
    axs[2].set_title(names[2])
    plt.show()
# Load the natural image and the fog image
natural_img = cv2.imread(r"../dataset/google_images/0a0d1748-48cd-4114-90cb-b5baf0b3cbe4___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_147274518_15141875973_large.jpg")

# fog_img = addOverlay(natural_img)
final_img = add_haze(natural_img)
# final_img2 = add_haze(fog_img)
final_img2 = add_haze2(natural_img)

# Display the result
fig, ax = plt.subplots(nrows = 1,ncols= 3,figsize=(10,5))

# Display the first image in the first subplot
ax[0].imshow(cv2.cvtColor(natural_img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Image 1')

# Display the second image in the second subplot
ax[1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
ax[1].set_title('Image 2')

ax[2].imshow(cv2.cvtColor(final_img2, cv2.COLOR_BGR2RGB))
ax[2].set_title('Image 3')

# Show the figure
plt.show()