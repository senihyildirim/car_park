import cv2
import pickle
import cvzone
import numpy as np

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

# Load parking space positions from a file
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Dimensions of the parking space
width, height = 107, 48


def checkParkingSpace(imgPro):
    # Counter to keep track of free parking spaces
    spaceCounter = 0

    # Iterate through each parking space position
    for pos in posList:
        x, y = pos

        # Extract the region of interest (ROI) for the parking space
        imgCrop = imgPro[y:y + height, x:x + width]
        # Count the number of non-zero pixels in the ROI
        count = cv2.countNonZero(imgCrop)

        # Define rectangle color and thickness based on the count of non-zero pixels
        if count < 900:
            color = (0, 255, 0)  # Green color for a free space
            thickness = 3
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red color for an occupied space
            thickness = 2

        # Draw rectangle around the parking space on the original image
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        # Display the count of non-zero pixels inside the parking space
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    # Display the total number of free parking spaces
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))


while True:
    # Check if the video has reached the end, then reset to the beginning
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read a frame from the video
    success, img = cap.read()

    # Convert the frame to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    # Apply adaptive thresholding to create a binary image
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)

    # Apply median blur to the binary image
    imgMedian = cv2.medianBlur(imgThreshold, 5)

    # Define a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)

    # Dilate the image to fill gaps in the parking spaces
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Call the function to check parking spaces and display information
    checkParkingSpace(imgDilate)

    # Display the processed image
    cv2.imshow("Image", img)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break
