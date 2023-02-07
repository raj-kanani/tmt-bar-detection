# import cv2
# import numpy as np
# import pdb;
# pdb.set_trace()
# # Load the image
# img = cv2.imread('2.jpg')
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian Blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
# # Detect edges using Canny Edge Detection
# edges = cv2.Canny(blurred, 50, 80)
#
# # Find contours in the edge map
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Find the contour with the largest area
# max_area = 0
# best_cnt = None
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > max_area:
#         max_area = area
#         best_cnt = cnt
#
# # Draw the contour on the original image
# cv2.drawContours(img, [best_cnt], 0, (0, 255, 0), 1)
#
# # Display the resulting image
# cv2.imshow('Detected Steel Bar', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#
# import cv2
# import numpy as np
#
# # Load the image
# img = cv2.imread('2.jpg')
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian Blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
# # Detect edges using Canny Edge Detection
# edges = cv2.Canny(blurred, 100, 200)
#
# # Find contours in the edge map
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# circles = cv2.HoughCircles(img,
#                            cv2.HOUGH_GRADIENT,
#                            minDist=8,
#                            dp=1)
# # Count the number of contours
# num_rebars = len(contours)
# num_rebar = len(circles)
# # Display the result
# print('Number of Rebars:', num_rebars)
# cv2.imshow('edges', edges)
# # cv2.imshow('edges', circles)
# # cv2.imshow('contours', contours)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# k = cv2.waitKey(0) & 0xFF
# print(k)
# if k == 27:
#     cv2.destroyAllWindows()


#
# ############# custom object detection
# import tkinter as tk
# import cv2
# from PIL import Image
# from PIL import ImageTk
# from tkinter import SUNKEN, filedialog
#
#
# def select_image():
#     global panelA, panelB, entry
#     path = filedialog.askopenfilename()
#     image = cv2.imread(path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (11, 11), 0)
#     canny = cv2.Canny(blur, 30, 100)
#     dilated = cv2.dilate(canny, (1, 1), iterations=0)
#     cnt, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     rgb2 = cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
#
#     image = Image.fromarray(image)
#     image = ImageTk.PhotoImage(image)
#
#     rgb2 = Image.fromarray(rgb2)
#     rgb2 = ImageTk.PhotoImage(rgb2)
#
#     panelA = tk.Label(image=image)
#     panelA.image = image
#     panelA.grid(row=0, column=0)
#
#     panelB = tk.Label(image=rgb2)
#     panelB.image = rgb2
#     panelB.grid(row=0, column=1)
#
#     entry = tk.Entry(width=30)
#     entry.grid(row=1, column=0, columnspan=2)
#     entry.insert(0, 'Number of Objects: ')
#     entry.delete(20, tk.END)
#     entry.insert(20, len(cnt))
#
#
# window = tk.Tk()
# panelA = None
# panelB = None
# entry = None
# button = tk.Button(text="select image", width=40, relief=SUNKEN, command=select_image)
# button.grid(row=2, column=0, columnspan=2, pady=10)
#
# window.mainloop()


########### detect circle on image
# import tkinter as tk
# import cv2
# from PIL import Image
# from PIL import ImageTk
# from tkinter import SUNKEN, filedialog
#
#
# def select_image():
#     global panelA, panelB, entry
#     path = filedialog.askopenfilename()
#     image = cv2.imread(path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (11, 11), 0)
#     canny = cv2.Canny(blur, 30, 100)
#     dilated = cv2.dilate(canny, (1, 1), iterations=0)
#     cnt, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     rgb2 = cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
#
#     image = Image.fromarray(image)
#     image = ImageTk.PhotoImage(image)
#
#     rgb2 = Image.fromarray(rgb2)
#     rgb2 = ImageTk.PhotoImage(rgb2)
#
#     panelA = tk.Label(image=image)
#     panelA.image = image
#     panelA.grid(row=0, column=0)
#
#     panelB = tk.Label(image=rgb2)
#     panelB.image = rgb2
#     panelB.grid(row=0, column=1)
#
#     entry = tk.Entry(width=30)
#     entry.grid(row=1, column=0, columnspan=2)
#     entry.insert(0, 'Number of Objects: ')
#     entry.delete(20, tk.END)
#     entry.insert(20, len(cnt))
#
#
# window = tk.Tk()
# panelA = None
# panelB = None
# entry = None
# button = tk.Button(text="select image", width=80, relief=SUNKEN, command=select_image)
# button.grid(row=2, column=0, columnspan=2, pady=10)
#
# window.mainloop()


########################### digit recognisation ##############################3

import tensorflow as tf
import cv2
import numpy as np
# Load the trained model


# Load the image
img = cv2.imread("font.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to get a binary image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for cnt in contours:
    # Get a bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(cnt)

    # Check if the width and height are within a specific range
    if w >= 50 and w <= 60 and h >= 80 and h <= 90:
        # Crop and resize the image to a standard size (e.g. 28x28)
        roi = thresh[y:y + h, x:x + w]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        model = tf.keras.models.load_model("cnn-mnist-model.h5")


        def recognize_digit(roi):
            # Pre-process the ROI
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Use the trained model to make a prediction
            prediction = model.predict(roi)

            # Return the predicted digit
            return np.argmax(prediction)


        # Pass the image through a neural network to recognize the digit
        digit = recognize_digit(roi)

        # Display the digit
        cv2.putText(img, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
