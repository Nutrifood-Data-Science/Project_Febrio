import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk

# Function to resize image proportionally based on max width
def resize_image(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_dimensions = (max_width, int(h * ratio))
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        return resized_image
    return image

# Function to align images using SIFT features and homography
def align_images_sift(image, template, maxFeatures=500, keepPercent=0.3, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kpsA, descsA = sift.detectAndCompute(imageGray, None)
    kpsB, descsB = sift.detectAndCompute(templateGray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descsA, descsB)
    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for i, m in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    H, mask = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned

# Function to recolor images to red and blue
def redblue_image(input_image1, input_image2):
    img_1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)

    img_merge_1 = cv2.merge((img_1, img_1, img_1))
    img_merge_2 = cv2.merge((img_2, img_2, img_2))

    red = np.zeros((1, 1, 3), np.uint8)
    red[:] = (0, 0, 255)
    blue = np.zeros((1, 1, 3), np.uint8)
    blue[:] = (255, 0, 0)
    white = np.zeros((1, 1, 3), np.uint8)
    white[:] = (255, 255, 255)

    lut_1 = np.concatenate((blue, white), axis=0)
    lut_2 = np.concatenate((red, white), axis=0)

    lut_1 = cv2.resize(lut_1, (1, 256), interpolation=cv2.INTER_CUBIC)
    lut_2 = cv2.resize(lut_2, (1, 256), interpolation=cv2.INTER_CUBIC)

    result_template = cv2.LUT(img_merge_1, lut_1)
    result_compared = cv2.LUT(img_merge_2, lut_2)

    return result_template, result_compared

# Function to compare two images with red and blue overlay
def product_compare_sift(image_path_1, image_path_2, max_width=1000):
    image = cv2.imread(image_path_1)
    template = cv2.imread(image_path_2)

    aligned = align_images_sift(image, template)

    aligned_resized = resize_image(aligned, max_width)
    template_resized = resize_image(template, max_width)

    red_image, blue_image = redblue_image(template_resized, aligned_resized)

    combined = cv2.addWeighted(red_image, 0.5, blue_image, 0.5, 0)
    final_result = resize_image(combined, max_width)

    return final_result

# Tkinter GUI setup
def load_image_1():
    global image_path_1
    image_path_1 = filedialog.askopenfilename()
    display_image(image_path_1, label_image_1)

def load_image_2():
    global image_path_2
    image_path_2 = filedialog.askopenfilename()
    display_image(image_path_2, label_image_2)

def compare_images():
    result = product_compare_sift(image_path_1, image_path_2, max_width=800)
    display_result(result)

def display_image(image_path, label):
    img = Image.open(image_path)
    img = img.resize((300, 300))  # Resize to fit in the label
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

def display_result(result):
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(result)
    img_tk = ImageTk.PhotoImage(img)
    label_result.config(image=img_tk)
    label_result.image = img_tk

# Initialize Tkinter window
root = tk.Tk()
root.title("Image Comparison Tool")

# Layout components
frame = tk.Frame(root)
frame.pack(pady=10)

btn_load_1 = tk.Button(frame, text="Load Image 1", command=load_image_1)
btn_load_1.grid(row=0, column=0, padx=10)

btn_load_2 = tk.Button(frame, text="Load Image 2", command=load_image_2)
btn_load_2.grid(row=0, column=1, padx=10)

btn_compare = tk.Button(root, text="Compare Images", command=compare_images)
btn_compare.pack(pady=10)

label_image_1 = Label(root)
label_image_1.pack(side="left", padx=10)

label_image_2 = Label(root)
label_image_2.pack(side="right", padx=10)

label_result = Label(root)
label_result.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
