import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to resize image to fit screen
def resize_to_screen(image, max_width=800, max_height=600):
    h, w = image.shape[:2]
    # Calculate the aspect ratio and resize proportionally
    if w > max_width or h > max_height:
        aspect_ratio = min(max_width / float(w), max_height / float(h))
        new_size = (int(w * aspect_ratio), int(h * aspect_ratio))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    return image

# Function to align the compared image to the template image as a reference
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # Convert images to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Use ORB to detect keypoints and extract local invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # Match the features using BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descsA, descsB, None)

    # Sort the matches by their distance (the smaller the distance, the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # Optionally visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = resize_to_screen(matchedVis, 1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # Allocate memory for the keypoints (x, y)-coordinates from the top matches
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # Loop over the top matches and store the coordinates
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # Compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # Use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # Return the aligned image
    return aligned

# Function to recolor images to red and blue
def redblue_image(input_image1, input_image2):
    # Convert both images to grayscale
    img_1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)

    # Merge the grayscale images to 3-channel
    img_merge_1 = cv2.merge((img_1, img_1, img_1))
    img_merge_2 = cv2.merge((img_2, img_2, img_2))

    # Allocate memory for red, blue and white colors
    red = np.zeros((1, 1, 3), np.uint8)
    red[:] = (0, 0, 255)
    blue = np.zeros((1, 1, 3), np.uint8)
    blue[:] = (255, 0, 0)
    white = np.zeros((1, 1, 3), np.uint8)
    white[:] = (255, 255, 255)

    # Create LUT (Look-Up Table) for red and blue coloring
    lut_1 = np.concatenate((blue, white), axis=0)
    lut_2 = np.concatenate((red, white), axis=0)

    lut_1 = cv2.resize(lut_1, (1, 256), interpolation=cv2.INTER_CUBIC)
    lut_2 = cv2.resize(lut_2, (1, 256), interpolation=cv2.INTER_CUBIC)

    # Apply LUT to both images
    result_template = cv2.LUT(img_merge_1, lut_1)
    result_compared = cv2.LUT(img_merge_2, lut_2)

    return result_template, result_compared

# Function to compare two images by overlaying each other
def product_compare1(image_path_1, image_path_2):
    # Read the input and template image
    image = cv2.imread(image_path_1)
    template = cv2.imread(image_path_2)

    # Align the compared image to the template
    aligned = align_images(image, template)

    # Apply red and blue coloring
    temp, align = redblue_image(template, aligned)

    # Overlay two images with opacity of 0.5 each
    overlay = temp.copy()
    output = align.copy()
    dst = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

    # Resize the final result to fit the screen
    final_result = resize_to_screen(dst, max_width=800, max_height=600)

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
    result = product_compare1(image_path_1, image_path_2)
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

label_image_1 = tk.Label(root)
label_image_1.pack(side="left", padx=10)

label_image_2 = tk.Label(root)
label_image_2.pack(side="right", padx=10)

label_result = tk.Label(root)
label_result.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
