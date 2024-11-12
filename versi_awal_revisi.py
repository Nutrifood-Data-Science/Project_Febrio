import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile
import shutil
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil  # To dynamically set thread count based on system capacity

# Global variables
output_dir = None
zip_file_path = None
MAX_THREADS = min(4, psutil.cpu_count(logical=False))  # Set threads dynamically based on CPU cores

# Function to align images
def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    H, mask = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    return aligned

# Function to recolor images to red and blue for comparison
def redblue_image(input_image1, input_image2):
    img_1 = cv2.cvtColor(input_image1, cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(input_image2, cv2.COLOR_RGB2GRAY)
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

# Function to compare images and produce overlay
def product_compare(image_path_1, image_path_2):
    image = cv2.imread(image_path_1)
    template = cv2.imread(image_path_2)
    aligned = align_images(image, template)
    temp, align = redblue_image(template, aligned)
    overlay = cv2.addWeighted(temp, 0.5, align, 0.5, 0)
    return overlay

# GUI function to select ZIP file
def select_zip_file():
    global zip_file_path
    zip_file_path = filedialog.askopenfilename(title="Select ZIP Containing Images", filetypes=[("ZIP files", "*.zip")])
    if zip_file_path:
        label_zip_path.config(text=f"ZIP File: {os.path.basename(zip_file_path)}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("temp_images_for_count")
        temp_top_folder = next(os.scandir("temp_images_for_count")).path
        subfolders = [f.path for f in os.scandir(temp_top_folder) if f.is_dir()]
        if len(subfolders) == 2:
            folder_a, _ = subfolders
            total_images = len(find_images_in_directory(folder_a))
            label_total_images.config(text=f"Total images to process: {total_images}")
        shutil.rmtree("temp_images_for_count")

# GUI function to select output directory
def select_output_directory():
    global output_dir
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if output_dir:
        label_output_dir.config(text=f"Output Folder: {output_dir}")

# Helper function to find images in a directory
def find_images_in_directory(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

# Main thread function to handle image comparison
def compare_images_thread():
    global btn_compare  # Access the button to re-enable it after processing
    start_time = time.time()  # Start tracking execution time
    temp_dir = "temp_images"
    folder_name = folder_name_entry.get()
    output_comparison_folder = os.path.join(output_dir, folder_name)
    
    if not os.path.exists(output_comparison_folder):
        os.makedirs(output_comparison_folder)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    top_level_folder = next(os.scandir(temp_dir)).path
    subfolders = [f.path for f in os.scandir(top_level_folder) if f.is_dir()]
    if len(subfolders) != 2:
        messagebox.showerror("Error", "ZIP file must contain exactly two folders with images.")
        shutil.rmtree(temp_dir)
        btn_compare.config(state=tk.NORMAL)
        return

    folder_a, folder_b = subfolders
    comp_images = find_images_in_directory(folder_a)
    ref_images = find_images_in_directory(folder_b)
    
    ref_images_dict = {os.path.basename(f).lower(): f for f in ref_images}
    total_images = len(comp_images)
    label_total_images.config(text=f"Total images to process: {total_images}")

    def process_image(comp_img_path):
        comp_img_name = os.path.basename(comp_img_path).lower()
        matched_ref_path = ref_images_dict.get(comp_img_name)
        if matched_ref_path:
            overlay = product_compare(comp_img_path, matched_ref_path)
            if overlay is not None:
                base_name = os.path.basename(comp_img_path)
                cv2.imwrite(os.path.join(output_comparison_folder, f"overlay_{base_name}"), overlay)
            return True
        return False

    successful_comparisons = 0
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_image, img) for img in comp_images]
        for idx, future in enumerate(as_completed(futures), 1):
            success = future.result()
            if success:
                successful_comparisons += 1
            progress = int((successful_comparisons / total_images) * 100)
            progress_bar['value'] = progress
            progress_label.config(text=f"{progress}% completed")
            root.update_idletasks()

    shutil.rmtree(temp_dir)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    label_time.config(text=f"Execution Time: {elapsed_time:.2f} seconds")
    
    messagebox.showinfo("Finished", f"Comparison completed and saved in {output_comparison_folder}")
    
    btn_compare.config(state=tk.NORMAL)

# GUI function to trigger image comparison
def compare_images():
    if not zip_file_path or not output_dir:
        messagebox.showerror("Error", "Please select a ZIP file and output directory.")
        return
    btn_compare.config(state=tk.DISABLED)
    threading.Thread(target=compare_images_thread).start()

# GUI function to reset form
def reset_form():
    global output_dir, zip_file_path
    output_dir = None
    zip_file_path = None
    folder_name_entry.delete(0, tk.END)
    label_zip_path.config(text="No ZIP file selected")
    label_output_dir.config(text="No output folder selected")
    progress_bar['value'] = 0
    progress_label.config(text="")
    label_time.config(text="")
    label_total_images.config(text="Total images to process: 0")

# GUI Setup
root = tk.Tk()
root.title("Image Comparison Tool")
frame = tk.Frame(root)
frame.pack(pady=10)
btn_select_zip = tk.Button(frame, text="Select ZIP", command=select_zip_file)
btn_select_zip.pack(pady=5)
label_zip_path = tk.Label(frame, text="No ZIP file selected")
label_zip_path.pack(pady=5)
btn_select_output = tk.Button(frame, text="Select Output Folder", command=select_output_directory)
btn_select_output.pack(pady=5)
label_output_dir = tk.Label(frame, text="No output folder selected")
label_output_dir.pack(pady=5)
label_total_images = tk.Label(frame, text="Total images to process: 0")
label_total_images.pack(pady=5)
folder_name_label = tk.Label(frame, text="Result Folder Name:")
folder_name_label.pack(pady=5)
folder_name_entry = tk.Entry(frame)
folder_name_entry.pack(pady=5)
btn_compare = tk.Button(frame, text="Compare Images", command=compare_images)
btn_compare.pack(pady=20)
progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
progress_bar.pack(pady=5)
progress_label = tk.Label(frame, text="0% completed")
progress_label.pack(pady=5)
label_time = tk.Label(frame, text="")
label_time.pack(pady=5)
btn_reset = tk.Button(frame, text="Reset", command=reset_form)
btn_reset.pack(pady=10)
root.mainloop()
