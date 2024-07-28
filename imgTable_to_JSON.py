import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from super_image import EdsrModel, ImageLoader
from PIL import Image, ImageOps
import cv2
import pytesseract
import os
import pandas as pd
import time
import re



#Set Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Change this to where your tesseract .exe is located. Make sure to download tesseract OCR if not there.
# Load the image
image = cv2.imread('tables/table2.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding with adjusted parameters
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

# Detect horizontal lines with adjusted structuring element size
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

# Detect vertical lines with adjusted structuring element size
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

# Combine horizontal and vertical lines
table_lines = cv2.add(horizontal_lines, vertical_lines)

# Find contours with adjusted retrieval mode and approximation method
contours, hierarchy = cv2.findContours(table_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours (optional, for visualization)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Display the image with contours
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.axis('off')  # This line removes the x-axis and y-axis
plt.show()

# Directory where cropped images will be saved
output_dir = 'table_cells'

# Check if the directory exists
if os.path.exists(output_dir):
    # Remove the directory and all its contents
    shutil.rmtree(output_dir)

# Create the directory
os.makedirs(output_dir)

# Extract and save individual cells along with their coordinates
cell_details = []
for i, contour in enumerate(contours):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Optional: Filter out too small or too large contours
    if w > 10 and h > 10:  # Adjust these values based on your specific needs
        # Crop the image to the bounding box size
        cell_image = image[y:y+h, x:x+w]
        
        # Save the cropped image with a unique filename in the specified directory
        image_filename = os.path.join(output_dir, f'cell_{i}.jpg')
        cv2.imwrite(image_filename, cell_image)

        # Define the filename for the coordinates
        coords_filename = os.path.join(output_dir, f'cell_{i}_coords.txt')
        
        # Write the coordinates to the .txt file
        with open(coords_filename, 'w') as f:
            f.write(f'x: {x}, y: {y}, w: {w}, h: {h}')
        
        # Add the details to the cell_details list
        cell_details.append({
            'image_path': image_filename,
            'coords_file': coords_filename,
            'x': x, 'y': y, 'w': w, 'h': h
        })

#Uncomment this if you want to upscale your images for 'possibly' better OCR______________________________________________
#Upscaling images

# for image in os.listdir(output_dir):
#     print(image)
#     if image.endswith('.jpg'):
#         image_path = f"{output_dir}/{image}"
#         image_loaded = Image.open(image_path)

#         # Calculate border size as a percentage of the image's dimensions
#         border_percentage = 0.05  # 5% border
#         border_width = int(image_loaded.width * border_percentage)
#         border_height = int(image_loaded.height * border_percentage)
#         bordered_image = ImageOps.expand(image_loaded, border=(border_width, border_height), fill='white')

#         model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
#         inputs = ImageLoader.load_image(bordered_image)
#         preds = model(inputs)

#         # Save the image with the border
#         ImageLoader.save_image(preds, f"{output_dir}/{image}")

#     else:
#         continue

#_____________________________________________________________________________________________________________
print(cell_details)

x_positions = sorted(set([int(cell['x']) for cell in cell_details]))
y_positions = sorted(set([int(cell['y']) for cell in cell_details]))

df = pd.DataFrame(index=range(len(y_positions)), columns=range(len(x_positions)))

for cell in cell_details[:-1]:  # Exclude the last cell
    col = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - int(cell['x'])))
    row = min(range(len(y_positions)), key=lambda i: abs(y_positions[i] - int(cell['y'])))
    df.iloc[row, col] = cell['image_path']

print(df)




def clean_text(text):
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

for index, row in df.iterrows():
    for col in df.columns:
        image_path = row[col]
        if pd.isna(image_path):  # Check if the path is NaN
            continue  # Skip this iteration if the path is NaN
        try:
            img = cv2.imread(image_path)
            # img = cv2.resize(img, None, fx=2, fy=2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                text = pytesseract.image_to_string(img, config='--psm 11')
                # Clean the OCR text
                cleaned_text = clean_text(text)
                df.at[index, col] = cleaned_text
            else:
                print(f"Image not found at {image_path}, skipping...")
        except Exception as e:
            print(f"Error processing image at {image_path}: {e}")

# Remove the first two rows
df = df.iloc[1:]

# Remove the first column
df = df.iloc[:, 1:]
df.to_csv('output.csv', index=False, encoding="utf-8")
df.to_json('output.json')
