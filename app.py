import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read CSV File
csv_path = 'frag2.csv'

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    print(np_path_XYs)
    
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Plot and Save
def plot_and_save(paths_XYs, filename):
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

path_XYs = read_csv(csv_path)
plot_and_save(path_XYs, 'plot.png')

# Image Processing
img = cv2.imread('plot.png')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
imgGry = cv2.GaussianBlur(imgGry, (5, 5), 0)

# Adjust thresholding parameters
ret, thrash = cv2.threshold(imgGry, 250, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
min_area = 100  # Set a minimum area to filter out noise
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

# Count the number of filtered contours (polylines)
num_polylines = len(filtered_contours)
print(f"Number of polylines detected: {num_polylines}")

# Shape Detection and Labeling (using filtered_contours instead of contours)
for contour in filtered_contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if 0.95 <= aspectRatio < 1.05:
            cv2.putText(img, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
        else:
            cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
    elif len(approx) == 10:
        cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0))

# Display the Image
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()