
import cv2
import numpy as np
from skimage.transform import rotate

### Method 1: Detect rotation using Hough Line Transform (Clicking two points)

def detect_rotation(slice_2d):
    """
    Detects the rotation angle of the spinal cord in a 2D MR slice.

    Parameters:
    - slice_2d (numpy array): A single 2D slice of the spinal cord.

    Returns:
    - angle (float): Estimated rotation angle in degrees.
    """
    # Convert image to 8-bit grayscale (if necessary)
    slice_2d = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply thresholding to enhance contrast (Otsu’s method)
    _, binary = cv2.threshold(slice_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Canny edge detection to find contours of the spinal cord
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Detect straight lines using Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:  # Extract rho and theta from the detected lines
            angle = np.degrees(theta) - 90  # Convert to degrees and shift by 90° for alignment
            angles.append(angle)

        # Compute the median angle for stability
        estimated_angle = np.median(angles)
        return estimated_angle

    else:
        # If no lines are found, return 0 (no rotation detected)
        return 0.0

def click_event(event, x, y, flags, param):
    """Stores clicked points."""
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        if len(points) == 2:

            cv2.destroyAllWindows()


### Method 2: Manual rotation using mouse clicks to draw a line: 

# Mouse callback function
def draw_line(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse press -> Start drawing
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move -> Update the endpoint
        if drawing:
            temp_img = image_resized.copy()
            cv2.line(temp_img, start_point, (x, y), (255, 0, 0), 2)
            cv2.imshow("Draw a Line", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:  # Mouse release -> Store the line
        drawing = False
        end_point = (x, y)
        lines.append((start_point, end_point, z))  # Save start, end, and z-axis
        print(f"Line drawn from {start_point} to {end_point} at Z={z}")
        cv2.destroyAllWindows()



################ Calculate Rotation Angle ################
def compute_rotation_angle_points(points):
    """Compute angle between the commissural line and horizontal axis."""
    dy = points[1, 1] - points[0, 1]
    dx = points[1, 0] - points[0, 0]
    angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    return angle

def compute_rotation_angle_lines(line):
    """Compute angle between the commissural line and horizontal axis."""
    start, end, _ = line
    dy = end[1] - start[1]
    dx = end[0] - start[0]
    angle = -np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    return angle


def rotate_image(image, angle):
    """Rotate image to align the commissural line with the horizontal axis."""
    rotated = rotate(image, -angle, resize=False, mode='edge')
    return rotated
