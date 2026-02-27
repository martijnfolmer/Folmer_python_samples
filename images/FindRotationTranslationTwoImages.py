import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
"""
    Using ORB, we can use the matched features between two images to calculate the rotation and 
    translation (horizontal/vertical shifts) that the image underwent. 
    
    Example usage : a camera is hanging somewhere, and something knocks it out of alignment. You can use
    a saved image and a current image from the camera, compare the features and check if it has rotated
    or shifted horizontally/vertically.
    
    Author :        Martijn Folmer
    Date created :  24-02-2026
"""


def _circular_mean_angle(angles_deg):
    """Calculate circular mean of angles in degrees, handling wraparound."""
    angles_rad = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    mean_angle_rad = math.atan2(mean_sin, mean_cos)
    return math.degrees(mean_angle_rad)


def _remove_outliers_iqr(data, factor=1.5):
    """Remove outliers using interquartile range method."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    mask = (data >= lower_bound) & (data <= upper_bound)
    return mask


def orb_match_and_draw(img1_path, img2_path, max_matches=20, output_path=None, show_image=True):
    """
    Match features between two images using ORB and calculate rotation and translation.
    
    Parameters:
    -----------
    img1_path : str
        Path to the first (reference) image
    img2_path : str
        Path to the second (current) image
    max_matches : int
        Maximum number of matches to use for calculation (default: 20)
    output_path : str, optional
        Path to save the visualization image
    show_image : bool
        Whether to display the matched image visualization (default: True)
    
    Returns:
    --------
    dict : Dictionary containing:
        - rotation_angle: Rotation angle in degrees (positive = counterclockwise)
        - translation_x: Horizontal shift in pixels (positive = right)
        - translation_y: Vertical shift in pixels (positive = down)
        - num_matches: Number of matches used for calculation
    """
    # Load images in grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the image paths is invalid.")

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints & compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors with Hamming distance and cross-check
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by ascending distance (best matches first)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]

    if len(matches) < 2:
        raise ValueError(f"Not enough matches found ({len(matches)}). Need at least 2 matches.")

    # grayscale images have shape (height, width)
    height1, width1 = img1.shape
    height2, width2 = img2.shape
    centerImg1 = (width1 / 2, height1 / 2)
    centerImg2 = (width2 / 2, height2 / 2)

    # Calculate rotation angles and translation vectors
    angles = []
    translations = []
    
    for match in matches:
        pt1 = kp1[match.queryIdx].pt  # (x, y) coordinates from img1
        pt2 = kp2[match.trainIdx].pt  # (x, y) coordinates from img2
        
        # Calculate angle from center for each point
        angle1 = math.degrees(math.atan2(pt1[1] - centerImg1[1], pt1[0] - centerImg1[0]))
        angle2 = math.degrees(math.atan2(pt2[1] - centerImg2[1], pt2[0] - centerImg2[0]))
        
        # Calculate angle difference (rotation)
        angleDiff = angle1 - angle2

        # Normalize to [-180, 180] range
        while angleDiff > 180:
            angleDiff -= 360
        while angleDiff < -180:
            angleDiff += 360
        angles.append(angleDiff)
        
        # Calculate translation vector (dx, dy)
        dx = pt2[0] - pt1[0]  # horizontal shift
        dy = pt2[1] - pt1[1]  # vertical shift
        translations.append((dx, dy))

    # Remove outliers from angles using IQR method
    angles_array = np.array(angles)
    angle_mask = _remove_outliers_iqr(angles_array)
    filtered_angles = angles_array[angle_mask]
    
    if len(filtered_angles) == 0:
        # Fallback to all angles if filtering removes everything
        filtered_angles = angles_array
    
    # Calculate circular mean of angles (handles wraparound)
    rotation_angle = _circular_mean_angle(filtered_angles)

    # Remove outliers from translations using IQR on both x and y separately
    translations_array = np.array(translations)
    tx_mask = _remove_outliers_iqr(translations_array[:, 0])
    ty_mask = _remove_outliers_iqr(translations_array[:, 1])
    translation_mask = tx_mask & ty_mask
    
    if np.sum(translation_mask) == 0:
        # Fallback to all translations if filtering removes everything
        translation_mask = np.ones(len(translations), dtype=bool)
    
    filtered_translations = translations_array[translation_mask]
    
    # Calculate median translation
    translation_x = np.median(filtered_translations[:, 0])
    translation_y = np.median(filtered_translations[:, 1])

    # Print results
    print(f"Rotation angle: {rotation_angle:.2f} degrees")
    print(f"Translation: ({translation_x:.2f}, {translation_y:.2f}) pixels (x, y)")
    print(f"Number of matches used: {len(matches)}")

    # Draw matches
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    matched_img = cv2.drawMatches(
        img1_color, kp1,
        img2_color, kp2,
        matches, None,
        # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Get image names from paths
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    # drawMatches creates a side-by-side image, so img2 starts at width1
    matched_height, matched_width = matched_img.shape[:2]
    img1_width = width1
    img2_start_x = img1_width
    
    # Draw border/outline between the two images
    border_color = (255, 255, 0)  # Yellow border
    border_thickness = 3

    # Draw vertical line separating the two images
    cv2.line(matched_img, (img1_width, 0), (img1_width, matched_height), 
             border_color, border_thickness)
    
    # Add image names in top left of each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Increased from 0.7
    thickness = 3  # Increased from 2
    text_color = (0, 255, 0)  # Green
    bg_color = (0, 0, 0)  # Black background
    
    # Get text size for background rectangle
    (text_width1, text_height1), _ = cv2.getTextSize(img1_name, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(img2_name, font, font_scale, thickness)
    
    # Draw background rectangles for text
    padding = 10
    cv2.rectangle(matched_img, (padding, padding), 
                  (padding + text_width1 + 15, padding + text_height1 + 15), bg_color, -1)
    cv2.rectangle(matched_img, (img2_start_x + padding, padding), 
                  (img2_start_x + padding + text_width2 + 15, padding + text_height2 + 15), bg_color, -1)
    
    # Draw image names
    cv2.putText(matched_img, img1_name, (padding + 7, padding + text_height1 + 7), 
                font, font_scale, text_color, thickness)
    cv2.putText(matched_img, img2_name, (img2_start_x + padding + 7, padding + text_height2 + 7), 
                font, font_scale, text_color, thickness)
    
    # Add rotation and translation information text
    info_text = [
        f"Rotation: {rotation_angle:.2f} deg",
        f"Translation: ({translation_x:.2f}, {translation_y:.2f}) px"
    ]
    
    # Position info text in bottom right
    info_font_scale = 1.0  # Increased from 0.6
    info_thickness = 2  # Increased from 1
    line_height = 35  # Increased from 25
    y_start = matched_height - len(info_text) * line_height - 15
    
    for i, text in enumerate(info_text):
        (text_w, text_h), _ = cv2.getTextSize(text, font, info_font_scale, info_thickness)
        x_pos = matched_width - text_w - 15
        y_pos = y_start + i * line_height + text_h
        
        # Draw background rectangle
        cv2.rectangle(matched_img, (x_pos - 8, y_pos - text_h - 8), 
                      (x_pos + text_w + 8, y_pos + 8), bg_color, -1)
        # Draw text
        cv2.putText(matched_img, text, (x_pos, y_pos), 
                    font, info_font_scale, text_color, info_thickness)

    # Save output
    if output_path:
        cv2.imwrite(output_path, matched_img)
        print(f"Result saved to {output_path}")

    # Display with matplotlib if requested
    if show_image:
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Return transformation data
    return {
        'rotation_angle': rotation_angle,
        'translation_x': translation_x,
        'translation_y': translation_y,
        'num_matches': len(matches)
    }



if __name__ == "__main__":
    IMGPATH1 = 'imagePath1'
    IMGPATH2 = 'imagePath2'
    MAXMATCHES = 50
    OUTPUTPATH = 'matched.png'

    orb_match_and_draw(
        img1_path=IMGPATH1,
        img2_path=IMGPATH2,
        max_matches=MAXMATCHES,
        output_path=OUTPUTPATH
    )
