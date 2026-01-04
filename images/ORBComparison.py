import cv2
import matplotlib.pyplot as plt

"""
    This script compares two images using ORB (Oriented FAST and Rotated BRIEF)
    feature detection and visualizes the best matching keypoints.
    
    This is useful for visual feature comparison and checking image alignment

    Author :        Martijn Folmer
    Date created :  04-01-2026
"""


def orb_match_and_draw(img1_path, img2_path, max_matches=20, output_path=None):
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

    # Draw matches
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    matched_img = cv2.drawMatches(
        img1_color, kp1,
        img2_color, kp2,
        matches, None,
        # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Display with matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()

    # Save output
    if output_path:
        cv2.imwrite(output_path, matched_img)
        print(f"Result saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    IMGPATH1 = 'images/img1.jpeg'
    IMGPATH2 = 'images/img2.jpeg'
    MAXMATCHES = 50
    OUTPUTPATH = 'matched.png'

    orb_match_and_draw(
        img1_path=IMGPATH1,
        img2_path=IMGPATH2,
        max_matches=MAXMATCHES,
        output_path='../readme_img/ORBComparison.png'
    )
