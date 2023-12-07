import cv2
import numpy as np

def sift_matcher(img1: np.ndarray, img2: np.ndarray, max_matches: int = 10, verbose: bool=False) -> np.ndarray:
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:max_matches]

    if verbose:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        cv2.imshow('Matches', match_img)
        cv2.waitKey()

    # Compute the N x 4 points for each match.
    matches_arr = np.empty((len(matches), 4), dtype=float)
    for i, mat in enumerate(matches):
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        matches_arr[i] = [x1, y1, x2, y2]

    return matches_arr


if __name__ == '__main__':
    img1 = cv2.imread('../images/yuna/Screen Shot 2023-12-02 at 3.48.02 PM.png')
    img2 = cv2.imread('../images/yuna/Screen Shot 2023-12-02 at 3.48.20 PM.png')

    arr_matches = sift_matcher(img1, img2, max_matches=10)
    print(arr_matches)