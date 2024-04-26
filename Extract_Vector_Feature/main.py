import math
import cv2 as cv
import numpy as np
from skimage.feature import hog
import oracledb

connection = oracledb.connect(
    user="pvd",
    password="12345",
    dsn="localhost:1521/oracle"
)
cursor = connection.cursor()

def main():
    paths = ["H:\DAT\HCSDL_DPT\Data\picture3.jpg", "H:\DAT\HCSDL_DPT\Data\picture13.jpg",
             "H:\DAT\HCSDL_DPT\Data\picture16.jpg", "H:\DAT\HCSDL_DPT\Data\picture28.jpg"]
    cluster_id = 0
    for p in paths:
        feature_vector = extract_features(p)
        feature_vector = ';'.join(str(num) for num in feature_vector)
        cursor.execute(f"SELECT image_id.NEXTVAL FROM DUAL")
        image_id = cursor.fetchone()[0]
        with open(p, 'rb') as f:
            image = f.read()
        sql = 'insert into flowers (id, image, cluster_id, feature) values (:id, :image, :cluster_id, :feature)'
        cursor.execute(sql, id=image_id, image=image, cluster_id=cluster_id, feature=feature_vector)
        sql = 'insert into centroids (cluster_id, feature_vector) values (:cluster_id, :feature_vector)'
        cursor.execute(sql, cluster_id=cluster_id, feature_vector=feature_vector)
        cluster_id += 1
        connection.commit()


def extract_features(image_path):
    image = cv.imread(image_path)
    hsv_vector = extract_hsv(image)
    hog_vector = extract_hog(image)
    return np.concatenate((hsv_vector, hog_vector))


def extract_hsv(image):
    h_vector = np.zeros(6, dtype=int)
    s_vector = np.zeros(8, dtype=int)
    v_vector = np.zeros(10, dtype=int)
    width = image.shape[1]
    height = image.shape[0]
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            H, S, V = convert_bgr_to_hsv(R, G, B)

            h_index = min(5, math.floor(H / 60))
            s_index = min(5, math.floor(S / 0.125))
            v_index = min(5, math.floor(V / 0.1))

            h_vector[h_index] += 1
            s_vector[s_index] += 1
            v_vector[v_index] += 1
    return np.concatenate((h_vector, s_vector, v_vector))


def convert_bgr_to_hsv(R, G, B):
    R, G, B = R / 255, G / 255, B / 255
    max_rgb = max(R, G, B)
    min_rgb = min(R, G, B)

    V = max_rgb
    S = 0
    if V != 0:
        S = (V - min_rgb) / V
    H = 0
    if R == G and G == B:
        H = 0
    elif V == R:
        H = 60 * (G - B) / (V - min_rgb)
    elif V == G:
        H = 120 + 60 * (B - R) / (V - min_rgb)
    elif V == B:
        H = 240 + 60 * (R - G) / (V - min_rgb)
    if H < 0:
        H = H + 360
    return [H, S, V]


def extract_hog(image):
    gray_image = convert_bgr_to_gray(image)
    (hog_vector, hog_image) = hog(gray_image,
                                  orientations=9,
                                  pixels_per_cell=(8, 8),
                                  transform_sqrt=True,
                                  cells_per_block=(2, 2),
                                  block_norm="L2",
                                  visualize=True)
    return hog_vector


def convert_bgr_to_gray(image):
    width = image.shape[1]
    height = image.shape[0]
    gray_image = np.zeros((height, width), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            gray_image[j, i] = 0.299 * R + 0.587 * G + 0.114 * B
    return gray_image


if __name__ == "__main__":
    main()
