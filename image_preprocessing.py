import cv2

def image_prep(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (84, 110))
    img_cropped = img_resized[18:102, :]
    return img_cropped


