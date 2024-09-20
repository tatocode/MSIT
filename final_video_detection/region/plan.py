import cv2


def plan1(img):
    high, width = img.shape[0], img.shape[1]
    cv2.rectangle(img, (0, 0), (width // 2, high // 2), 1, thickness=-1)
    cv2.rectangle(img, (width // 2, 0), (width, high // 2), 2, thickness=-1)
    cv2.rectangle(img, (0, high // 2), (width // 2, high), 3, thickness=-1)
    cv2.rectangle(img, (width // 2, high // 2), (width, high), 4, thickness=-1)
    return img

def plan2(img):
    high, width = img.shape[0], img.shape[1]
    cv2.rectangle(img, (0, 0), (width // 3, high), 1, thickness=-1)
    cv2.rectangle(img, (width // 3, 0), ((width*2) // 3, high), 2, thickness=-1)
    cv2.rectangle(img, ((width*2) // 3, 0), (width, high), 3, thickness=-1)
    return img