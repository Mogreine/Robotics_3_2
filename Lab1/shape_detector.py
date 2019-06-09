import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect_cylinder(self, contour):
        shape = 'undefined'
        per = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * per, True)

        # looking at vertices number, if 4 - it's either a rectangle or a cylinder
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ratio = w / h

            if 0.95 <= ratio <= 1.05:
                return False
            else:
                return True
        else:
            return False

