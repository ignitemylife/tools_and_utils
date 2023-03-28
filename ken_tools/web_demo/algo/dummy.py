import cv2
import pdb

class Dummy():
    img_posfix = ['jpg', 'png', 'jpeg']
    def __call__(self, x):
        if x.split('.')[-1] in self.img_posfix:
            img = cv2.imread(x)
            return img
        return [333, 444]