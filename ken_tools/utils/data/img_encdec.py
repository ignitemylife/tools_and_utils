import cv2
import base64
import numpy as np

def encodeImg2B64Str(img_name: str, cv2_encode: bool = True) -> str:
    img = cv2.imread(img_name)
    if cv2_encode:
        flgs, buf = cv2.imencode('.png', img)
        if not flgs:
            return ''
    else:
        buf = img

    s = base64.b64encode(buf)
    return s.decode('utf-8')


def decodeB64Str2Img(s: str, cv2_decode=True) -> np.ndarray:
    s = s.encode('utf-8')
    s = base64.b64decode(s)

    if cv2_decode:
        img = cv2.imdecode(np.asarray(bytearray(s)), cv2.IMREAD_COLOR)

        return img


def encodeNpy2B64Str(img: np.ndarray, cv2_encode: bool = True) -> str:
    img_enc = cv2.imencode('.png', img)[1] if cv2_encode else img

    s = base64.b64encode(img_enc)
    return s.decode('utf-8')


def decodeB64Str2Npy(s: str, cv2_decode: bool = True, dtype: np.dtype = np.uint8, dsize=None) -> np.ndarray:
    img_dec = np.frombuffer(base64.b64decode(s.encode('utf-8')), dtype=dtype)
    img = cv2.imdecode(img_dec, cv2.IMREAD_COLOR) if cv2_decode else img_dec
    if dsize is not None:
        img = img.reshape(*dsize)
    return img


if __name__ == "__main__":
    imgname = '/Users/konglingshu/Desktop/test_blip2_images/2.jpg'
    ori_img = cv2.imread(imgname)

    # encode numpy array
    print(f'==== using numpy array ====')
    s = encodeNpy2B64Str(ori_img, True)
    img = decodeB64Str2Npy(s, True, dsize=ori_img.shape)
    print(len(s))
    print(img.shape)
    print(np.allclose(img, ori_img))

    print(f'==== using img name =====')
    s = encodeImg2B64Str(imgname)
    img = decodeB64Str2Img(s)
    print(len(s))
    print(img.shape)
    print(np.allclose(img, ori_img))
