import cv2
import os

VIDEO_PATH = "buffer"
DATA_SET_PATH = "DataSetFromVideo"
IMAGE_WORD = "A"
IMAGE_PATH = "word A"
IMAGE_SIZE = 900

Y = 200
X = 50

frame = 0
i = 655
directory = os.listdir(VIDEO_PATH)
os.makedirs(f"./{DATA_SET_PATH}/{IMAGE_PATH}")

for video in directory:
    capture = cv2.VideoCapture(f'./{VIDEO_PATH}/{video}')
    while (True):
        success, frame = capture.read()
        # print(success, video)
        if success:
            cv2.imshow(IMAGE_WORD, frame[Y: Y + IMAGE_SIZE])
            cv2.waitKey(10)
            a = cv2.imwrite(f'./{DATA_SET_PATH}/{IMAGE_PATH}/{IMAGE_WORD}{i}.jpg', frame[Y: Y + IMAGE_SIZE])
            print(a)

            i += 1
        else:
            break
    capture.release()
