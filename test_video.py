import time
from datetime import datetime
import numpy as np
import cv2
from shapely.geometry import Polygon
from utils import img_warped_preprocess, plot_one_box, plot_one_box1
import threading
from load_model import load_model
import threading
import kafka
from minio import Minio
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import collections
from PIL import Image, ImageDraw, ImageFont

class RunModel(threading.Thread):
    def __init__(self):
        super().__init__()
        # self.rtsp = "rtsp://vcc_cam:Vcc12345678@172.18.5.143:554/stream1"
        self.rtsp = "/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/data_test/video3.avi"
        # self.rtsp = '/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/videos_input/20221101_080000_test1.mp4'
        self.model_retinaface = load_model.retinaface
        self.model_arcface = load_model.arcface
    
    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)
        while True:
            
            try:
                a = time.time()
                timer = cv2.getTickCount()
                ret, frame = cap.read()
                # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                if not ret:
                    cap = cv2.VideoCapture(self.rtsp)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Inference
                result_boxes, result_scores, result_landmark = self.model_retinaface.infer(frame_rgb)
                bboxx,scoress,labels = self.test(frame_rgb,result_boxes, result_scores, result_landmark)
                    # Arcface_paddle
                print('bboxx {},scoress {},labels {}:'.format(bboxx,scoress,labels))
                print('check lenght:',len(bboxx))
                if len(bboxx) != 0:
                    plot_one_box1(frame,bboxx,scoress,labels)
                
                b = time.time() - a
                # print("Total time: {}s".format(b))
                FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                print("FPS:", round(FPS))
                cv2.putText(frame, 'FPS: ' + str(int(FPS)), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 70), 2)
                frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
                cv2.imshow("vid_out", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                frame_count += 1
            except Exception as error:
                print("Error:",error)
                self.model_retinaface.destroy()
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(1)

    def test(self,frame_rgb,result_boxes, result_scores, result_landmark):
        bboxx = []
        scoress = []
        labels = []
        for i in range(len(result_boxes)):
            bbox = np.array([result_boxes[i][0], result_boxes[i][1], result_boxes[i][2], result_boxes[i][3]])
            landmarks = result_landmark[i]
            landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
            landmark = landmark.reshape((2,5)).T
            scores = result_scores[i]
            nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
            labels, np_feature = self.model_arcface.predict(nimg, print_info=True)
            print('labels:',labels)
            bboxx.append(bbox)
            scoress.append(scores)
            labels.append(labels[0])
        return bboxx,scoress,labels
    def draw(self, img, box_list, labels):
        self.color_map.update(labels)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

        for i, dt in enumerate(box_list):
            bbox, score = dt[2:], dt[1]
            label = labels[i]
            color = tuple(self.color_map[label])

            xmin, ymin, xmax, ymax = bbox

            font_size = max(int((xmax - xmin) // 6), 10)
            font = ImageFont.truetype(self.font_path, font_size)

            text = "{} {:.4f}".format(label, score)
            th = sum(font.getmetrics())
            tw = font.getsize(text)[0]
            start_y = max(0, ymin - th)

            draw.rectangle(
                [(xmin, start_y), (xmin + tw + 1, start_y + th)], fill=color)
            draw.text(
                (xmin + 1, start_y),
                text,
                fill=(255, 255, 255),
                font=font,
                anchor="la")
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)], width=2, outline=color)
if __name__ == '__main__':
    runModel = RunModel()
    runModel.start()
# python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# opencv-python==4.2.0.32