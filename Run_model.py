import time
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon
from utils import img_warped_preprocess, plot_one_box
import threading
from load_model import load_model
import threading
import cv2, io, json, psycopg2, torch, time,threading
# from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
# from minio import Minio
# from kafka import KafkaProducer
# from kafka.errors import KafkaError
from glob_var import db_connect, kafka_connect, minio_connect
from module.arcface_paddle import insightface_paddle as face

# Initialize some variables
from sort import Sort
minio_address, minio_address1, bucket_name, client = minio_connect()
topic_event,topic_face,producer = kafka_connect()

face_check = []
name_check = []
id_check = []
parser = face.parser()
args = parser.parse_args()
args.output = "output/"
args.use_gpu = True
args.rec = True
args.enable_mkldnn = False
args.rec_model = "ArcFace" #ArcFace
args.index = "datasets/index.bin"

class RunModel(threading.Thread):
    def __init__(self,cam_dict):
        super().__init__()
        self.cameraID = cam_dict.cameraID
        self.rtsp = cam_dict.streaming_url
        # self.rtsp = "/home/aitraining/workspace/huydq46/Face_Attendance_System/datasets/data_test/video3.avi"
        self.construction_id = int(cam_dict.construction_id)
        self.doStop = False
        self.threadID = int(self.cameraID)
        self.model_retinaface = load_model.retinaface
        self.model_arcface = face.InsightFace(args)
        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        if not threading.Thread.is_alive(self):
                self.start()
        self.count =0
    def run(self):
        print("thread ----- ",self.cameraID)
        try: 
            frame_count = 0
            cap = cv2.VideoCapture(self.rtsp)
            print('self.rtsp:',self.rtsp)
            error_frame = 0
            self.count+=1
            print('self.count',self.count)
            while True:
                
                timer = cv2.getTickCount()
                try:
                    if self.doStop:
                        print('Stop!!!')
                        break

                    ret, frame = cap.read()
                    # frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
                    # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                except:
                    cap = cv2.VideoCapture(self.rtsp)
                    error_frame += 1
                    if error_frame == 5: break
                    continue
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Inference
                    result_boxes, result_scores, result_landmark = self.model_retinaface.infer(frame_rgb)
                    print('=================================================')
                    predict=self.tracker.update(result_boxes)
                    print('predict',predict)
                    boxes_track = predict[:,:-1]
                    boces_ids = predict[:,-1].astype(int)
                    
                    for i, (bbox,landmarks,id_, score) in enumerate(zip(boxes_track, result_landmark,boces_ids, result_scores)):
                        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
                        landmark = np.array([landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8],
                                            landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]])
                        landmark = landmark.reshape((2,5)).T
                        # Align face
                        nimg = img_warped_preprocess(frame_rgb, bbox, landmark, image_size='112,112')
                        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                        # cv2.imshow('check',nimg)
                        # Arcface_paddle
                        labels, np_feature = self.model_arcface.predict(nimg, print_info=True)
                        # Draw face
                        if id_ >= 6:
                            self.tracker.reset_count()
                        plot_one_box(
                            result_boxes[i],
                            landmark,
                            frame,
                            label="{}-{:.2f}".format(labels[0], score),
                            id=id_)
                        # SEND
                        print('labels:',labels)
                        print('score:',score)
                        if len(labels) != 0 and score > 0.98 and id_ not in id_check :
                            print('========================SEND=====================')
                            cv2.imwrite('/home/aitraining/workspace/huydq46/Face_Attendance_System/output/'+labels[0]+'_'+current_time+'.jpg',frame)
                        id_check.append(id_)
                    frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
                    FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                    # print("FPS:", round(FPS))
                    # cv2.putText(frame, 'FPS: ' + str(int(FPS)), (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 25, 255), 2)
                    # cv2.imshow("vid_out", frame)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    time.sleep(0.05)
            cap.release()
            cv2.destroyAllWindows()
        except BaseException as e:
            print("error Run model  -----  ",str(e))
        
    
    def Sql_insert(self, insert_scrip, insert_value):
        c.execute(insert_scrip, insert_value)
        conn.commit()

    def Push_database(self, event_result, cameraID, current_time, path_img):
        Insert_event = 'INSERT INTO vcc_events_management.event(type_id, camera_id, created_at, captured_image_url) VALUES (%s, %s, %s, %s)'
        Insert_event_values = (2, cameraID, current_time, path_img)
        Insert_people_detection_event = "INSERT INTO vcc_events_management.people_detection_event(id,direction, user_id, user_type, is_stranger, is_wear_helmet, is_wear_shirt, is_wear_shoes) VALUES(%s, %s, %s, %s,%s, %s, %s, %s)"
        self.Sql_insert(Insert_event, Insert_event_values)
        get_data = 'SELECT id FROM vcc_events_management.event ORDER by id LIMIT all'
        c.execute(get_data)
        event_id = c.fetchall()
        Insert_people_detection_event_values = (
        event_id[-1], None, None, None, None, event_result[1][0], event_result[1][1], event_result[1][2])
        self.Sql_insert(Insert_people_detection_event, Insert_people_detection_event_values)

# if __name__ == '__main__':
#     runModel = RunModel()
#     runModel.start()
# python -m pip install paddlepaddle-gpu==2.3.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# opencv-python==4.2.0.32
