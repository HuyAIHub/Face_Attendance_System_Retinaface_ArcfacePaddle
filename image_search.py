from ast import walk
import os
from module.arcface_paddle import insightface_paddle as face
# from imutils import paths
import pickle
import cv2
import os

parser = face.parser()
args = parser.parse_args()
args.rec = True
args.index = "./datasets/index.bin"
# Initialize the faces embedder
embedding_model = face.InsightFace(args)


files = []

for r , d , f in os.walk("./datasets/CNTT"):
    for file in f:
        if ('.jpg' in file):
            exact_path = r + file
            files.append(exact_path)

print(files)

#facial embedding
representations = []

for img_path in files:
    img = cv2.imread(img_path)
    box_list,face_embedding = embedding_model.predict_np_img(img)