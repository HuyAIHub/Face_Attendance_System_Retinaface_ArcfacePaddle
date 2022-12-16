import os

from module.arcface_paddle import insightface_paddle as face

# path_folders = "datasets/CNTT/"
# path_folder_output = "datasets/"

# data_label = open(path_folder_output + 'label.txt', 'w', encoding='UTF-8')
# for i, path_folder in enumerate(os.listdir(path_folders)):
#     label = path_folder
#     for j, path in enumerate(os.listdir(path_folders + path_folder)):
#         path_img = "./" + path_folder + "/" + path
#         line = path_img + '\t' + label + '\n'
#         data_label.write(line)
# data_label.close()

parser = face.parser()
args = parser.parse_args()
args.build_index = "datasets/index.bin"
args.img_dir = "datasets/CNTT"
args.label = "datasets/label.txt"
args.use_gpu = True
args.enable_mkldnn = False
args.rec_model = "ArcFace" #ArcFace
predictor = face.InsightFace(args)
predictor.build_index()