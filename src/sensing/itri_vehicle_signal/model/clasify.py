import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import csv
import argparse


"""These are the main training settings. Set each before running
this file."""
parser = argparse.ArgumentParser(description="")
parser.add_argument("--seq_length", type=int, default=20,
                    help="the length of a sequence")
parser.add_argument("--class_limit", type=int, default=4,
                    help="how much classes need to clasify")
parser.add_argument("--saved_model", type=str, default="data/checkpoints/lstm-images.039-0.014.hdf5",
                    help="the path of model")
parser.add_argument("--video_file", type=str, default="utils/turnright.mp4",
                    help="the path of video that need to clasify")
args = parser.parse_args()


def save_result_as_csv(result):
    # The list of column names as mentioned in the CSV file
    headersCSV = ['video_name','flasher','no_signal','turn_left','turn_right']
    with open('ncu_dataset_clasify_result_0327.csv', 'a', newline='') as f_object:
        dictwriter_object = csv.DictWriter(f_object, fieldnames=headersCSV)
        for i in result :
            dic = {'video_name':i[0],'flasher':i[1][0][0],'no_signal':i[1][0][1],'turn_left':i[1][0][2],'turn_right':i[1][0][3]}
            dictwriter_object.writerow(dic)
        f_object.close()
    # with open('ncu_dataset_clasify_result_0317.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in result:
    #         print(i[1][0])
    #         writer.writerow(i)

seq_length = args.seq_length
class_limit = args.class_limit
saved_model = args.saved_model
video_file = args.video_file

# if (len(sys.argv) == 5):
#     seq_length = int(sys.argv[1])
#     class_limit = int(sys.argv[2])
#     saved_model = sys.argv[3]
#     video_file = sys.argv[4]
# else:
#     print ("Usage: python clasify.py sequence_length class_limit saved_model_name video_file_name")
#     print ("Example: python clasify.py 75 2 lstm-features.095-0.090.hdf5 some_video.mp4")
#     exit (1)

capture = cv2.VideoCapture(os.path.join(video_file))
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

print("original shape: (%d x %d)" % (width, height))

#exit(0)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

#save all result of mp4 in folder after clasify
clasify_result = []

filename, file_extension = os.path.splitext(video_file)
result_file = filename + "_result.mp4"
video_writer = cv2.VideoWriter(result_file, fourcc, 30, (int(width), int(height)))

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=(height, width, 3))

width = int(width)
height = int(height)

# get the model.
# extract_model = Extractor(image_shape=(224, 224, 3))
saved_LSTM_model = load_model(saved_model)
# print(saved_LSTM_model.summary())
plot_model(saved_LSTM_model, expand_nested=True, show_shapes=True, to_file='debug_model_clasify.png')

small_frames = False

frames = []
resized_frames = []
frame_count = 0
while True:
    no_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    #print ("no_frame: %d/%d" % (int(no_frame), int(total_frames)))

    if(total_frames < seq_length):
        small_frames = True

    ret, frame = capture.read()

    # Bail out when the video file ends
    if not ret:
        if(small_frames == True):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            ret, frame = capture.read()
        else:
            break

    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)

    # Save each frame of the video to a list
    frame_count += 1
    resized_frames.append(resized_frame)
    frames.append(frame)

    #print ("count=%d, seq_len=%d" % (int(frame_count), int(seq_length)))

    if frame_count < seq_length:
        continue # capture frames untill you get the required number for sequence
    else:
        frame_count = 0

    # For each frame extract feature and prepare it for classification
    # sequence = []
    # for image in resized_frames:
    #     features = extract_model.extract_image(image)
    #     sequence.append(features)

    # Clasify sequence
    prediction = saved_LSTM_model.predict(np.expand_dims(resized_frames, axis=0))
    print(prediction)
    clasify_result.append([video_file, prediction])
    values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    # Add prediction to frames and write them to new video
    for image in frames:
        for i in range(len(values)):
			# cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(image, values[i], (10, 10 * i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA)
        video_writer.write(image)

    frames = []
    resized_frames = []

    if(small_frames == True):
        break

video_writer.release()
save_result_as_csv(clasify_result)