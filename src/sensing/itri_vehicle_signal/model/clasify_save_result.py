import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
from sklearn.metrics import precision_recall_curve

import csv

def save_result_as_csv(result):
	with open('itridataset_clasify_result.csv', 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for i in result:
			writer.writerow(i)


if (len(sys.argv) == 5):
    seq_length = int(sys.argv[1])
    class_limit = int(sys.argv[2])
    saved_model = sys.argv[3]
    video_file = sys.argv[4]
else:
    print ("Usage: python clasify.py sequence_length class_limit saved_model_name video_file_name")
    print ("Example: python clasify.py 75 2 lstm-features.095-0.090.hdf5 some_video_name")
    exit (1)

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
extract_model = Extractor(image_shape=(224, 224, 3))
saved_LSTM_model = load_model(saved_model)

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
    sequence = []
    for image in resized_frames:
        features = extract_model.extract_image(image)
        sequence.append(features)

    # Clasify sequence
    prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
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
