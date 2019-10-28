#!/usr/bin/env python3

import collections, os, io, sys, tarfile, tempfile, urllib, time, rospy, cv2
import threading
from argparse import ArgumentParser
from queue import Queue
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage as ImageMsg

from msgs.msg import Boundary, FreeSpace, FreeSpaceResult
import matplotlib
from matplotlib import gridspec
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from utils import get_dataset_colormap

# from tensorflow.python.client import timeline

parser = ArgumentParser()
parser.add_argument("-d", "--display", help="disable display, with true or false", dest="display", default="False")
parser.add_argument("-f", "--fill", help="Fill fps to 30", dest="fill", default="True")
args = parser.parse_args()

if args.display.lower() in ('yes', 'true', 't', 'y'):
    _debug = True
else:
    _debug = False

if args.fill.lower() in ('no', 'false', 'f', 'n'):
    _required_fps = False
else:
    _required_fps = True

_enableRT = False
if _enableRT:
    import tensorflow.contrib.tensorrt as trt
pil_im = Image.new('RGB', (60, 30), color = 'white')
pil_im_0 = Image.new('RGB', (60, 30), color = 'white')
pil_im_2 = Image.new('RGB', (60, 30), color = 'white')
Current = []
Switchable = []
Unswitchable = []
result_sepMap = np.empty([1,1])

fs_pub = rospy.Publisher('freespace', FreeSpaceResult, queue_size=10)

### For image distance estimation
### ===============================
Hor_regionHeightFC = [1207, 1120, 1044, 985, 940, 900, 868, 840, 818, 796, 780, 764, 748, 736, 724, 680, 648, 625, 608, 596, 584, 0]
Hor_regionDist = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100]
Ver_regionHeightFC = [0, 135, 463, 741, 990, 1262, 1543, 1806, 1919]
Ver_regionDist = [4, 3, 2, 1, 0, -1, -2, -3, -4]
slope = [1, 1.3, 2, 4, 0, -4, -2, -1.3, -1]
Lidar_offset = 1
disEsti_imgHeight = 1208
disEsti_imgWidth = 1920
### ===============================

## Load model in TensorFlow
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    
    INPUT_SIZE = 769

    def __init__(self, pb_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        
        graph_def = None
        # Extract frozen graph from pb_path.
        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Convert to TensorRT graph def
        if _enableRT:
            trt_graph_def = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=['SemanticPredictions:0'],
                max_batch_size=16,
                max_workspace_size_bytes=2<<20,
                is_dynamic_op=True, # important
                minimum_segment_size=3,
                precision_mode="fp16")
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        if _enableRT:
            trt_engine_ops = len([1 for n in trt_graph_def.node if str(n.op)=='TRTEngineOp'])
            allnode = len(trt_graph_def.node)
            print("trt_engine_ops:      " + str(trt_engine_ops) + ",in total:     " + str(allnode))

            tf.reset_default_graph()

        with self.graph.as_default(): 
            if _enableRT:     
                tf.import_graph_def(trt_graph_def, name='')
            else:
                tf.import_graph_def(graph_def, name='')
        
        configa = tf.ConfigProto()
        configa.gpu_options.allow_growth = True
        configa.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=configa,graph=self.graph)
        
            
    def run(self, image
    ,q ,im
    ):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size

        # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        resize_ratio = 1 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        resized_image = image.resize(target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
            # ,options=options, run_metadata=run_metadata
            )

        seg_map = batch_seg_map[0]

        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
        
        # with open('timeline_02_step_%d.json' % int(end), 'w') as f:
        #     f.write(chrome_trace)

        q.put(seg_map)
        im.put(resized_image)

        # return resized_image, seg_map

rp = RosPack()
model = DeepLabModel(rp.get_path('openroadnet') + "/frozen_inference_graph_4k.pb")
Trans_Mat = np.load(rp.get_path('openroadnet') + "/trans_mat_t.npy")

if _debug:
    plt.figure(figsize=(20, 10))
    # grid_spec = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    grid_spec = gridspec.GridSpec(1, 1)

### Calculate cordinate in 3D from pixel(Unuse now)
def GetPointDist(pixel_x, pixel_y, x_regionHeight, x_regionDist, y_regionHeight, y_regionDist, Lidar_offset):
    x_distMeter = 0
    y_distMeter = 0

    x_loc = pixel_y
    y_loc = pixel_x

    if len(x_regionDist) != 0:
        x_distMeter = ComputeObjectXDist(x_loc, x_regionHeight, x_regionDist)
    if len(y_regionDist) != 0:
        y_distMeter = ComputeObjectYDist(y_loc, x_loc, y_regionHeight, y_regionDist)
    
    return x_distMeter + Lidar_offset, y_distMeter

# (Unuse now)
def ComputeObjectXDist(pixel_loc, regionHeight, regionDist):
    distance = -1
    unitLength = 0.0
    bias = 0
    offset = 0

    for i in range(1,len(regionHeight)):
        if pixel_loc >= regionHeight[i] and pixel_loc <= regionHeight[i-1]:
            regionpixel = regionHeight[i-1] - regionHeight[i]
            regionMeter = regionDist[i] - regionDist[i-1]
            unitLength = float(regionMeter) / float(regionpixel)
            bias = pixel_loc - regionHeight[i]
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset

        elif pixel_loc <= regionHeight[i] and pixel_loc >= regionHeight[i-1]:
            regionpixel = regionHeight[i] - regionHeight[i-1]
            regionMeter = regionDist[i] - regionDist[i-1]
            unitLength = float(regionMeter)/float(regionpixel)
            bias = regionHeight[i] - pixel_loc
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset
    
    multiplier = pow(10, 2)
    distance = int(distance * multiplier) / (multiplier*1.0)

    return distance

# (Unuse now)
def ComputeObjectYDist(pixel_locY, pixel_locX, regionHeight, regionDist):
    distance = 0
    unitLength = 0.0
    bias = 0
    offset = 0.0
    
    new_regionHeight = regionHeight.copy()

    for a in range(len(regionHeight)):
        _y = disEsti_imgHeight - pixel_locX
        if slope[a] != 0:
            new_regionHeight[a] = regionHeight[a] + int((1/slope[a]) * _y)

    for i in range(1,len(new_regionHeight)):
        if pixel_locY >= new_regionHeight[i] and pixel_locY <= new_regionHeight[i-1]:
            regionpixel = new_regionHeight[i-1] - new_regionHeight[i]
            regionMeter = regionDist[i] - regionDist[i-1]
            if regionpixel != 0:
                unitLength = float(regionMeter) / float(regionpixel)
            bias = pixel_locY - new_regionHeight[i]
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset

        elif pixel_locY <= new_regionHeight[i] and pixel_locY >= new_regionHeight[i-1]:
            regionpixel = new_regionHeight[i] - new_regionHeight[i-1]
            regionMeter = regionDist[i] - regionDist[i-1]
            if regionpixel != 0:
                unitLength = float(regionMeter)/float(regionpixel)
            bias = new_regionHeight[i] - pixel_locY
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset
        # Modifying: when pixel_locY < new_regionHeight[0] or > new_regionHeight[4] what will happen
        else:
            if pixel_locY < new_regionHeight[0]:
                distance = 5

            elif pixel_locY > new_regionHeight[len(new_regionHeight) - 1]:
                distance = -5
    
    multiplier = pow(10, 2)
    distance = int(distance * multiplier) / (multiplier*1.0)
    
    return distance

def GetPointDistFromMatrix(pixel_x, pixel_y):
    return Trans_Mat[pixel_x][pixel_y][0], Trans_Mat[pixel_x][pixel_y][1]

### Drawing result using matplotlib
def drawing(im, im1, seg_map):
    
    im_ = im.copy()
    plt.clf()
    
    # plt.subplot(grid_spec[0])
    # plt.imshow(im1)
    # plt.axis('off')
    # plt.title('Segmentation map')

    # plt.subplot(grid_spec[1])
    # plt.scatter(Switchable, Unswitchable)
    # plt.axis([0, 100, 5, -5])
    # plt.grid(True)

    # plt.axis('off')
    # plt.title('Projection')

    # for x in range(len(Current)-1):   
    #      cv2.line(im1, (int(Current[x][0]), int(Current[x][1])), (int(Current[x+1][0]), int(Current[x+1][1])), (255, 0, 0), 3)
    plt.subplot(grid_spec[0])
    plt.imshow(im_)  
    plt.imshow(im1, alpha=0.4)
    plt.axis('off')
    plt.title('Overlap')
    plt.draw()  
    plt.pause(.01)

### Calculate boundary with segmentation map
def calBoundary(segMap
,Q_c, Q_s, Q_u
):
    # initialize
    current = []
    switch = []
    unswitch = []

    step_x = 10
    step_y = 3
    init_y = 200

    seg_width = segMap.shape[1]
    seg_height = segMap.shape[0]    

    # Take HUGE time
    for a in range(0, seg_width, step_x):
        for b in range(init_y, seg_height, step_y):
            if segMap[b][a] >= 1:
                pix_x = int(a*(disEsti_imgWidth/seg_width))
                pix_y = int(b*(disEsti_imgHeight/seg_height))
                
                x, y = GetPointDistFromMatrix(pix_x, pix_y)                
                
                current.append([x,y])
                break
            if b >= seg_height - 1:
                pix_x = int(a*(disEsti_imgWidth/seg_width))
                               
                x, y = GetPointDistFromMatrix(pix_x, disEsti_imgHeight-1)
                
                current.append([x,y])
                break

    if _debug:
        for a in range(len(current)):
            switch.append(current[a][0])
            unswitch.append(current[a][1])

    Q_c.put(current)
    Q_s.put(switch)
    Q_u.put(unswitch)

    # return current, switch, unswitch

### Transform segmentation map from grayscale to color
def vis_segmentation(image, seg_map):
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_cityscapes_name()).astype(np.uint8)    
    # seg_map is an array with label e.g. 0, 1, 2...
    drawing(image, seg_image, seg_map)

### Inference model to obtain result
def run_demo_image(image_name):
    global Current, Switchable, Unswitchable, result_sepMap
    start = time.time()

    # Threading ------
    tmp_Cur = Queue()
    tmp_Swi = Queue()
    tmp_Uns = Queue()
    tmp_sepMap = Queue() 
    resized_im_q = Queue() 
    
    t1 = threading.Thread(target=model.run, args=(image_name, tmp_sepMap, resized_im_q))    

    t1.start()

    t2 = threading.Thread(target=calBoundary, args=(result_sepMap, tmp_Cur, tmp_Swi, tmp_Uns))
    t2.start()

    t1.join()
    t2.join()

    result_sepMap = np.asarray(list(tmp_sepMap.get()))
    Current = tmp_Cur.get()
    Switchable = tmp_Swi.get()
    Unswitchable = tmp_Uns.get() 
    resized_im = resized_im_q.get()

    # seg_map is 'numpy.ndarray'
    # resized_im, seg_map = model.run(image_name)
    # Current, Switchable, Unswitchable = calBoundary(seg_map)    

    fps = 1/(time.time() - start)
    print('[FPS]:    ' + str(fps))
    print()
    if not _required_fps:
        publisher(Current)
    
    if _debug == True:
        vis_segmentation(resized_im, result_sepMap)

### Callback function to subscribe messages which needed
def callback(imgmsg):
    global pil_im
    np_arr = np.fromstring(imgmsg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img)
    if not _debug:
        run_demo_image(pil_im)

### Publish boundary information
def publisher(bound):   
    
    fSpace_msg = FreeSpace()
    fSpaceRst_msg = FreeSpaceResult()

    for i in range(len(bound)):
        bd_msg = Boundary()
        bd_msg.x = bound[i][0]
        bd_msg.y = bound[i][1]
        fSpace_msg.current.append(bd_msg)
    # fSpace_msg.switchable.append(bd_msg)
    # fSpace_msg.unswitchable.append(bd_msg)
    fSpaceRst_msg.freespace.append(fSpace_msg)
    fSpaceRst_msg.header.stamp = rospy.Time.now()
    fs_pub.publish(fSpaceRst_msg)

### Main function
def listener():
    global pil_im
    rospy.init_node('OpenRoadNet', anonymous=True)
    rospy.Subscriber("/gmsl_camera/port_a/cam_1/image_raw/compressed",ImageMsg, callback)

    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        if _debug:
            run_demo_image(pil_im)
        if _required_fps:
            publisher(Current)
        rate.sleep()
    
    rospy.spin()

if __name__ == '__main__':
    listener()