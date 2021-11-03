#!/usr/bin/env python
# pylint: disable=no-name-in-module
import matplotlib
import numpy as np
matplotlib.use('Agg')
import lanelet2
import tempfile
import os
from lanelet2.core import AttributeMap, TrafficLight, Lanelet, LineString3d, Point2d, Point3d, getId, \
    LaneletMap, BoundingBox2d, BasicPoint2d
from lanelet2.projection import UtmProjector
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import argparse
import math
import csv
import xml.etree.ElementTree as ET

def lane2array(args):
    print("====================================")
    print("map file: {}".format(args.map))
    if args.output == 0:
        print("Output: Only numpy")
    elif args.output == 1:
        print("Output: Numpy and figure")
    if args.target == 0:
        print("type: Vehicle")
    elif args.target == 1:
        print("type: Motorcycle")
    elif args.target == 2:
        print("type: Pedestrian")
    print("====================================")
    
    map_name = str(args.map + '.osm')
    map_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res/", map_name)
    tree = ET.parse(map_file)
    root = tree.getroot()
    lat = root[1].attrib['lat']
    lon = root[1].attrib['lon']
    lat = round(float(lat), 1)
    lon = round(float(lon), 1)
    projector = UtmProjector(lanelet2.io.Origin(lat, lon))
    map = lanelet2.io.load(map_file, projector)
    del tree
    del root
    
    lanes = map.laneletLayer
    polygons = []
    xs = []
    ys = []
    xs_3d = []
    ys_3d = []

    for elem in lanes:
        polygons.append(elem.polygon2d())
        # print(elem.polygon2d()[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get bbox of the map
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    for polys in polygons:
        for p in polys:
            # print(p.attributes)
            # multiplied by 3 to fit the scale of 3 pixels/meter
            xs.append(round(3 * float(p.attributes["local_x"]), 2))
            ys.append(round(3 * float(p.attributes["local_y"]), 2))
        # repeat the first point to create a 'closed loop'
        xs.append(xs[0])
        ys.append(ys[0])
        # draw lanelet in a figure (show only)
        ax.fill(xs,ys, color='black')
        if args.target > 0:
          w = 3*int(args.target)
          ax.plot(xs,ys, linewidth=w, color='black')
        xs_3d.append(xs)
        ys_3d.append(ys)
        xs = []
        ys = []

    # find max and min
    xmax = max(max(xs_3d, key=max))
    xmin = min(min(xs_3d, key=min))
    ymax = max(max(ys_3d, key=max))
    ymin = min(min(ys_3d, key=min))
    
    print("Notice: These values are the Max and min of this map.")
    print("You should note xmin and ymin down for future uses in process_data in IPP model.")
    print("xmax: {}".format(str(xmax/3))) 
    print("xmin: {}".format(str(xmin/3)))
    print("ymax: {}".format(str(ymax/3)))
    print("ymin: {}".format(str(ymin/3)))

    width = int(math.ceil(xmax) - math.floor(xmin))
    height = int(math.ceil(ymax) - math.floor(ymin))

    img = Image.new('1', (width, height), 1)

    for i in range(len(xs_3d)):
        xys = []
        for j in range(len(xs_3d[i])):
            xys.append(int(xs_3d[i][j] - xmin))
            xys.append(int(ys_3d[i][j] - ymin))
        ImageDraw.Draw(img).polygon(xys, outline=0, fill=0)
        if args.target > 0:
          w = 12 * int(args.target)
          ImageDraw.Draw(img).line(xys, width=w)
    # save_img = str(map_name[:-4] + "_img.png")
    # img.save(save_img)
    img_np = np.array(img)
    # print(img_np.shape)
    # np.save(map_name[:-4], img_np)
  
    # output
    if args.output == 0:
        np.save(map_name[:-4], img_np)
        print("Output only numpy")
    elif args.output == 1:
        np.save(map_name[:-4], img_np)
        save_fig = str(map_name[:-4] + "_fig.png")
        plt.savefig(save_fig, bbox_inches='tight', pad_inches=0)
        print("Output numpy and figure")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", help="assign the map you use here", default='ITRI_lanelet2_map')
    parser.add_argument("--output", type=int, help="0=numpy, 1=numpy+figure", default=0)
    parser.add_argument("--target", type=int, help="0=vehicle, 1=motorcycle, 2=pedestrian", default=0)
    args = parser.parse_args()
    
    lane2array(args) 