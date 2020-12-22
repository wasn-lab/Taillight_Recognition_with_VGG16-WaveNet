import sys
import os
import cv2
import numpy as np

def find_roi_by_lane_instance(image_file):
    img = cv2.imread(image_file)
    top_points = {}
    bottom_points = {}
    rows, cols, channels = np.nonzero(img)
    for row, col, _channel in zip(rows, cols, channels):
        color = tuple(img[row][col])
        if color not in top_points:
            top_points[color] = (row, col)
        else:
            if row < top_points[color][0]:
                top_points[color] = (row, col)

        if color not in bottom_points:
            bottom_points[color] = (row, col)
        else:
            if row > bottom_points[color][0]:
                bottom_points[color] = (row, col)
    roi = []
    for point in sorted(top_points.values(), key=lambda p: (p[1], p[0])):
        roi.append(point[::-1])
    bps = list(bottom_points.values())
    bps.sort(key=lambda p: (p[1], p[0]), reverse=True)
    max_row = img.shape[0] - 1
    if bps:
        first_point = bps[0]
        roi.append(first_point[::-1])
        if len(bps) > 1:
            if first_point[0] != max_row:
                roi.append((first_point[1], max_row))
        for point in bps[1:]:
            roi.append(point[::-1])
    return roi


def fill_roi(img_file, roi):
    img = cv2.imread(img_file)
    roi = np.array([roi])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi, (255,255,0))
    return cv2.bitwise_or(img, mask)


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        filename = line[:-4] + "_lane_instance.png"
        print("Handle {}".format(filename))
        if not os.path.isfile(filename):
            continue
        roi = find_roi_by_lane_instance(filename)
        if (len(roi) >= 3):
            output_fn = line[:-4] + "_lane_roi.png"
            output = fill_roi(filename, roi)
            print("Write {}".format(output_fn))
            cv2.imwrite(output_fn, output)
