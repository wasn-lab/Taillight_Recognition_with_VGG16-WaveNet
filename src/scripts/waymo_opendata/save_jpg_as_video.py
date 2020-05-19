import os
import cv2
import argparse
import logging


def get_image_filenames(dirname):
    ret = []
    for item in os.listdir(dirname):
        fpath = os.path.join(dirname, item)
        if not os.path.isfile(fpath):
            continue
        if item.endswith(".jpg") or item.endswith(".png"):
            ret.append(fpath)
    ret.sort()
    return ret


def save_as_video(dirname, output_file, width, height, fps):
    vwriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('X', '2', '6', '4'), fps, (width, height))
    for _fn in get_image_filenames(dirname):
        logging.warning("Add %s", _fn)
        img = cv2.imread(_fn)
        img = cv2.resize(img, (width, height))
        vwriter.write(img)
    vwriter.release()
    logging.warning("Write %s", output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", required=True, help="Directory name that contains image files")
    parser.add_argument("--output-file", "-o", default="out.avi", help="Output file")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--fps", type=int, default=15, help="Video frame rate")

    args = parser.parse_args()
    save_as_video(args.dir, args.output_file, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()

