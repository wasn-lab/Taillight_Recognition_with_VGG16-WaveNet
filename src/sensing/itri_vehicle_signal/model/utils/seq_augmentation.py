import os
import sys
import glob
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa



seq = iaa.Sequential([
	# # crop images from each side by 0 to 16px (randomly chosen)
	# iaa.Crop(px=(0, 16)),

	# # horizontally flip 50% of the images
	# iaa.Fliplr(0.5),

	# # blur images with a sigma of 0 to 3.0
	# iaa.GaussianBlur(sigma=(0, 3.0)),

	# # Strengthen or weaken the contrast in each image.
	# iaa.LinearContrast((0.75, 1.5)),

    # # Add gaussian noise.
    # # For 50% of all images, we sample the noise once per pixel.
    # # For the other 50% of all images, we sample the noise per pixel AND
    # # channel. This can change the color (not only brightness) of the
    # # pixels.
	# iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

	# # Make some images brighter and some darker.
	# # In 20% of all cases, we sample the multiplier once per channel,
	# # which can end up changing the color of the images.
	# iaa.Multiply((0.8, 1.2), per_channel=0.2),

	# Apply affine transformations to each image.
	# Scale/zoom them, translate/move them, rotate them and shear them.
	# iaa.Affine(
		# scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
		# translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
		# rotate=(-25, 25),
		# shear=(-8, 8)
	# )

	# iaa.AddToBrightness((0, 100)),
	iaa.AddToBrightness((-10, 60)),

	#iaa.AddToHue((-255, 255)),

	#Increases or decreases hue and saturation by random values
	#add random values between (a,b) to the hue and saturation
	# iaa.AddToHueAndSaturation((-100, 100), per_channel=True),
	iaa.AddToHueAndSaturation((-15, 15), per_channel=True),

	#Apply random four point perspective transformations to images
	#most transformations don’t change the image very much, 
	#while some “focus” on polygons far inside the image.
	iaa.PerspectiveTransform(scale=(0.01, 0.1)),


	# The augmenter has the parameters alpha and sigma.
	# alpha controls the strength of the displacement: higher values mean that pixels are moved further. 
	# sigma controls the smoothness of the displacement: higher values lead to smoother patterns – as if the image was below water 
	# – while low values will cause indivdual pixels to be moved very differently from their neighbours, leading to noisy and pixelated images.
	iaa.ElasticTransformation(alpha=(0, 1), sigma=(0.25, 0.5)),

	# iaa.AddToHueAndSaturation((-50, 50), per_channel=True),

	# iaa.AddToSaturation((-255, 255)),

], random_order=True) # apply augmenters in random order



def aug_frames(path, aug_count):

	# print("start augment")
	seq_len = 20

	nb_frames = 0

	data_files = []

	path = os.path.abspath(path)

	for root, dirs, files in os.walk(path):
		#print (root, dirs, files)
		frames = sorted(glob.glob(os.path.join(root, '*jpg')))
		nb_frames = len(frames)

		#print("root: %s" % root)
		#print("nb_frames: %d" % nb_frames)
		#print(frames)

		if(nb_frames <= 0):
			continue

		#print(aug_count)

		images = []

		for frame in frames:
			img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
			images.append(img)


		for aug_i in range(aug_count):

			print("path: %s" % path)

			dirname, basename = os.path.split(path)

			#print("dirname: %s" % dirname)
			#print("basename: %s" % basename)

			print("base sequence: [ %s ]" % basename)

			aug_seq_name = dirname + "/aug_seq_%02d_" % (aug_i) + basename

			aug_seq_name = "aug_seq_%02d_" % (aug_i) + basename

			aug_seq_path = dirname + "/" + aug_seq_name + "/light_mask"

			#print("aug_seq_path: %s" % aug_seq_path)
			print("    create augmented sequence: [ %s ]" % aug_seq_name)

			mkdir(aug_seq_path)

			seq_det = seq.to_deterministic()

			#images_aug = seq(images=images)

			images_aug = [seq_det.augment_image(img) for img in images]

			# write image files
			i=0
			for img_aug in images_aug:
				filename = "frame%08d.jpg" % i
				output_file = aug_seq_path + "/" + filename
				cv2.imwrite(output_file, images_aug[i])
				#print("        [ %s ]" % filename)
				i+=1

			print("        [%d] jpg file created" % i)


def mkdir(path):
	try:
		os.makedirs(path)
	except OSError:
		pass

def main():
	if (len(sys.argv) == 3):
		extract_path = sys.argv[1]
		aug_count = int(sys.argv[2])
		aug_frames(extract_path, aug_count)
	else:
		print ("invalid argument")
		print ("Usage: python seq_augmentation.py [path] [aug_count]")

if __name__ == '__main__':
    main()


