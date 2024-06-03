#%loaddistance_transform.py
"""
InputsimageswithblackbackgroundasasTensor
withshapeof(1,height,width,3)
"""
import cv2
import numpy as np

"""
Inputs(1,height,width,3)tensor.
Outputs(height,width)asgrayscaleimage
"""

def numpy_to_image(image):

	image=image*1.0
	#RemoveVGGoptimization
	#ifimage.shape==(1,400,400,3):
	if image.ndim==4:
		image+=np.array([123.68,116.779,103.939]).reshape((1,1,1,3))
		#Cutunneededaxis
		image=image[0]
	elif image.ndim==3:
	#elifimage.shape==(400,400,3):
		image+=np.array([123.68,116.779,103.939]).reshape((1,1,3))
	
	image=np.clip(image,0,255).astype("uint8")

	#Makegrayscale
	image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)	
	return image_gray

"""
Inputs3-channelimage,
returns1.Distancetransformationwithbackgroundas(+),foregroundas(-)
		2.Sumofallelementsinpixel-wisemultiplyofdist_tandoriginalimage

"""
def dist_t(image):
	
	#Grayscaleimage
	image_gray=numpy_to_image(image)

	#Binaryimage
	#Thethresholdresultswillbeatuple,with[value,image]
	image_th=cv2.threshold(image_gray,50,255,cv2.THRESH_BINARY)[1]
	#cv2.imwrite("image_f_dist.jpg",image_th)

	#Invertimages
	image_inv=(255-image_th)

	###Distancetransform

	#Distancetransformationofbackground
	dist_t_bg=cv2.distanceTransform(image_inv,cv2.DIST_L2,3)

	#Distancetransformationofcharactersorpatterns
	dist_t_fg=cv2.distanceTransform(image_th,cv2.DIST_L2,3)

	#Makenewdistancetransformationwith(-)inside&(+)background
	dist_template=dist_t_bg+dist_t_fg*(-1)

#Imagefloat
	image_float=image_gray/255.0

	#Multiplypixel-wisewithdist1
	mult=np.multiply(image_float,dist_template)

	#Sumofallelementsinmultipliedmult
	dist_sum=np.sum(mult)

	#returnonlydist_t_gasdist_template,dist_sum
	return dist_t_bg, dist_sum
	
"""
Assign1stoinputimagecharacters,
andcalculatedistancelossaspixelwise
"""

def dist_loss(image,dist_template,orig_sum):

	#Grayscaleimage
	image_gray=numpy_to_image(image)
	cv2.imwrite("image_gray.jpg",image_gray)

	image_float=image_gray/255.0

	#Multiplypixel-wisewithdist1
	mult=np.multiply(image_float,dist_template)

	#Sumofallelementsinmultipliedmult
	dist_sum=np.sum(mult)

	#Absolutevalueofdifferencebetweenorig_sum&dist_sum
	dist_loss=abs(orig_sum-dist_sum)
	
	#withopen("output.txt","w")asf:
	#	f.write(dist_sum)
	#	f.write(dist_loss)

	return dist_loss