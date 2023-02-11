import sys
import os
import argparse
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QPushButton, QMainWindow
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter
import cv2
import tensorflow as tf
import numpy as np
from StyleTransfer import *
import distance_transform
import utility
from model import *
import time
import scipy.misc
import scipy.io
from gui import Ui_MainWindow

# Output folder for the images.
OUTPUT_DIR = '/Stylization/Output/'

# Content image to use.
content_input_path = "input/font_contents/"
content_with_ext   = "lab6.jpg"
content_image_path = content_input_path + content_with_ext
content_image      = content_with_ext[:-4]

# Style image to use.
style_input_path   = "input/styles/"
style_with_ext     = "flower.png"
style_image_path   = style_input_path + style_with_ext
style_image        = os.getcwd() + "\Stylization\Style\style.jpg"

# Invertion of images
content_invert = 1
style_invert = 1
result_invert = content_invert
###############################################################################
# Algorithm constants
###############################################################################

# path to weights of VGG-19 model
VGG_MODEL = "imagenet-vgg-verydeep-19.mat"
# The mean to subtract from the input to the VGG model.
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

parser = argparse.ArgumentParser(description='A Neural Algorithm of Artistic Style')
parser.add_argument('--w1', '-w1',type=float, default='1',help='w1')
parser.add_argument('--w2', '-w2',type=float, default='1',help='w2')
parser.add_argument('--w3', '-w3',type=float, default='1',help='w3')
parser.add_argument('--w4', '-w4',type=float, default='1',help='w4')
parser.add_argument('--w5', '-w5',type=float, default='1',help='w5')
parser.add_argument("--IMAGE_WIDTH", "-width",type=int, default = 400, help = "width & height of image")
parser.add_argument("--CONTENT_IMAGE", "-CONTENT_IMAGE", type=str, default = content_image_path, help = "Path to content image")
parser.add_argument("--STYLE_IMAGE", "-STYLE_IMAGE", type=str, default = style_image_path, help = "Path to style image")

parser.add_argument("--alpha",  "-alpha",type=float,  default="0.001",   help="alpha")
parser.add_argument("--beta",   "-beta", type=float,  default="0.8",     help="beta")
parser.add_argument("--gamma",  "-gamma",type=float,  default="0.001",    help="gamma")
parser.add_argument("--epoch",  "-epoch",type=int, default=50, help="number of epochs to run" )
args = parser.parse_args()

# Number of iterations to run.
ITERATIONS = args.epoch

# Image dimensions constants.
# image = Image.open(content_image_path)
#IMAGE_WIDTH = image.size[0]
IMAGE_WIDTH = args.IMAGE_WIDTH
IMAGE_HEIGHT = IMAGE_WIDTH
COLOR_CHANNELS = 3

# Style image layer weights
w1 = args.w1
w2 = args.w2
w3 = args.w3
w4 = args.w4
w5 = args.w5

# Content & Style weights
alpha = args.alpha
beta = args.beta
gamma = args.gamma


import sys
from PyQt5.QtWidgets import QWidget, QProgressBar, QPushButton, QApplication
from PyQt5.QtCore import QBasicTimer



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def click_pushButton_01(self):
        self.lineEdit.setText("1. Open Content Image")
        contentImgName = QFileDialog.getOpenFileName(self, 'Open file')[0]
        contentImg = cv2.imread(contentImgName,  cv2.IMREAD_UNCHANGED)
        resContent = cv2.resize(contentImg, (400, 400), interpolation = cv2.INTER_AREA)
        resContentName = os.getcwd() + "\Stylization\Content\content.jpg"
        cv2.imwrite(resContentName, resContent)
        self.plainTextEdit.appendPlainText("1 Opened Content Image " + contentImgName)
        self.stackedWidget.setCurrentIndex(0)
        self.label_1.setPixmap(QtGui.QPixmap(resContentName))
        return

    def click_pushButton_02(self):
        self.lineEdit.setText("2. Open Style Image")
        styleImgName = QFileDialog.getOpenFileName(self, 'Open file')[0]
        styleImg = cv2.imread(styleImgName, cv2.IMREAD_UNCHANGED)
        resstyle = cv2.resize(styleImg, (400, 400), interpolation=cv2.INTER_AREA)
        resstyleName = os.getcwd() + "\Stylization\Style\style.jpg"
        cv2.imwrite(resstyleName, resstyle)
        self.plainTextEdit.appendPlainText("2 Opened Style Image " + styleImgName)
        self.label_2.setPixmap(QtGui.QPixmap(resstyleName))
        return


    def click_pushButton_03(self):
        self.lineEdit.setText("3. Image Segmentation for the Content Image")
        resContentName = os.getcwd() + "\Stylization\Content\content.jpg"
        img = cv2.imread(resContentName)
        self.plainTextEdit.appendPlainText("3 Extracting the product shape from the Content Image "+resContentName)

        copy = img.copy()
        # Connect the mouse button to our callback function

       # param = [img]
        #cv2.setMouseCallback("Rectangle Window", click_and_crop, param)
        x, y, w, h = cv2.selectROI("Select the Area", img)
        start = (x, y)
        end = (x + w, y + h)
        rect = (x, y, w, h)
      #  cv2.rectangle(copy, start, end, (0, 0, 255), 3)
        h, w = copy.shape[:2]
        mask = np.zeros(copy.shape[:2], np.uint8)
        fgdModel = np.zeros((1, 65), np.float64)
        bgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        eximg = img * mask2[:, :, np.newaxis]
        copy = eximg.copy()
        h, w = eximg.shape[:2]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        cv2.floodFill(eximg, mask, (0, 0), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)
        cv2.floodFill(eximg, mask, (38, 313), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)
        cv2.floodFill(eximg, mask, (363, 345), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)

        # write result to disk
        cv2.imwrite(os.getcwd() + "\Stylization\Segmented Content\extracted.jpg", eximg)


        cv2.imwrite(os.getcwd() + "\Stylization\Segmented Content\extracted.jpg", eximg)
        cv2.imshow("Extracted Image", eximg)
        self.label_3.setPixmap(QtGui.QPixmap(os.getcwd() + "\Stylization\Segmented Content\extracted.jpg"))
        resstyleName = os.getcwd() + "\Stylization\Style\style.jpg"
        self.label_4.setPixmap(QtGui.QPixmap(resstyleName))
        self.plainTextEdit.appendPlainText("3 Extracted and Saved the Image as "+os.getcwd() + "\Stylization\Segmented Content\extracted.jpg")
        #CONTENT_IMAGE = os.getcwd() + "\Stylization\Segmented Content\extracted.jpg"
        #STYLE_IMAGE =  resstyleName
        return

    def click_pushButton_04(self):
        OUTPUT_DIR = os.getcwd()+ '/Stylization/Output/' + str(time.time())
        start_time =  time.time()
        self.lineEdit.setText("4. Stylize the Content Image")
        with tf.device("/gpu:0"):
            with tf.compat.v1.Session() as sess:

                # Load images.
                content_image = utility.load_image(os.getcwd() + '\Stylization\Segmented Content\extracted.jpg', IMAGE_HEIGHT, IMAGE_WIDTH, invert=content_invert)
                style_image = utility.load_image(os.getcwd() + '\Stylization\Style\style.jpg', IMAGE_HEIGHT, IMAGE_WIDTH, invert=style_invert)
                #utility.save_image(OUTPUT_DIR+"/"+style_name+".jpg", style_image, invert = style_invert)

                # Load the model.
                model = load_vgg_model(VGG_MODEL, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
                # Content image as input image
                initial_image = content_image.copy()
                # Initialize all variables
                sess.run(tf.compat.v1.global_variables_initializer())

                # Construct content_loss using content_image.
                sess.run(model['input'].assign(content_image))
                content_loss = content_loss_func(sess, model)

                # Construct shape loss using content image
                sess.run(model["input"].assign(initial_image))
                dist_template_inf, content_dist_sum = distance_transform.dist_t(content_image)
                ### take power of distance template
                dist_template = np.power(dist_template_inf, 8)
                dist_template[dist_template > np.power(2, 30)] = np.power(2, 30)

                shape_loss = shape_loss_func(sess, model, dist_template, content_dist_sum)

                # Construct style_loss using style_image.
                sess.run(model['input'].assign(style_image))
                style_loss = style_loss_func(sess, model)

                # Instantiate equation 7 of the paper.
                total_loss = alpha * content_loss + beta * style_loss + gamma * shape_loss

                # Then we minimize the total_loss, which is the equation 7.
                optimizer = tf.compat.v1.train.AdamOptimizer(1.0)
                train_step = optimizer.minimize(total_loss)

                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(model['input'].assign(initial_image))
                for it in range(ITERATIONS + 1):
                    sess.run(train_step)

                    if it % 10 == 0:
                        # Print every 10 iteration.
                        mixed_image = sess.run(model['input'])
                        self.plainTextEdit.appendPlainText('Stylize the Content Image')
                        self.plainTextEdit.appendPlainText('Iteration %d' % (it))
                        self.plainTextEdit.appendPlainText('sum         : '+ str(sess.run(tf.reduce_sum(mixed_image))))
                        self.plainTextEdit.appendPlainText('total_loss  : '+ str(sess.run(total_loss)))
                        self.plainTextEdit.appendPlainText("content_loss: " + str(alpha * sess.run(content_loss)))
                        self.plainTextEdit.appendPlainText("style_loss  : " + str(beta * sess.run(style_loss)))
                        self.plainTextEdit.appendPlainText("shape loss  : "+ str(gamma * sess.run(shape_loss)))

                        if not os.path.exists(OUTPUT_DIR):
                            os.mkdir(OUTPUT_DIR)
                        filename = OUTPUT_DIR + '/%d.jpg' % (it)
                        utility.save_image(filename, mixed_image, invert=result_invert)
                        self.label_3.setPixmap(QtGui.QPixmap(filename))
                    if sess.run(total_loss) < 1:
                        break
            sess.close()
        end_time = time.time()
        self.plainTextEdit.appendPlainText("Time taken = " + str(end_time - start_time))
        self.plainTextEdit.appendPlainText("4 Stylized the extracted content image")
        obj = OUTPUT_DIR + '/50.jpg'
        im = cv2.imread(os.getcwd() + '/Stylization/Content/content.jpg')

        self.label_3.setPixmap(QtGui.QPixmap(obj))
        # Create an all white mask
        obj = cv2.imread(obj)
        mask = 255 * np.ones(obj.shape, obj.dtype)

        # The location of the center of the src in the dst
        width, height, channels = im.shape
        center = (height // 2, width // 2)

        # Seamlessly clone src into dst and put the results in output
        clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
       # mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

        # Write results
        cv2.imwrite(OUTPUT_DIR+"\merged.jpg", clone)
        self.label_3.setPixmap(QtGui.QPixmap(OUTPUT_DIR+"\merged.jpg"))
        self.stackedWidget.setCurrentIndex(0)  # 打开 stackedWidget > page_0
        return
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

if __name__ == '__main__':
    mainapp = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(mainapp.exec_())  # 结束进程，退出程序
