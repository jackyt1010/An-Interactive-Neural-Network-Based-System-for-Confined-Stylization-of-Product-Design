import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from StyleTransfer import *
import distance_transform
import utility
from model import *
from pathlib import Path
import tensorflow as tf
import time
import sys
import os

def drawLine(event,x,y,flags,params):
    fgdModel = np.zeros((1, 65), np.float64)
    bgdModel = np.zeros((1, 65), np.float64)
    ixLineAdd, iyLineAdd, ixLineRemove, iyLineRemove, ix, iy, drawing, color, whichClick,imageHeight, imageWidth, lineAdd, lineRemove, img, mask, originalImage, grabcut_textbox = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],  params[8], params[9], params[11], params[12],  params[13],  params[14], params[15], params[16], params[17]
   # global ixLineAdd,iyLineAdd,ixLineRemove, iyLineRemove, ix, iy, drawing, color, whichClick, imageHeight, imageWidth, lineAdd, lineRemove
    # Left Mouse Button Down Pressed
    if(event==1):
        drawing = True
        color = 50
        whichClick = 1
        ix = x
        iy = y
    # Right mouse button Down Pressed
    if(event==2):
        drawing = True
        color = 50
        whichClick = 2
        ix = x
        iy = y
   # if(event==0):
    if(drawing==True):
            ix = x
            iy = y
            #For Drawing Line
            #cv2.line(img,pt1=(ix,iy),pt2=(x,y),color=(0,255,0),thickness=3)
          
            if(whichClick == 1):
                for i in range(0, 5):
                    if(((ix+i) < imageWidth) and ((ix-i) >= 0)):
                        #print(ix+i,i)
                        ixLineAdd.append(ix+i)
                    if(((iy+i) < imageHeight) and ((iy-i) >= 0)):    
                        iyLineAdd.append(iy+i)
                    if(((ix-i) >= 0) and ((ix+i) < imageWidth)):
                        ixLineAdd.append(ix-i)
                    if(((iy-i) >= 0) and ((iy+i) < imageHeight)):
                        iyLineAdd.append(iy-i)
            elif(whichClick==2):
                for i in range(0, 5):
                    if(((ix+i) < imageWidth) and ((ix-i) >= 0)):
                        #print(ix+i,i)
                        ixLineRemove.append(ix+i)
                    if(((iy+i) < imageHeight) and ( (iy-i) >= 0)):
                        iyLineRemove.append(iy+i)
                    if(((ix-i) >= 0) and ((ix+i) < imageWidth)):
                        ixLineRemove.append(ix-i)
                    if(((iy-i) >= 0) and ((iy+i) < imageHeight)):
                        iyLineRemove.append(iy-i)
                    #print(ixLineRemove,i)
                
    # Mouse button released
    if(event==4 or event==5):
        drawing = False
        if(len(iyLineAdd)>0 and len(ixLineAdd)>0):
            n = min(len(iyLineAdd), len(ixLineAdd))
            lineAdd = np.vstack((iyLineAdd[:n], ixLineAdd[:n])).T
            #iyLineAdd = np.unique(iyLineAdd, axis=0)
            #ixLineAdd = np.unique(ixLineAdd, axis=0)
        if(len(iyLineRemove)>0 and len(ixLineRemove)>0):
            n = min(len(iyLineRemove), len(ixLineRemove))
            lineRemove = np.vstack((iyLineRemove[:n], ixLineRemove[:n])).T
            #iyLineRemove = np.unique(iyLineRemove, axis=0)
            #ixLineRemove = np.unique(ixLineRemove, axis=0)
    if len(lineAdd)>0:
        mask[lineAdd[:,0],lineAdd[:,1]] = 1
    if len(lineRemove)>0:
        mask[lineRemove[:,0],lineRemove[:,1]] = 0
    #Grabcut algorithm with template mask, save the mask before the grabcut changes it
    maskTmp = mask.copy()
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,25,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    imgTmp = originalImage*mask[:,:,np.newaxis]

def GrabCut(grabcut_textbox):
   current_dir = current_directory = os.path.dirname(os.path.abspath(__file__))
   fileName = 'original.jpg'
   img = cv2.imread(current_dir+'/images/'+fileName)[:,:,::-1]
   mask = np.zeros(img.shape[:2],np.uint8)
   imgTmp = img.copy()
   originalImage = img.copy()
# GrabCut parameters
   bgdModel = np.zeros((1,65),np.float64)
   fgdModel = np.zeros((1,65),np.float64)
   params = [imgTmp]
   drawing = False

# ## Select a rectangle for the foreground, press 'q' when finished!
# Show rectangle including foreground item on image
   x, y, w, h = cv2.selectROI("Select the Target Area", img)
   start = (x, y)
   end = (x + w, y + h)
   rect = (x, y, w, h)
      #  cv2.rectangle(copy, start, end, (0, 0, 255), 3)
   h, w = img.shape[:2]
   imageHeight, imageWidth = h, w 
   mask = np.zeros(img.shape[:2], np.uint8)
   cv2.grabCut(img,mask,rect,bgdModel,fgdModel,30,cv2.GC_INIT_WITH_RECT)

# Show current image only with rectangle
   mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
   img = img*mask2[:,:,np.newaxis]
# Making Window For The Image
   
# Adding Mouse CallBack Event

# Starting The Loop So Image Can Be Shown and allow exit on "Q" key pressed
   #while(True):
   #    if cv2.waitKey(20) & 0xFF == 13:
   #        break
           
  # cv2.destroyAllWindows()
  # figure, ax = plt.subplots(1)

 

# ## First pass of the GrabCut Algorithm


# ## Loop with user interaction until satisfied
# ### Press 'q' to close the window!

# In[6]:
# Loop for the algorithm
   counter = 0
   params=[]
   while(True):
    
    # Variables used
       drawing = False
       params.append(drawing)
       ixLineAdd, iyLineAdd,ixLineRemove, iyLineRemove, ix, iy, whichClick = [], [],[], [], 0, 0, 0
       lineAdd = []
       lineRemove = []
       imgTmp = img.copy()
       color = 50
      # if counter > 0:
      #     mask = maskTmp.copy()
    # After the first iteration, replace the mask since it was converted from [0:3] to [0:1]
       params = [0]*18
       params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],  params[8], params[9], params[11], params[12],  params[13], params[14], params[15], params[16], params[17]  =ixLineAdd, iyLineAdd, ixLineRemove, iyLineRemove, ix, iy, drawing, color, whichClick,imageHeight, imageWidth, lineAdd, lineRemove, img, mask, originalImage, grabcut_textbox
    # Making Window For The Image
       cv2.namedWindow("Draw Lines to Better cut the target Image")
    # Adding Mouse CallBack Event
       cv2.imshow("Draw Lines to Better cut the target Image",img)
    # Starting The Loop So Image Can Be Shown
       while(True):
           cv2.setMouseCallback("Draw Lines to Better cut the target Image",drawLine, params)
    # Edit the mask for the algorithm based on the user's inputs
           if cv2.waitKey(20) & 0xFF == 13:
               break
       counter = counter + 1
       break
   cv2.destroyAllWindows()
    # Show final result and ask to continue the algorithm or not
      # print("Final result: ")
      # plt.imshow(img),plt.show()
      # if input('Would you like to continue editing?\n') != 'y':
      #     break
      # clear_output(wait=True)


# ## Save & remove background

# In[7]:
   mask = np.zeros([h + 2, w + 2], np.uint8)
   cv2.floodFill(img, mask, (0, 0), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)
   cv2.floodFill(img, mask, (38, 313), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)
   #cv2.floodFill(imgTmp, mask, (363, 345), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)
   #cv2.floodFill(imgTmp, mask, (363, 345), (255, 255, 255), (3, 151, 65), (3, 151, 65), flags=8)
   im = Image.fromarray(img)
   path = Path(current_dir+'/images/'+'extracted.png')
   im.save(path)

# Transfer black as transparent and save new image
  # img = Image.open(path)
  # img = img.convert("RGBA")
  # datas = img.getdata()
  # mask_1D = mask.flatten() 
  # index = 0
  # newData = []
  # for item in datas:
  #     if item[0] == 0 and item[1] == 0 and item[2] == 0 and mask_1D[index] == 0:
  #         newData.append((255, 255, 255, 0))
  #     else:
  #         newData.append(item)
  #     index = index + 1

  # img.putdata(newData)
  # img.save(path)
   img = ctk.CTkImage(Image.open(path), size=(200, 200))
   label2 = ctk.CTkLabel(grabcut_textbox,text = "", image=img)
   label2.place(x=0, y=0)
def Stylize(stylize_textbox):
        ITERATIONS=200
        current_dir = os.path.dirname(os.path.abspath(__file__))
        OUTPUT_DIR = current_dir+'/images/output/'
        filename = filedialog.askopenfilename() 
        print(filename)
        
        Image.open(filename).save(current_dir + "\images\style.jpg")
        
        simg = ctk.CTkImage(Image.open(current_dir + "\images\style.jpg"), size=(200, 200))
        label3 = ctk.CTkLabel(stylize_textbox,text = "", image=simg)
        label3.place(x=0, y=0)

        start_time = time.time()
        with tf.device("/gpu:0"):
            with tf.compat.v1.Session() as sess:
                image = Image.open(os.getcwd() + '\images\extracted.png')
            
                IMAGE_WIDTH = image.size[0]
                IMAGE_HEIGHT = image.size[1]
                # Load images.
                content_image = utility.load_image(os.getcwd() + '\images\extracted.png',
                                                   200, 200, invert=1)

                style_image = utility.load_image(os.getcwd() + '\images\style.jpg', 200,
                                                 200, invert=1)
                # utility.save_image(OUTPUT_DIR+"/"+style_name+".jpg", style_image, invert = style_invert)

                # Load the model.
                model = load_vgg_model(VGG_MODEL, 200, 200, 3)
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

                optimizer = tf.compat.v1.train.AdamOptimizer(1.0)
                train_step = optimizer.minimize(total_loss)

                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(model['input'].assign(initial_image))
                for it in range(ITERATIONS + 1):
                    sess.run(train_step)

                    if it % 100 == 0:
                        # Print every 10 iteration.
                        mixed_image = sess.run(model['input'])
                      #  self.plainTextEdit.appendPlainText('Stylize the Content Image')
                      #  self.plainTextEdit.appendPlainText('Iteration %d' % (it))
                      #  self.plainTextEdit.appendPlainText('sum         : ' + str(sess.run(tf.reduce_sum(mixed_image))))
                      #  self.plainTextEdit.appendPlainText('total_loss  : ' + str(sess.run(total_loss)))
                      #  self.plainTextEdit.appendPlainText("content_loss: " + str(alpha * sess.run(content_loss)))
                      #  self.plainTextEdit.appendPlainText("style_loss  : " + str(beta * sess.run(style_loss)))
                      #  self.plainTextEdit.appendPlainText("shape loss  : " + str(gamma * sess.run(shape_loss)))

                        if not os.path.exists(OUTPUT_DIR):
                            os.mkdir(OUTPUT_DIR)
                        filename = OUTPUT_DIR + '/%d.jpg' % (it)
                        utility.save_image(filename, mixed_image, invert=result_invert)
                    if sess.run(total_loss) < 1:
                        break
            sess.close()
        end_time = time.time()
        #self.plainTextEdit.appendPlainText("Time taken = " + str(end_time - start_time))
        #self.plainTextEdit.appendPlainText("4 Stylized the extracted content image")
        obj = OUTPUT_DIR + '/200.jpg'
        im = cv2.imread(os.getcwd() + '/images/original.jpg')
        im = cv2.resize(im, (200, 200))
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
        path =  os.getcwd() + '/images/result.jpg'
        cv2.imwrite(path, clone)
        resimg = ctk.CTkImage(Image.open(path), size=(200, 200))
        label3 = ctk.CTkLabel(stylize_textbox,text = "", image=resimg)
        label3.place(x=0, y=0)
        
        #self.label_3.setPixmap(QtGui.QPixmap(OUTPUT_DIR + "\merged.jpg"))
        #self.stackedWidget.setCurrentIndex(0)  # 打开 stackedWidget > page_0    
    
def create_ui_components(root):
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root.title("Cloths Stylization Assistant")
    root.configure(bg='#252422')
    root.geometry("680x310")
    root.resizable(width=False, height=False)
    font_size = 20

    upload_textbox = ctk.CTkTextbox(root, width=200, font=("Arial", font_size), text_color='#639cdc', wrap="word")
    #upload_textbox.grid(row=0, column=0, padx=10, pady=20, sticky="nsew")
    upload_textbox.place(x=20, y=20)
    grabcut_textbox = ctk.CTkTextbox(root, width=200, font=("Arial", font_size), text_color='#639cdc', wrap="word")
    #grabcut_textbox.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")
    grabcut_textbox.place(x=240, y=20)
    stylize_textbox = ctk.CTkTextbox(root, width=200, font=("Arial", font_size), text_color='#639cdc', wrap="word")
    #stylize_textbox.grid(row=0, column=2, padx=10, pady=20, sticky="nsew")
    stylize_textbox.place(x=460, y=20)
    upload_button = ctk.CTkButton(root, width=200, text="Upload Image", command=lambda: UploadAction(upload_textbox))
    #upload_button.grid(row=1, column=0, padx=10, pady=1, sticky="nsew")
    upload_button.place(x=20, y=240)
    grabcut_button = ctk.CTkButton(root, width=200, text="GrabCut", command=lambda: GrabCut(grabcut_textbox))
    #grabcut_button.grid(row=1, column=1, padx=10, pady=1, sticky="nsew")
    grabcut_button.place(x=240, y=240)
    stylize_button = ctk.CTkButton(root, width=200, text="Stylize", command=lambda: Stylize(stylize_textbox))
    #stylize_button.grid(row=1, column=2, padx=10, pady=1, sticky="nsew")
    stylize_button.place(x=460, y=240)

def UploadAction(upload_textbox): 
    current_dir = current_directory = os.path.dirname(os.path.abspath(__file__))
    filename = filedialog.askopenfilename() 
    img = ctk.CTkImage(Image.open(filename), size=(200, 200))    
    label = ctk.CTkLabel(upload_textbox,text = "", image=img)
    label.place(x=0, y=0)
    im = Image.open(filename)
    im.resize((200, 200))
    im.save(current_dir + "\images\original.jpg")
def main():

    root = ctk.CTk()
    create_ui_components(root)


    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
   # root.grid_rowconfigure(2, weight=1)
    #root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

     # Add the clear transcript button to the UI
   # clear_transcript_button = ctk.CTkButton(root, text="Upload Image", command=lambda: clear_context(transcriber, audio_queue, ))
   # clear_transcript_button.grid(row=1, column=0, padx=10, pady=3, sticky="nsew")

    #freeze_state = [False]  # Using list to be able to change its content inside inner functions
    #def freeze_unfreeze():
    #    freeze_state[0] = not freeze_state[0]  # Invert the freeze state
    #    freeze_button.configure(text="Unfreeze" if freeze_state[0] else "Freeze")

   # freeze_button.configure(command=freeze_unfreeze)
    root.mainloop()

if __name__ == "__main__":
    main()