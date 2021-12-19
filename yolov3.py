from __future__ import division

from models import Darknet
from utils.utils import *


from PIL import Image

import torch
#from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class yolov3:
    def __init__(self, cfg_path, classes_path, weights_path, img_sz=416, conf_thres = 0.8, nms_thres = 0.4):
        self.image_size = img_sz
        self.conf_thres = conf_thres
        self.nms_thres = 0.4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Darknet(cfg_path, img_size=img_sz).to(self.device)

        self.model.load_darknet_weights(weights_path)    #takes only weights files

        self.model.eval()    #model set to evaluation mode

        with open(classes_path, "r") as class_file:
            self.classes = class_file.read().split("\n")[:-1]    #class file has empty line ending 

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def __call__(self, input_image, save_image_plot = False):    #input_image is either string of path or PIL image

        #Imports and converts image for detection to pytorch tensor

        #if input_image is path string
        image = None
        if type(input_image) == str:
            image = Image.open(input_image)
        else:
            image = input_image

        #scales dimensions by the smaller of the width and height that scales to 416
        scaled_h = int(image.size[0] * min(self.image_size / image.size[0], self.image_size / image.size[1]))
        scaled_w = int(image.size[1] * min(self.image_size / image.size[0], self.image_size / image.size[1]))

        image = image.resize((scaled_h,scaled_w), Image.NEAREST)

        #square image of self.image_size by self.image_size to be inputted
        square_image = Image.new("RGB",(self.image_size,self.image_size))
        square_image.paste(image, ((self.image_size-scaled_h)//2,(self.image_size-scaled_w)//2))

        image = transforms.functional.to_tensor(square_image)   

        image = image.unsqueeze(0)
        image = Variable(image.type(self.Tensor))

        # Get detections
        with torch.no_grad():
            self.detections = self.model(image)
            self.detections = non_max_suppression(self.detections, self.conf_thres, self.nms_thres)
            self.detections = self.detections[0]


        #MOVE TO DEDICATED PLOT FUNCTION

        # Create plot           
        img = None
        image_path = None
        if type(input_image) == str:
            img = np.array(Image.open(input_image))
            image_path = input_image
        else:
            img = np.array(input_image)
            image_path = "output"

        if self.detections is not None:
            
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            
            # Draw bounding boxes and labels of detection
            # Rescale boxes to original image
            

            detections = rescale_boxes(self.detections, self.image_size, img.shape[:2])
            color = [1,1,1,1]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                box_w = x2 - x1
                box_h = y2 - y1

                
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=self.classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                )

            if save_image_plot: 
                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                filename = image_path.split("/")[-1].split("\\")[-1].split(".")[0]
                plt.savefig(f"{filename}.png",dpi=300, bbox_inches="tight", pad_inches=0.0)
            
            
            plt.close('all')
            
            return self.detections.tolist()   #python list in form x1, y1, x2, y2


    def plot_detection(self, image, show_image = False):
        if self.detections == None:
            print("No detections")
            

        