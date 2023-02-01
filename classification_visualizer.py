import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


class ClassificationVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

    def draw(self, inputs, labels, outputs,name, test):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(12,12)
        plt.suptitle(self.title)
        plt.legend(loc='best')
        g=outputs.tolist()
        M=np.array(g)
        j=[]
        for i in range(len(g)):
            gind=g[i]
            k=np.argmax(gind)
            j.append(k)
        inputs = inputs
        batch_size,_,_,_ = list(inputs.shape)

        output_probabilities = F.softmax(outputs, dim=1).tolist()
        output_probabilities_dog = [x[0] for x in output_probabilities]
        if  test==True:
            random_idxs = random.sample(list(range(batch_size)), k=5*5)
            wait= 2
        else:
            random_idxs = list(range(batch_size))
            wait= 0
        objecttype=[]
        for plot_idx, image_idx in enumerate(random_idxs, start=1):
            h=j[image_idx]
            output_probability_dog = output_probabilities_dog[image_idx]

            is_dog = True if output_probability_dog > 0.0196 else False
            #success = True if (label.data.item() == 0 and is_dog) or (label.data.item() == 1 and not is_dog) else False

            image_t = inputs[image_idx,:,:,:]
            image_pil = self.tensor_to_pil_image(image_t)

            ax = self.figure.add_subplot(5,5,plot_idx) # define a 5 x 5 subplot matrix
            plt.imshow(image_pil)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            color = 'green' if is_dog else 'red' 
            title =  name[h]
            #title += ' ' + str(image_idx)
            #title = image_idx.getClassFromFileName()[1]
            
            ax.set_xlabel(title, color=color)
            objecttype.append(title)
        plt.draw()
        key = plt.waitforbuttonpress(wait)
        #if not plt.fignum_exists(3):
        #    print('Terminating')
        #    exit(0)
        plt.close()
        return(objecttype)