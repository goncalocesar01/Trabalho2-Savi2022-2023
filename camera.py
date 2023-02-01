import cv2
import mediapipe as mp
from gtts import gTTS
from statistics import mean
import torch
import glob
import os
from tqdm import tqdm
from model import Model
from dataset import Dataset
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from classification_visualizer import ClassificationVisualizer
import ctypes
def main():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    while True:
        success, image = cap.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
            # checking whether a hand is detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: # working with each hand
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8 :
                        bbox = cv2.rectangle(image,(cx,cy),(cx+200,cy+200),(255,0,0),3)
                        roi = image[cy:(cy+200), cx:(cx+200)]
                        if  cv2.waitKey(50) == ord('p'):
                            name_input = 'apple_1_1' + ".png"
                            cv2.imwrite('./imagescamera/' + name_input, roi)
                            model_path = 'model.pkl'
                            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # cuda: 0 index of gpu
                            model = Model()  # Instantiate model
                            loss_function = torch.nn.CrossEntropyLoss()
                            model.to(device)
                                # datasets
                            path = './imagescamera'
                            image_filenames = glob.glob(path + '/*.png')
                            dataset_test = Dataset(image_filenames)
                            loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)
                            class_name = ['apple','ball','banana','bellpepper','binder','bowl','calculator','camera','cap','cellphone','cerealbox','coffeemug','comb',
                            'drybattery', 'flashlight','foodbag','foodbox','foodcan','foodjar','foodcup','garlic','gluestick','greens','handtowel','instantnoodle','keyboard',
                            'kleenex','lemon','lightbulb','lime','marker','mushroom','notebook','onion','orange','peach','pear','pitcher','plate','pliers','potato',
                            'rubbereraser','scissors','shampoo','sodacan','sponge','tomato','stapler','toothbrush','toothpaste','waterbottle']
                            test_visualizer = ClassificationVisualizer('Test Images')
                            checkpoint = torch.load(model_path)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            idx_epoch = checkpoint['epoch']
                            model.to(device)  # move the model variable to the gpu if one exists 
                            test_losses = []
                            for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test),
                                                                        desc=Fore.GREEN + 'Testing batches for Epoch ' + str(idx_epoch) + Style.RESET_ALL):
                                image_t = image_t[:,:3,:,:]
                                image_t = image_t.to(device)
                                label_t = label_t.to(device)    
                                    # Apply the network to get the predicted ys and show
                                label_t_predicted = model.forward(image_t)
                                names = test_visualizer.draw(image_t, label_t, label_t_predicted,class_name, False) 
                                print(names)
                                # the image to the classifier
                                #im = Image.fromarray(object['crop'])
                                first_time=True
                                for name in names:
                                    if first_time==True:
                                        text_to_speach = 'The object is a' + name
                                        first_time= False
                                tts = gTTS(text=text_to_speach, lang='en')
                                filename = "./imagescamera/hello.mp3"
                                tts.save(filename)
                                os.system(f"start {filename}")
                                handle = ctypes.windll.user32.FindWindowW("WMPlayerApp", None)
                                cv2.waitKey(5000)
                                ctypes.windll.user32.PostMessageW(handle, 0x0112, 0xF060, 2)
                    mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
        cv2.imshow("Output", image)  
        if cv2.waitKey(1) == ord('o'):
            break
if __name__ == "__main__":
    main()
