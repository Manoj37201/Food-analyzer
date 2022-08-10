import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import random
import os
import sys
from PIL import Image, ImageTk

window = tk.Tk()

window.title("FOOD CLASSIFICATION - FRUITS & VEGETABLES")

window.geometry("650x750")
window.configure(background ="white")

title = tk.Label(text="Click to start analysis and classification", background = "white", fg="green", font=("times", 15, "bold" ))
##title.grid()
title.grid(column=2, row=0, padx=10, pady=10)
def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks.
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'fruit.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')
    def stages():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            #definig the range of red color
            red_lower=np.array([0,0,212],np.uint8)
            red_upper=np.array([131,255,255],np.uint8)
            

            red=cv2.inRange(hsv, red_lower, red_upper)
            kernal = np.ones((5 ,5), "uint8")
            red=cv2.dilate(red, kernal)
            res=cv2.bitwise_and(img, img, mask = red)
            #Tracking the Red Color
            contours,hierarchy =cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            count=0
            
            
            for cnt in contours:
                value = cnt[0,0,0]
                pyval = value.item()
                global sum1
                sum1=0
                sum1=sum1+pyval 
##            print(sum1)
            print("The calorie in the predicted food item is {}%".format(sum1))
##                print(type(cnt[0,0,0]))


    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    #tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 12, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
##        pred= random.randint(90,98)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
##        print(model_out)
        print('model {}'.format(np.argmax(model_out)))



        #if model_out > 0.5

        if np.argmax(model_out) == 0:
            stages()
            print("The predicted image of the Avocado is with a accuracy of {} %".format(model_out[0]))
            str_label = 'Avocado'
        elif np.argmax(model_out) == 1:
            stages()
            print("The predicted image of the Banana is with a accuracy of {} %".format(model_out[1]))
            str_label = 'Banana'
        elif np.argmax(model_out) == 2:
            stages()
            print("The predicted image of the Beetroot is with a accuracy of {} %".format(model_out[2]))
            str_label = 'Beetroot'
        elif np.argmax(model_out) == 3:
            stages()
            print("The predicted image of the Blueberry is with a accuracy of {} %".format(model_out[3]))
            str_label = 'Blueberry'
        elif np.argmax(model_out) == 4:
            stages()
            print("The predicted image of the cactusfruit is with a accuracy of {} %".format(model_out[4]))
            str_label = 'cactusfruit'
        elif np.argmax(model_out) == 5:
            stages()
            print("The predicted image of the cauliflower is with a accuracy of {} %".format(model_out[5]))
            str_label = 'cauliflower'
        elif np.argmax(model_out) == 6:
            stages()
            print("The predicted image of the chestnut is with a accuracy of {} %".format(model_out[6]))
            str_label = 'chestnut'
        elif np.argmax(model_out) == 7:
            stages()
            print("The predicted image of the corn is with a accuracy of {} %".format(model_out[7]))
            str_label = 'corn'
        elif np.argmax(model_out) == 8:
            stages()
            print("The predicted image of the eggplant is with a accuracy of {} %".format(model_out[8]))
            str_label = 'eggplant'
        elif np.argmax(model_out) == 9:
            stages()
            print("The predicted image of the onion is with a accuracy of {} %".format(model_out[9]))
            str_label = 'onion'
        elif np.argmax(model_out) == 10:
            stages()
            print("The predicted image of the orange is with a accuracy of {} %".format(model_out[10]))
            str_label = 'orange'
        elif np.argmax(model_out) == 11:
            stages()
            print("The predicted image of the potato is with a accuracy of {} %".format(model_out[11]))
            str_label = 'potato'
  
        if str_label == 'Avocado':
            status = "Avocado"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10)          
            predss="Predicted image of Avocado is  {} % accuracy".format(model_out[0]*100)
            calo="The Nutritional value per Avocado:\nEnergy: 114 cal\nDietary fiber: 6g\nTotal sugar: 0.2g\nPotassium: 345mg\nSodium: 5.5mg\nMagnesium: 19.5mg\nVitamin A: 43μg\nVitamin E: 1.3mg\nVitamin K: 14μg\nVitamin B-6: 0.2mg\nMonounsaturated fatty acids: 6.7g"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)

            
        elif str_label == 'Banana':
            status = "Banana"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of Banana is  {} % accuracy".format(model_out[4]*100) 
            calo="Nutritional values per average Banana:\nEnergy:105 cal\nCarbohydrates:27 g\nFiber: 3.1g\nProtein: 1.3g\nMagnesium: 31.9mg\nPhosphorus: 26mg\nPotassium: 422mg\nSelenium: 1.9mcg\nCholine: 11.6mg\nVitamin C: 10.3mg\nFolate: 23.6mcg\nBeta carotene: 30.7mcg\nAlpha carotene: 29.5mcg"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
            
        elif str_label == 'Beetroot':
            status="Beetroot"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of Beetroot is  {} % accuracy".format(model_out[1]*100)
            calo="Nutritional Values per 100g:\nEnergy: 43 cal\nCarbohydrates: 10g\nProtein: 1.6g\nVitamin C, Iron, Vitamin B-6, Magnesium, Calcium"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'Blueberry':
            status = "Blueberry"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  Blueberry is  {} % accuracy".format(model_out[2]*100)
            calo="Nutritional Values per 80g:\nEnergy: 32Kcal/135KJ\nCarbohydrates: 7.3g\nProtein: 0.7g\nFibre: 1.2g\nPotassium: 53mg\nVitamin E: 0.75mg\nVitamin C: 5mg"            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'cactusfruit':
            status = "cactusfruit"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  cactusfruit is  {} % accuracy".format(model_out[3]*100)
            calo="Nutritional Values per fruit:\nEnergy: 42g\nCarbohydrates: 10g\nProtein: 1g\nFat: 0.5g\nFiber: 4g\nCholesterol: 0mg\nSodium: 5mg"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'cauliflower':
            status = "cauliflower"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  cauliflower is  {} % accuracy".format(model_out[5]*100)
            calo="Nutritional Values per serving:\nEnergy: 25 cal\nCarbohydrates: 5g\nFat: 0g\nDietary fiber: 2g\nSugar: 2g\nProtein: 2g\nSodium: 30mg"            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'chestnut':
            status = "chestnut"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  chestnut is  {} % accuracy".format(model_out[6]*100)
            calo="Nutritional Values per 100g:\nEnergy: 131 cal\nCarbohydrates: 28g\nProtein: 2g\nTotal Fat: 1.4g\nCholesterol: 0mg\nSodium: 27mg\nPotassium: 715mg"            
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'corn':
            status = "corn"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of corn is  {} % accuracy".format(model_out[7]*100)
            calo="Nutritional Values per ear:\nEnergy: 90 cal\nCarbohydrates: 19g\nProtein: 3g\nFat: 1g\nFiber: 1g\nSugars: 5 g\nVitamin C: 3.6mg"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'eggplant':
            status = "eggplant"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of eggplant is  {} % accuracy".format(model_out[8]*100)
            calo="Nutritional Values per 100g:\nEnergy: 25 cal\nCarbohydrates: 6g\nProtein: 1g\nTotal Fat: 0.2g\nCholesterol: 0mg\nSodium: 2mg\nPotassium: 229mg"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'onion':
            status = "onion"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of onion is  {} % accuracy".format(model_out[9]*100)
            calo="Nutritional Values per 100g:\nEnergy: 40 cal\nCarbohydrates: 9g\nProtein: 1.1g\nTotal Fat: 0.1g\nCholesterol: 0mg\nSodium: 4 mg\nPotassium: 146mg"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'orange':
            status = "orange"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  orange is  {} % accuracy".format(model_out[10]*100)
            calo="Nutritional Values per 100g:\nEnergy: 40 cal\nCarbohydrates: 12g\nProtein: 0.9g\nTotal Fat: 0.1g\nCholesterol: 0mg\nSodium: 0mg\nPotassium: 181mg"          
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
            
        elif str_label == 'potato':
            status = "potato"
            button2.destroy()
            r = tk.Label(text='STATUS: ' + status, background="cyan", fg="red", font=("", 15))
            r.grid(column=0, row=3, padx=10, pady=10) 
            predss="Predicted image of  potato is  {} % accuracy".format(model_out[11]*100)
            calo="Nutritional Values per 100g:\nEnergy: 77 cal\nCarbohydrates: 17g\nProtein: 2g\nTotal Fat: 0.1 g\nCholesterol: 0mg\nSodium: 6mg\nPotassium: 421mg"           
            calo = tk.Label(text=calo, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=4, padx=10, pady=10)            
            calo = tk.Label(text=predss, background="darkcyan", fg="white", font=("", 15))
            calo.grid(column=0, row=5, padx=10, pady=10)            
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=6, padx=10, pady=10)
       

def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\Manoj M\\OneDrive\\Desktop\\food_classification (2)\\food_classification\\food_classification_fruit\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="225", width="450")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    global button2
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=2, row=1, padx=5, pady = 5)



window.mainloop()



