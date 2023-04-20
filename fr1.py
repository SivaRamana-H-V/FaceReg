import tkinter as tk
from PIL import ImageTk, Image
from tkinter import *
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path

window = tk.Tk()
window.state('zoomed')
window.geometry("1366x768")

bg1 = ImageTk.PhotoImage(file = "3.jpg")
label1 = tk.Label(window, image = bg1)
label1.place(x=-2,y=-2)

window.title("Face_Recogniser")
window.configure(background ='white')
window.rowconfigure(0, weight = 1)
window.columnconfigure(0, weight = 1)

message = tk.Label(
    window, text ="""
KiTE Staffs
Department of AI&DS
""",
    bg ="blue", fg = "white", width = 30,
    height = 3, font = ('Times New Roman', 30, 'bold'))

message.place(x=330, y=30)

df = pd.read_csv(r"UserDetails\UserDetails.csv") 

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Reading the trained model
recognizer.read("TrainingImageLabel\Trainer.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)
# getting the name from "userdetails.csv"
df = pd.read_csv(r"UserDetails\UserDetails.csv") 
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX       
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)   
    for(x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])                                  
        if(conf < 50):
            aa = df.loc[df['Id'] == Id]['Name'].values
            tt = str(Id)+"-"+aa   
        else:
            Id ='Unknown'               
            tt = str(Id) 
        if(conf > 75):
            noOfFile = len(os.listdir("ImagesUnknown"))+1
            cv2.imwrite("ImagesUnknown\Image"+
            str(noOfFile) + ".jpg", im[y:y + h, x:x + w])           
        cv2.putText(im, str(tt), (x, y + h),
        font, 1, (255, 255, 255), 2)       
    cv2.imshow('im', im)
    if (cv2.waitKey(1)== ord('q')):
        break
cam.release()
cv2.destroyAllWindows()

window.mainloop()    
   
