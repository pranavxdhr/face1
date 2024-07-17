import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import ImageGrab
import tkinter as tk
from tkinter import messagebox
from threading import Thread

# Path to the folder containing training images
path = r"C:\Users\Dell\Desktop\face1\Traning_images"
classNames = []
images = []

# List all the files in the training images folder
myList = os.listdir(path)
print("Files in training images folder:", myList)

# Load the images and store their names
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Error: Unable to load image {cl}")

print("Class names:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encode = encodes[0]
            encodeList.append(encode)
    return encodeList

def markAttendance(name):
    file_path = r'C:\Users\Dell\Desktop\face1\Attendance.csv'
    
    # Check if the file exists, if not create it and write the header
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time\n')

    with open(file_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')
            show_popup(name)

def show_popup(name):
    # Show a popup message
    messagebox.showinfo("Attendance", f"Attendance marked for {name}")

def captureScreen(bbox=(300,300,690+300,530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

def webcam_feed():
    global running
    cap = cv2.VideoCapture(0)

    while running:
        success, img = cap.read()
        if not success:
            print("Error: Unable to read from webcam")
            break
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
            else:
                name = "Unknown"
            
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if name != "Unknown":
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_webcam():
    global running
    running = True
    thread = Thread(target=webcam_feed)
    thread.start()

def stop_webcam():
    global running
    running = False
    messagebox.showinfo("Goodbye", "Thank you, next")

# Create the GUI window
root = tk.Tk()
root.title("Webcam Control")

start_button = tk.Button(root, text="Start Webcam", command=start_webcam)
start_button.pack()

stop_button = tk.Button(root, text="Stop Webcam", command=stop_webcam)
stop_button.pack()

root.mainloop()
