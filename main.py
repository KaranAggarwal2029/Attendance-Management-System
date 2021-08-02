import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = './Data'
images = []
classNames = []
templi = []
myList = os.listdir(path)
for cl in myList:
    imgsPath = os.listdir(path+ '\\' + cl)
    templi = []
    for imgp in imgsPath:
        img = cv2.imread(path+ '\\' + cl + '\\' + imgp)
        img = cv2.resize(img,(0,0),None,0.25,0.25)
        loc = face_recognition.face_locations(img)
        if len(loc) ==0:
            print(cl +'   ' + imgp)
            print("Face not Found")
            continue
        templi.append(img)
    images.append(templi)
    classNames.append(cl)

print(classNames)
# print(images[0][0].shape)

#Already Mrked attendance
nameList = []
with open('Attendance.csv', 'r') as f:
    myDataList = f.readlines()
    for line in myDataList:
        entry = line.split(',')
        nameList.append(entry[0])

def findEncodings(images):
    encodeList = []
    for same_person_img in images:
        tempencodelist = []
        for img2 in same_person_img:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            loc = face_recognition.face_locations(img2)
            encode = face_recognition.face_encodings(img2,loc)[0]
            tempencodelist.append(encode)
        encodeList.append(tempencodelist)
    return encodeList


def markAttendance(name):
    if name not in nameList:
        with open('Attendance.csv', 'a') as f:
            nameList.append(name)
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')

encodeListKnown = findEncodings(images)
print("Encoding Done")

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        minlist = []
        matches = []
        for encodeList in encodeListKnown:
            matchestemp = face_recognition.compare_faces(encodeList, encodeFace)
            faceDis = face_recognition.face_distance(encodeList, encodeFace)
            matches.append(matchestemp[np.argmin(faceDis)])
            minlist.append(faceDis[np.argmin(faceDis)])

        matchIndex = np.argmin(minlist)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            if minlist[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
