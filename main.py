import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
video_capture = cv2.VideoCapture(0)

# load known faces
srijans_image = face_recognition.load_image_file("faces/srijan.jpeg")
srijans_encoding = face_recognition.face_encodings(srijans_image)

harrys_image = face_recognition.load_image_file("faces/harry.jpeg")
harrys_encoding = face_recognition.face_encodings(harrys_image)

known_face_encodings = [srijans_encoding, harrys_encoding]
known_faces_names = ["Srijan", "Harry"]

# list of expected students
students = known_faces_names.copy()

face_locations = []
face_encodings = []

# get the current date and time

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding) # similarity of the 2 faces
        best_match_index = np.argmin(face_distance)

        if matches(best_match_index):
            name = known_faces_names[best_match_index]

        # add the text if a person is present
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_TRIPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "Present", bottomLeftCornerOfText, fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()

