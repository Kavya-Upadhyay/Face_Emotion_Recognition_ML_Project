Face Emotion Recognition in Virtual Meetings

This project aims to enhance virtual meetings by detecting emotions in real-time using facial recognition techniques. By leveraging computer vision and deep learning, the program identifies emotions like happiness or sadness from live webcam feeds, making virtual interactions more engaging and responsive.

🔥 Features
   1.Real-Time Emotion Detection: Utilizes a webcam to capture facial expressions in real-time.
   2.Emotion Classification: Identifies emotions such as "Happy" or "Sad" with a confidence score.
   3.User-Friendly Interface: Uses OpenCV to display the captured video with emotion labels for easy visualization.
   4.Lightweight: Relies on pre-trained models from the FER library, making it easy to use without heavy training.

📸 Demo
The application uses your computer's webcam to capture live video and detects emotions on faces using a pre-trained model.


🛠️ Technologies Used
    1.Python: Core programming language.
    2.OpenCV: Library for real-time computer vision.
    3.FER: Python library for Facial Emotion Recognition.

🚀 Code Overview
  The main components of the code are:

  1.Webcam Capture: Captures live video feed using OpenCV.
  2.Face Detection: Uses a pre-trained Haar Cascade classifier to detect faces in the video.
  3.Emotion Detection: Utilizes the FER library to predict emotions on detected faces.
  4.Display: Draws rectangles around detected faces and displays the predicted emotion label.


import cv2
from fer import FER

detector = FER()

video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = video_data[y:y+h, x:x+w]
        emotion, score = detector.top_emotion(face)
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if emotion == 'happy':
            label = f'Happy: {score:.2f}'
        else:
            label = f'Sad: {score:.2f}'
        
        cv2.putText(video_data, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", video_data)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()


⚠️ Known Issues
   1.The accuracy of emotion detection may vary depending on lighting and webcam quality.
   2.Currently, it only classifies "Happy" and "Sad" emotions. Extending to more emotions requires additional model tuning.

🗂️ File Structure
  1.emotion_recognition.py: Main script for emotion recognition.
  2.README.md: Documentation file.
  3.requirements.txt: List of dependencies for the project.


  
