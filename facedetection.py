import cv2
import face_recognition
import os

# --- INITIALIZING MY DATABASE ---
# I created these lists to store the face data and the names of the people
known_face_encodings = []
known_face_names = []

# This is the folder where I kept the images for training
dataset_path = "face_dataset"

# --- STEP 1: PROCESSING THE DATASET ---
# I am looping through every image in my 'face_dataset' folder
for file in os.listdir(dataset_path):
    # Loading the image file into the program
    image = face_recognition.load_image_file(f"{dataset_path}/{file}")
    
    # This is the "Feature Extraction" part. It turns a face into a 128-point math vector.
    encodings = face_recognition.face_encodings(image)

    # I only add the encoding if the program actually found a face in the image
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        # I used split() to get the name from the filename (e.g., 'Rishuraj.jpg' becomes 'Rishuraj')
        known_face_names.append(file.split(".")[0])

print("Faces Loaded Successfully")

# --- STEP 2: STARTING THE WEBCAM ---
# Opening the camera (0 is my laptop's default webcam)
video = cv2.VideoCapture(0)

while True:
    # Reading the live frame from the camera
    ret, frame = video.read()
    if not ret:
        break

    # OpenCV uses BGR, but face_recognition library needs RGB, so I have to convert it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Finding the location of faces in the live camera frame
    face_locations = face_recognition.face_locations(rgb_frame)
    # Creating encodings for the faces found in the live frame to compare them later
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # --- STEP 3: MATCHING AND LABELING ---
    # Using a zip loop to go through locations and encodings at the same time
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        name = "Unknown" # Default if no match is found
        
        # This function compares the live face vector to my known face database
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match is found (True), I find the index and get the person's name
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # --- STEP 4: DRAWING THE OUTPUT ---
        # Drawing a green box (0, 255, 0) around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Putting the name text above the box
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Showing the final window with the recognition boxes
    cv2.imshow("Face Recognition", frame)

    # If I press the 'ESC' key (27), the program stops
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- STEP 5: CLEANING UP ---
# Releasing the camera and closing the windows
video.release()
cv2.destroyAllWindows()
