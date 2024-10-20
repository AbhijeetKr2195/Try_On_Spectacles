import cv2
import numpy as np

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

sunglasses_image = 'sunglasses.png'

# Load the sunglasses image with alpha channel
sunglasses = cv2.imread(sunglasses_image, cv2.IMREAD_UNCHANGED)

# Store the last known eye positions and angle
last_eye_centers = None
last_angle = 0

def overlay_transparent(background, overlay, x, y):
    """Overlay a transparent image on top of a background image."""
    h, w = overlay.shape[:2]

    # Ensure the overlay doesn't exceed the frame dimensions
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    # Separate RGB and alpha channels
    overlay_rgb = overlay[..., :3]
    overlay_alpha = overlay[..., 3] / 255.0
    inv_alpha = 1.0 - overlay_alpha

    roi = background[y:y+h, x:x+w]

    for c in range(3):
        roi[..., c] = (overlay_alpha * overlay_rgb[..., c] +
                       inv_alpha * roi[..., c])

    background[y:y+h, x:x+w] = roi
    return background

def rotate_image(image, angle):
    """Rotate the image around its center by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def get_eye_centers(eyes):
    """Ensure the two largest detected eyes are ordered as left and right."""
    eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x-coordinate
    ex1, ey1, ew1, eh1 = eyes[0]
    ex2, ey2, ew2, eh2 = eyes[1]

    left_eye_center = (ex1 + ew1 // 2, ey1 + eh1 // 2)
    right_eye_center = (ex2 + ew2 // 2, ey2 + eh2 // 2)

    return left_eye_center, right_eye_center

def calculate_angle(left_eye, right_eye):
    """Calculate the angle between the eyes."""
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    
    angle = -np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

            if len(eyes) >= 2:
                # Get the two largest eyes and calculate new centers and angle
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                left_eye, right_eye = get_eye_centers(eyes)
                angle = calculate_angle(left_eye, right_eye)

                # Update the last known positions and angle
                last_eye_centers = (left_eye, right_eye)
                last_angle = angle
            elif last_eye_centers:
                # Use the last known positions if eyes aren't detected
                left_eye, right_eye = last_eye_centers
                angle = last_angle

            # Calculate the size of the sunglasses based on the eye distance
            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            sunglass_width = int(2.2 * eye_distance)
            sunglass_height = int(sunglass_width * sunglasses.shape[0] / sunglasses.shape[1])

            # Rotate the sunglasses to match the angle between the eyes
            rotated_sunglasses = rotate_image(sunglasses, angle)

            # Calculate the overlay position
            sunglass_x = x + left_eye[0] - sunglass_width // 4
            sunglass_y = y + left_eye[1] - sunglass_height // 2

            # Resize the rotated sunglasses to fit the detected width
            rotated_sunglasses = cv2.resize(
                rotated_sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_AREA
            )

            # Overlay the sunglasses on the frame
            frame = overlay_transparent(
                frame, rotated_sunglasses, sunglass_x, sunglass_y
            )

    elif last_eye_centers:
        # If no face is detected, maintain the last known positions
        left_eye, right_eye = last_eye_centers
        angle = last_angle

        # Calculate the size of the sunglasses based on the eye distance
        eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        sunglass_width = int(2.2 * eye_distance)
        sunglass_height = int(sunglass_width * sunglasses.shape[0] / sunglasses.shape[1])

        # Rotate and resize the sunglasses
        rotated_sunglasses = rotate_image(sunglasses, angle)
        rotated_sunglasses = cv2.resize(
            rotated_sunglasses, (sunglass_width, sunglass_height), interpolation=cv2.INTER_AREA
        )

        # Overlay the sunglasses using the last known coordinates
        sunglass_x = left_eye[0] - sunglass_width // 4
        sunglass_y = left_eye[1] - sunglass_height // 2

        frame = overlay_transparent(
            frame, rotated_sunglasses, sunglass_x, sunglass_y
        )

    # Show the result
    cv2.imshow('Sunglass Try-On', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
