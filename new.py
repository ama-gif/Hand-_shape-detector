import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the area of a polygon
def calculate_polygon_area(points):
    points = np.array(points)
    hull = cv2.convexHull(points)
    area = cv2.contourArea(hull)
    return area

# Function to calculate the perimeter of a polygon
def calculate_perimeter(points):
    perimeter = 0
    for i in range(len(points)):
        perimeter += calculate_distance(points[i], points[(i + 1) % len(points)])
    return perimeter

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to process the hand tracking and shape calculation
def process_frame(frame):
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # List of key finger points to track (index, middle, ring, pinky tips)
    key_landmarks = [8, 12, 16, 20]  # Fingertips
    shape_points = []
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw all hand landmarks
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract key finger points
            h, w, c = frame.shape
            for landmark_id in key_landmarks:
                lm = landmarks.landmark[landmark_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                shape_points.append((cx, cy))
                # Draw larger circles at fingertips
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
            
            # Draw connecting lines to form a shape
            for i in range(len(shape_points)):
                cv2.line(frame, shape_points[i], shape_points[(i + 1) % len(shape_points)], 
                         (0, 255, 0), 2)
            
            # Calculate and display shape dimensions
            if len(shape_points) >= 3:  # Need at least 3 points for a shape
                # Calculate area
                area = calculate_polygon_area(shape_points)
                
                # Calculate perimeter
                perimeter = calculate_perimeter(shape_points)
                
                # Calculate dimensions (using bounding box)
                x_coords = [p[0] for p in shape_points]
                y_coords = [p[1] for p in shape_points]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                # Display the information
                cv2.putText(frame, f"Area: {area:.2f} sq.px", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Perimeter: {perimeter:.2f} px", (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Width: {width} px", (30, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Height: {height} px", (30, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add this information as text on the frame
                cv2.putText(frame, "Saving frames to output folder", (frame.shape[1] - 300, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame, shape_points

# Main function - completely headless, no GUI functions
def main():
    print("Starting headless hand shape detector...")
    print("Press Ctrl+C to stop the program")
    
    # Check if 'output' directory exists, if not create it
    if not os.path.exists('output'):
        os.makedirs('output')
        print("Created 'output' directory")
    
    try:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam. Check if it's properly connected.")
            return

        frame_count = 0
        save_interval = 10  # Save every 10 frames
        
        print(f"Capturing frames and saving to 'output' folder every {save_interval} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break

            # Process the frame
            processed_frame, shape_points = process_frame(frame)
            
            # Save frames periodically
            frame_count += 1
            if frame_count % save_interval == 0:
                timestamp = int(time.time())
                filename = f'output/frame_{timestamp}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"Saved frame as {filename}")
                
                # Save measurements if a shape is detected
                if shape_points:
                    area = calculate_polygon_area(shape_points)
                    perimeter = calculate_perimeter(shape_points)
                    x_coords = [p[0] for p in shape_points]
                    y_coords = [p[1] for p in shape_points]
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    
                    print(f"  Area: {area:.2f} sq.px")
                    print(f"  Perimeter: {perimeter:.2f} px")
                    print(f"  Width: {width} px")
                    print(f"  Height: {height} px")
                    
                    with open(f'output/measurements_{timestamp}.txt', 'w') as f:
                        f.write(f"Area: {area:.2f} sq.px\n")
                        f.write(f"Perimeter: {perimeter:.2f} px\n")
                        f.write(f"Width: {width} px\n")
                        f.write(f"Height: {height} px\n")
                        f.write(f"Shape Points: {shape_points}\n")
            
            # Small delay to reduce CPU usage
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nProgram stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release resources
        if 'cap' in locals() and cap is not None:
            cap.release()
        print("Program ended")

if __name__ == "__main__":
    main()