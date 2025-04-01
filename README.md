# Hand-_shape-detector
Hand Shape Detector
A Python application that tracks hand movements using your webcam and calculates the dimensions of shapes formed by your fingertips in real-time.
Features

Hand Tracking: Uses MediaPipe to detect and track hand landmarks in real-time
Shape Formation: Creates a shape by connecting the tips of your index, middle, ring, and pinky fingers
Measurement Calculation: Computes area, perimeter, width, and height of the formed shape
Headless Operation: Works without GUI dependencies, ideal for servers or systems without display support
Frame Saving: Automatically saves processed frames and measurements to an output directory
Low Resource Usage: Optimized for minimal CPU consumption

Requirements

Python 
OpenCV (cv2)
NumPy
MediaPipe
Output


The program creates an output directory (if it doesn't exist already) where it saves:

Images: Processed frames showing the hand tracking, shape, and measurements
Measurement Files: Text files containing the precise measurements for each saved frame:

Area (in square pixels)
Perimeter (in pixels)
Width (in pixels)
Height (in pixels)
Coordinate points of the shape vertices



How It Works

The program captures video from your webcam
MediaPipe's hand tracking model identifies hand landmarks
The tips of the index, middle, ring, and pinky fingers are highlighted
These fingertips are connected to form a quadrilateral shape
The area, perimeter, width, and height of this shape are calculated
Every 10 frames, the program saves both the visual frame and measurement data

Customization
You can modify these parameters in the code:

save_interval: Change how frequently frames are saved (default: every 10 frames)
key_landmarks: Adjust which finger points are tracked (default: tips of index, middle, ring, and pinky)
min_detection_confidence and min_tracking_confidence: Adjust hand detection sensitivity

Troubleshooting

Webcam Issues: If the program cannot detect your webcam, ensure it's properly connected and not being used by another application
Performance: If the program runs slowly, try increasing the sleep time or the save interval
Hand Detection: For better tracking, ensure good lighting and keep your hand clearly visible in the camera frame
