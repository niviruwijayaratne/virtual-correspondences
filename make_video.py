import os
import cv2

image_dir = 'outputs'
video_name = 'two_corresp.mp4'

images = [img for img in os.listdir(image_dir)]
frame = cv2.imread(os.path.join(image_dir, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
for image in images:
    video.write(cv2.imread(os.path.join(image_dir, image)))
    
cv2.destroyAllWindows()
video.release()