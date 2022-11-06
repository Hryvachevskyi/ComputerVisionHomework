# Step 0. Run the necessary imports.
import cv2
import time
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Step 1. Prepare video
videofile = 'Data/cars.mp4'
vcap = cv2.VideoCapture(videofile)

# Step 2. Set up tracker
# Select trecker
tracker_types = ['MIL', 'KCF', 'CSRT']
tracker_type = tracker_types[2]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

# Genrate tracking template
read_success, frame = vcap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
plt.figure()
plt.imshow(frame)
plt.title('First frame')
plt.pause(1)
plt.close()

x1, y1, width, height = 690, 525, 85, 60

obj = frame[y1:y1 + height, x1:x1 + width]
plt.figure()
plt.imshow(obj)
plt.title('Object')
plt.pause(1)
plt.close()

# Initialize tracker
bbox = (x1, y1, width, height)
ok = tracker.init(frame, bbox)

#Step 3. Tracking loop
current_frame_number, frame_count = 1, 1000
while read_success and current_frame_number < frame_count:
    read_success, frame = vcap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or not read_success:
        print('\33[101m' + 'ERROR. Frame read error.' + '\033[0m')
        break

    ok, bbox = tracker.update(frame)
    print(ok, bbox)

    # Show the tracker working
    x1, y1 = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]
    cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
    if ok:
        cv2.putText(frame, 'True\n#{0}'.format(current_frame_number), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, 'True\n#{0}'.format(current_frame_number), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'False\n#{0}'.format(current_frame_number), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, 'False\n#{0}'.format(current_frame_number), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Objec trecing', frame)
    time.sleep(0.2)

    current_frame_number += 1

vcap.release()

"""
# Step 6

Results comparison:

'MIL' – correctly detected the object only for the first few frames. As soon as the object abruptly changed its position (moved to the right by ~50 pixels), the detector lost it. However, until the end of the video, the status of the detector was TRUE, although it tracked "garbage", and in the middle of the video it caught a cloud and tracked it for quite some time.

'KCF' - lost object at the same moment as MIL, but changed its status to FALSE and did not give false positive results until the end of the video. This behavior is better than that of MIL in my opinion.

'CSRT' - This detector tracked the object very well. Adapted to changing the position and size of the object. It is robust to a slight overlap of an object by another object in time (up to ~10 frames). Lost objects when it was overlap by another for ~20+ frames. After losing the object and several frames of false positive detection, it changed its status to FALSE. Best result in my opinion.
"""

