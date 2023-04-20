import cv2

cap = cv2.VideoCapture(0) # open the default camera
count = 0 # initialize counter to zero

while True:
    ret, frame = cap.read() # read a frame from the camera
    
    # Display the count on the frame
    text = 'Samples taken: ' + str(count)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame) # show the frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait for a key event
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'): # take a sample image on 's' key press
        ret, sample_frame = cap.read() # read a sample frame from the camera
        cv2.imwrite('sample_image_' + str(count) + '.jpg', sample_frame) # save the sample image
        count += 1 # increment the counter

cap.release() # release the camera
cv2.destroyAllWindows() # close all windows
