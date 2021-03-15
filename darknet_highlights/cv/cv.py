import cv2
from tqdm import tqdm
import time
import numpy as np
import pandas as pd


def add_bbox(frame, idxs, boxes, confidences, classIDs, COLORS, LABELS):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return frame

def process_video(video: str, 
                  data_file: str, 
                  config_file: str, 
                  weights: str, 
                  names_file: str,
                  confidence_thresh = 0.5, 
                  thresh = 0.3,
                  output_file = None,
                  net_size = (1920, 1056),  # for maui63 network
                  ):
    
    vidcap = cv2.VideoCapture(video)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    
    framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    
    print('Running YOLO on video:')
    time.sleep(0.5)
    
    net = cv2.dnn.readNetFromDarknet(config_file, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    LABELS = open(names_file).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3))
    
    writer = None
    (W, H) = (None, None)
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    df = pd.DataFrame(columns = ['timestamp', 'num_objects',
                                 'prob', 'name', 'box'])
    
    count = 0
    for framenum in tqdm(range(framecount)):
        
        # Redundant here
        #vidcap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        
        success, frame = vidcap.read()
        
        if not success:
            raise ValueError('Frame number is not valid')
        
        count+=1
        frametime = count/fps
        
        # %% Net
    
        if W == None and H == None:
            (H, W) = frame.shape[:2]
        
        frame, idxs, boxes, confidences, classIDs = run_net_on_frame(
            frame,
            net,
            net_size,
            output_layers,
            confidence_thresh,
            thresh,
            W, H,
            COLORS,
            LABELS)
        
        frame = add_bbox(frame, idxs, boxes, confidences,
                         classIDs, COLORS, LABELS)        
        
        # check if the video writer is None
        if writer is None and output_file != None:
            # initialize our video writer
            # TODO: choose filetype based on output_file extension
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            
            writer = cv2.VideoWriter(output_file, fourcc, fps,
                (frame.shape[1], frame.shape[0]), True)
    
        # write the output frame to disk
        if output_file != None:
            writer.write(frame)
        
        # If an object is detected append the data
        if len(idxs) > 0:
            frame_info = {
                'timestamp': frametime,
                'num_objects': len(idxs),
                'prob': confidences,
                'name': classIDs,
                'box': boxes,
                }
            series = pd.Series(frame_info)
            df = df.append(series, ignore_index=True)
        
    return df


def process_image(image: str, 
                  data_file: str, 
                  config_file: str, 
                  weights: str, 
                  names_file: str,
                  confidence_thresh = 0.5, 
                  thresh = 0.3,
                  output_file = None,
                  net_size = (1920, 1056),  # for maui63 network
                  ):
    
    frame = cv2.imread(image) 
    
    net = cv2.dnn.readNetFromDarknet(config_file, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    LABELS = open(names_file).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3))
        
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    (H, W) = frame.shape[:2]
        
    frame, idxs, boxes, confidences, classIDs = run_net_on_frame(
        frame,
        net,
        net_size,
        output_layers,
        confidence_thresh,
        thresh,
        W, H,
        COLORS,
        LABELS)
    
    frame = add_bbox(frame, idxs, boxes, confidences, classIDs, COLORS, LABELS)
    
    if output_file != None:
        cv2.imwrite(output_file, frame)
        
    df = pd.DataFrame({
                'num_objects': len(idxs),
                'prob': confidences,
                'name': classIDs,
                'box': boxes,
                })
    
    return df
    
    
def run_net_on_frame(frame,
                     net,
                     net_size,
                     output_layers,
                     confidence_thresh,
                     thresh,
                     W, H,
                     COLORS,
                     LABELS):
    
    blob = cv2.dnn.blobFromImage(frame, 1/255, net_size, swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)
    
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_thresh:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, thresh)
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, idxs, boxes, confidences, classIDs
    

if __name__ == '__main__':
    video = '../../../testingvideos/mauitest_11_40s_1080.mp4'
    data_file = '../../../maui_sf_and_100m.data'
    config_file = '../../../yolov4-tiny-maui-sf-and-100m.cfg'
    weights = '../../../yolov4-tiny-maui-sf-and-100m_best.weights'
    names_file = '../../../maui.names'
    
    output_file = "__temp__.avi"
    
    df = process_video(video, data_file, config_file, weights, names_file,
                       output_file = output_file)
        