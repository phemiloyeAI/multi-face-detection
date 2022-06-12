import os
import cv2
import copy
import argparse
import tempfile
import numpy as np
from PIL import Image

import norfair
from norfair import Detection, Tracker, Paths, Video

import torch

import streamlit as st
from stqdm import stqdm

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords


class FaceDetection:

    def __init__(self, weights, conf_thresh, iou_thresh, device, output_dir, blur):
        self.model = self.load_model(weights, device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.output_dir = output_dir
        self.blur_face = blur
    
    @st.cache
    def load_model(self, weights, device):
        model = attempt_load(weights, device)
        return model

    def frame_preprocessing(self, frame):
        frame_size = 800
        frame0 = copy.deepcopy(frame)
        h0, w0 = frame.shape[:2]
        r = frame_size / max(h0, w0) 
        if r != 1:  
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            frame0 = cv2.resize(frame0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(frame_size, s=self.model.stride.max()) 

        new_frame = letterbox(frame0, new_shape=imgsz)[0]
        new_frame = new_frame[:, :, ::-1].transpose(2, 0, 1).copy() 
        
        new_frame = torch.from_numpy(new_frame).to(self.device)
        new_frame = new_frame.float() 
        new_frame /= 255.0 
        if new_frame.ndimension() == 3:
            new_frame = new_frame.unsqueeze(0)
        return new_frame, frame

    def get_detections(self, frame):
        frame, frame0 = self.frame_preprocessing(frame)
        dets = self.model(frame)[0]
        dets = non_max_suppression_face(dets, self.conf_thresh, self.iou_thresh)
        frame_preds, bboxes = self.post_process_detections(dets, frame, frame0)
        no_faces = dets[0].shape[0]
        return bboxes, frame0, frame_preds, no_faces

    def post_process_detections(self, dets, frame, frame0):
        bboxes = []
        for _, det in enumerate(dets): 
            if len(det):
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], frame0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 

                det[:, 5:15] = self.scale_coords_landmarks(frame.shape[2:], det[:, 5:15], frame0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    frame0 = self.draw_bounding_boxes(frame0, xyxy)
                    bboxes.append(xyxy)
        return frame0, bboxes

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None: 
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0] 
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  
        coords[:, :10] /= gain

        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    def draw_bounding_boxes(self, img, xyxy):
        h,w, _ = img.shape
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
    
        if self.blur_face == "Yes":
            roi = img[y1:y2, x1:x2]
            blur_roi = cv2.GaussianBlur(roi, (21, 21), 30)
            img[y1:y2, x1:x2] = blur_roi
    
        cv2.rectangle(img, (x1,y1), (x2, y2), (255,0,0), thickness=2, lineType=cv2.LINE_AA)

        return img
    
    def save_frame(self, frame, text):
        _, _, frame, no_faces = self.get_detections(frame)
        self.overlay_text_count(frame, text, no_faces)
        cv2.imwrite(self.output_dir, frame)
    
    def overlay_text_count(self, frame, text, count):
        text = text + str(count)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size[0]; x, y = (15, 10)
        cv2.rectangle(frame, (x, y), (x+text_w+30, y+text_h+30), (255, 255, 255), -1)
        cv2.putText(frame, text, (x+15, y+text_h+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            
# Norfair utils 
def get_norfair_detections(detections):
    norfair_detections = []

    for bbox in detections:
        bbox = np.array(bbox).reshape(2, 2)
        norfair_detections.append(Detection(points=bbox))

    return norfair_detections

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def center(points):
    return [np.mean(np.array(points), axis=0)]

# Process input data
def process_uploaded_file(weights, input_file, conf_thresh, iou_thresh, device, output, blur):
    detector = FaceDetection(weights, conf_thresh, iou_thresh, device, output, blur)
    ext = input_file.name.split(".")[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"]:
        
        max_distance_between_points = 30
        tracker = Tracker(
            distance_function=euclidean_distance,
            distance_threshold=max_distance_between_points
        )

        path_drawer = Paths(center, attenuation=0.02, thickness=2, radius=2, color=(139,0,0))

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(input_file.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'VP09')
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

        inc = 1
        disp_frame = False
     
        for _ in stqdm(range(length)):
            success, frame = cap.read()
            if not success:
                break
            else:
                dets, frame, _, _ = detector.get_detections(frame)
                norfair_detections = get_norfair_detections(dets)
                tracked_objects = tracker.update(detections=norfair_detections)
                    
                norfair.draw_tracked_objects(frame, tracked_objects, id_size=0.3, id_thickness=1, radius=0)
                path_drawer.draw(frame, tracked_objects)
                
                if inc == fps:
                    count = len([tracks.initializing_id for tracks in tracked_objects])
                    text = "People Counter: "
                    detector.overlay_text_count(frame, text, count)
                    writer.write(frame)                 
                    disp_frame = True
                    inc = 1

                if disp_frame:
                    text = "People Counter: "
                    detector.overlay_text_count(frame, text, count)
                    writer.write(frame)
                
                inc += 1

        cap.release()
        writer.release()

    if ext in ["jpg", "jpeg", "png"]:
        image = np.array(Image.open(input_file))
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        text = "Faces Counter: "
        detector.save_frame(cv2_img, text)

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--input', type=str, default='data/images/test.jpg', help='source')
    parser.add_argument('--output', type=str, help='path to save processed video or image')
    parser.add_argument('--conf_thresh', default=0.3, type=float, help='final_prediction_threshold')
    parser.add_argument('--iou_thresh', default=0.5, type=float, help='bounding_box_threshold')
    parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
 
    args = command_line_args()

    process_uploaded_file(args.weights, args.input, args.conf_thresh, args.iou_thresh, args.device, args.output)