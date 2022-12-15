import pickle

import numpy as np
import pandas as pd

import predict
import bbox_visualizer as bbv
import cv2
from tqdm import tqdm

names = {"0": "Right_Scissors", "1": "Left_Scissors", "2": "Right_Needle_driver", "3": "Left_Needle_driver",
         "4": "Right_Forceps", "5": "Left_Forceps", "6": "Right_Empty", "7": "Left_Empty"}


def get_gt_label(t_label, direction):
    """
    Arguments:
        - t_label: label used in input data (T{i} where i in [0,3])
        - direction: left or right

    Translates the input labels to the labels the model was trained on

    Return: translated label index
    """
    if direction == "right":
        tool_usage = {"T0": "6", "T1": "2", "T2": "4", "T3": "0"}
        return tool_usage[t_label]
    elif direction == "left":
        tool_usage = {"T0": "7", "T1": "3", "T2": "5", "T3": "1"}
        return tool_usage[t_label]


def find_gt_class(idx, gt_dict, dir):
    """
    Arguments:
        - idx: index of the frame in the video
        - gt_dict: ground truth dictionary holding tuples of segments of continuous classes
        - dir:  direction (left\right)

    extract gt label from the gt_dict

    Return: gt label from input data
    """
    dir_list = gt_dict[dir]
    for tup in dir_list:
        if tup[0] <= idx <= tup[1]:
            return tup[2]


def data_time_series_smoothing(predictions, i, k, thresh, direction):
    """
    Arguments:
        - predictions: a list of two reordered outputs one for right hand and one for left
        - i: index of frame
        - k: size of sliding window, k each size and the undex itself, window size 2k+1
        -thresh: threshold of conf level from it outputs will be taken into consideration and disregarded otherwise
        - direction: left or right

    Smooth the data outputs, both labels and predicted bboxes

    Return: new smoothed prediction
    """
    if direction == "right":
        predictions = [pred[0] for pred in predictions]
    elif direction == "left":
        predictions = [pred[1] for pred in predictions]
    else:
        raise

    n = len(predictions)
    min_idx, max_idx = max(0, i - k), min(n - 1, i + k)
    relevants = predictions[min_idx:max_idx + 1]

    for q in range(len(relevants)):
        relevants[q] = [float(a) for a in relevants[q]]
        relevants[q][-1] = str(int(relevants[q][-1]))

    curr_x, curr_y, curr_w, curr_h, curr_conf, curr_class = (float(a) for a in predictions[i])
    curr_class = str(int(curr_class))

    # label smoothing
    # using argmax, taking predicition confidece into consideration from past, present and future
    rel_labels = [rel[-1] for rel in relevants if float(rel[-2]) > thresh]
    rel_labels.append(curr_class)
    try:
        est_label = (max(set(rel_labels), key=rel_labels.count))
    except:
        est_label = curr_class
    # if est_val != predictions[i][-1]:
    # print(i, rel_labels, predictions[i][-1], str(est_val))

    # bbox smoothing
    # using the weighted moving average method to smooth the predicted bboxes
    est_bbox = []
    try:
        rel_confs = sum([float(rel[-2]) for rel in relevants if float(rel[-2]) > thresh])
        rel_confs += float(curr_conf)  # double weight on current conf i.e. the current bbox
        for j in range(4):
            # {1: x, y: 2, 3: w, 4: w}
            rel_weighted_vals = sum([float(rel[j]) * float(rel[-2]) for rel in relevants if float(rel[-2]) > thresh])
            rel_weighted_vals += float(predictions[i][j]) * float(curr_conf)
            est_val = rel_weighted_vals / rel_confs
            est_bbox.append(str(round(est_val, 3)))
    except:
        est_bbox = [curr_x, curr_y, curr_w, curr_h]

    new_pred = est_bbox[0], est_bbox[1], est_bbox[2], est_bbox[3], str(round(curr_conf, 3)), str(est_label)
    return new_pred


def get_prediction_video_and_labels(vid_name):
    """
    Arguments:
        - vid_name: name of the video without suffix we want to create a prediction video for.

    Running inference on the video frame by frame, smoothing the labels and bounding boxes and finally creating the
    prediction videos

    Return: true labels and smoothed predicted labels (and creates the mp4 predicted video, named {vid_name}_new.mp4)
    """
    # vid_name = "P025_tissue2"
    vid_dir_path = rf'./videos/{vid_name}'

    # get ground truth data
    print("creating ground truth dict...")
    gt_dict = dict()
    for dir in ["right", "left"]:
        gt_dict[dir] = []
        with open(f"./data/HW1_dataset/tool_usage/tools_{dir}/{vid_name}.txt", "r") as file:
            rows = file.read().split("\n")
            for row in rows:
                if not row:
                    continue
                min_, max_, t_label = row.split()
                gt_dict[dir].append((int(min_), int(max_), get_gt_label(t_label, dir)))

    cap = cv2.VideoCapture(f"{vid_dir_path}.wmv")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x = 1

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    print("creating initial predictions...")
    predictions = []
    frames = []
    i = -1
    while cap.isOpened():
        i += 1
        # print(i)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            xywh, xyxy = predict.predict(array=frame)
            pred_right, pred_left = xyxy

            pred_right = [round(float(x.item()), 3) for x in pred_right]
            pred_right[-1] = int(pred_right[-1])
            pred_right = [str(x) for x in pred_right]
            pred_left = [round(float(x.item()), 3) for x in pred_left]
            pred_left[-1] = int(pred_left[-1])
            pred_left = [str(x) for x in pred_left]

            predictions.append([pred_right, pred_left])
            # add bounding boxes
            frames.append(frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    y_true = []
    y_pred = []
    for i in range(len(predictions)):
        right_gt, left_gt = find_gt_class(i, gt_dict, "right"), find_gt_class(i, gt_dict, "left")
        y_true += [right_gt, left_gt]

    print("creating video while using smoothing function...")
    new_frames = []
    for i in tqdm(range(len(predictions))):
        xyxy = [data_time_series_smoothing(predictions, i, 2, 0.75, "right"),
                data_time_series_smoothing(predictions, i, 2, 0.75, "left")]
        right_bbox = [[int(float(x)) for x in xyxy[0][:4]]]
        right_conf = xyxy[0][4]
        right_class = xyxy[0][5]
        left_bbox = [[int(float(x)) for x in xyxy[1][:4]]]
        left_conf = xyxy[1][4]
        left_class = xyxy[1][5]
        frame = frames[i]
        y_pred += [right_class, left_class]

        frame = bbv.draw_multiple_rectangles(frame, right_bbox, bbox_color=(255, 0, 0))
        frame = bbv.add_multiple_T_labels(frame, [f'{names[right_class]} ({right_conf})'], right_bbox,
                                          text_bg_color=(255, 0, 0))
        frame = bbv.draw_multiple_rectangles(frame, left_bbox, bbox_color=(0, 250, 0))
        frame = bbv.add_multiple_T_labels(frame, [f'{names[left_class]} ({left_conf})'], left_bbox,
                                          text_bg_color=(0, 250, 0))

        text_right = f"right: gt - {names[find_gt_class(i, gt_dict, 'right')]}, pred - {names[right_class]}"
        text_left = f"left: gt - {names[find_gt_class(i, gt_dict, 'left')]}, pred - {names[left_class]}"
        frame = cv2.putText(frame, text_right, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, text_left, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2, cv2.LINE_AA)
        new_frames.append(frame)

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    writer = cv2.VideoWriter(f"{vid_name}_new.mp4", fps=30, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                             frameSize=(640, 480))

    for i in range(len(new_frames)):
        writer.write(new_frames[i])
    writer.release()

    print("Done!")
    return y_true, y_pred
