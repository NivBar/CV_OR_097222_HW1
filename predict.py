import torch
import cv2

zero = torch.tensor(0)


def get_max_labels(tups, def_right=torch.tensor(2), def_left=torch.tensor(7)):
    """
    Arguments:
        - tups: list of tuples gotten from prediction (x,y,w,h,conf,label)
        - def_right: default value for right class (will most likely change during smoothing)
        - def_left:  default value for left class (will most likely change during smoothing)

    Find the most probable right and left tags while handling corner cases.

    Return: [most probable right output, most probable left output]
    """
    # even labels are right and odd labels are left
    try:
        max_right = max([tup for tup in tups if int(tup[-1]) % 2 == 0], key=lambda x: x[-2])
    except:
        max_right = []
    try:
        max_left = max([tup for tup in tups if int(tup[-1]) % 2 == 1], key=lambda x: x[-2])
    except:
        max_left = []

    if max_left and max_right:  # in case beth exist
        return max_right, max_left
    if len(tups) == 1:  # in case only one side is located
        if max_right:
            return [max_right, (zero, zero, zero, zero, zero, def_left)]
        elif max_left:
            return [(zero, zero, zero, zero, zero, def_right), max_left]
    elif len(tups) == 0:  # in case none is located
        return [(zero, zero, zero, zero, zero, def_right), (zero, zero, zero, zero, zero, def_left)]
    elif max_right:  # in case there are multiple right only
        return [max_right, (zero, zero, zero, zero, zero, def_left)]
    elif max_left:  # in case there are multiple left only
        return [(zero, zero, zero, zero, zero, def_right), max_left]


model = torch.hub.load('ultralytics/yolov5', model='custom',
                       path=fr".\best.pt",
                       source='github')


def predict(file_path=None, array=None, from_path=False):
    """
    Arguments:
        - file name: name of file without suffix
        - array: image representation by an array
        - from_path: using path or array to run the prediction, if true use path otherwise use array
    """
    if from_path:
        frame = cv2.imread(file_path)
    else:
        frame = array
    frame = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
    pred = model(frame)
    pred.render()
    xywh = get_max_labels([list(x) for x in pred.xywh[0]])
    xyxy = get_max_labels([list(x) for x in pred.xyxy[0]])
    return xywh, xyxy

