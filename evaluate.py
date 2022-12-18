import video
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

vid_names = ["P022_balloon1", "P023_tissue2", "P024_balloon1", "P025_tissue2", "P026_tissue1"]
labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
names = video.names

y_trues = []
y_preds = []
for vid in vid_names:
    print(f"handling {vid}...")
    y_true, y_pred = video.get_prediction_video_and_labels(vid)
    y_trues += y_true
    y_preds += y_pred

P, R, F1, _ = precision_recall_fscore_support(y_trues, y_preds, average=None,
                                              labels=labels)

CM = confusion_matrix(y_true=y_trues, y_pred=y_preds, labels=labels)
ACC = CM.diagonal() / CM.sum(axis=1)

data = []
for i in range(8):
    data.append({"class": names[str(i)], "precision": P[i], "recall": R[i], "F1 score": F1[i], "accuracy": ACC[i],
                 "total instances": sum([1 if x == str(i) else 0 for x in y_trues])})

df = pd.DataFrame(data=data)
df.to_csv("video_evaluation.csv", index=False)
