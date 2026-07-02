
from ultralytics import YOLO
import torch


import numpy as np


class SimpleRecorder:
    def __init__(self):
        self.batch_stats = []
        self.epoch_history = []

    def on_train_batch_end(self, trainer):
        # 从模型内部的 criterion 获取 stats
        criterion = getattr(trainer.model, 'criterion', None)
        if criterion is not None and hasattr(criterion, 'current_stats'):
            self.batch_stats.append(criterion.current_stats)

    def on_train_epoch_end(self, trainer):
        if not self.batch_stats:
            print(f"Epoch {trainer.epoch}: No stats collected, skip saving.")
            return
        avg = {}
        for k in self.batch_stats[0].keys():
            avg[k] = np.mean([s[k] for s in self.batch_stats])
        self.epoch_history.append(avg)
        self.batch_stats = []
        np.save("assign_stats_history.npy", self.epoch_history)
        print(f"Epoch {trainer.epoch}: "
              f"fg_ratio={avg['fg_ratio']:.4f}, "
              f"pos/gt={avg['pos_per_gt_mean']:.1f}, "
              f"avg_target_score={avg['avg_target_score']:.3f}")


if __name__ == '__main__':

    model = YOLO(r"C:\Users\caoke\Desktop\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml")

    # 创建记录器并注册回调
    recorder = SimpleRecorder()
    model.add_callback("on_train_batch_end", recorder.on_train_batch_end)
    model.add_callback("on_train_epoch_end", recorder.on_train_epoch_end)

    model.info(verbose=True)
    results = model.train(
        data=r"C:\Users\caoke\Desktop\ultralytics-main\datasets\data\data.yaml",
        epochs=50,
        imgsz=640,
        batch=-1,
        amp=False,
        cache="ram",
        workers=1,
        project="results",
        # iou_type="siou",
        # label_smoothing=0.0,
        name="yolov8Anchor_free+task_alignes50times",
    )