from ultralytics import YOLO

model = YOLO("../spatial-perception/checkpoints/yolo11x.pt")

metrics = model.val(data="./coco.yaml", save_json=True)  ## save_json为True,可以把预测结果存成json文件， 便于评估或在线提交
