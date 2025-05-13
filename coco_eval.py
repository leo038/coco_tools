import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics.data import converter

CLASS_MAP = converter.coco80_to_coco91_class()
print(f"coco91 class_map: {CLASS_MAP}， len:{len(CLASS_MAP)}")


def yolo_result_to_coco(init_json_path, save_json_name="./convert_result.json"):
    f = open(init_json_path, 'r', encoding='utf-8')
    dts = json.load(f)
    output_dts = []
    for dt in dts:
        # dt['image_id'] = filename2id[dt['image_id'] + '.jpg']
        category_id_coco = CLASS_MAP[int(dt['category_id'] - 1)]  ##yolo评估结果中保存的json文件， 其id是从1到80, 而coco中是1道90
        dt.update({"category_id": category_id_coco})
        output_dts.append(dt)
    with open(save_json_name, 'w') as f:
        json.dump(output_dts, f)
    print(f"json file convert finished!!! saved to {save_json_name}")
    return save_json_name


def coco_evaluate(gt_path, dt_path, id_type):
    cocoGt = COCO(gt_path)
    if id_type == "coco80":
        convert_json = yolo_result_to_coco(dt_path)
        cocoDt = cocoGt.loadRes(convert_json)
    else:
        cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="Assign the groud true path.",
                        default='/data/joyiot/leo/datasets/coco/annotations/instances_val2017.json')
    parser.add_argument("--dt", type=str, help="Assign the detection result path.",
                        default='/data/joyiot/leo/codes/spatial-perception/runs/detect/val17/predictions.json')
    parser.add_argument("--id_type", default="coco80", help="id_type is coco80 or coco91")

    args = parser.parse_args()
    gt_path = args.gt
    dt_path = args.dt
    id_type = args.id_type

    coco_evaluate(gt_path, dt_path, id_type)
