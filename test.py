import fiftyone as fo
import fiftyone.zoo as foz
# 10개의 클래스에 대해서만 데이터 획득

coco_simple_valid = fo.Dataset('coco_simple_minit')


# # coco dataset에서 특정 개수의 이미지만 가지고 오는 코드
coco_simple_valid = foz.load_zoo_dataset("coco-2017", split='train',max_samples=10, shuffle=True,)
#dataset export -> dataset 객체를 컴퓨터 내 로컬 파일로 저장
coco_simple_valid.export(
    export_dir='./coco_simple/train2017',
    dataset_type = fo.types.COCODetectionDataset,
    label_field = "ground_truth",
    labels_path = "../annotations/instances_train2017.json"
)
