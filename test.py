import fiftyone as fo
import fiftyone.zoo as foz
# 10개의 클래스에 대해서만 데이터 획득

# person bicycle motorcycle car Bus traffic_light
# street_sign, Truck
voc_t = fo.Dataset('voc_train_mini')
voc_tst = fo.Dataset('voc_test_mini')
voc_v = fo.Dataset('voc_valid_mini')

# coco_test.persistent =True
# coco_valid.persistent = True
#coco_train.persistent = True

# # coco dataset에서 특정 개수의 이미지만 가지고 오는 코드
voc_train  =  foz.load_zoo_dataset("voc-2012", split='train')
voc_test = foz.load_zoo_dataset("voc-2012", split='test')
voc_valid = foz.load_zoo_dataset("voc-2012", split='validation')
#dataset export -> dataset 객체를 컴퓨터 내 로컬 파일로 저장
voc_valid.export(
    export_dir='/home/cclab/바탕화면/voc/val2017',
    dataset_type = fo.types.COCODetectionDataset,
    label_field = "ground_truth",
    labels_path = "../annotations/instances_val2017.json"
)
voc_test.export(
    export_dir='/home/cclab/바탕화면/voc/test2017',
    dataset_type = fo.types.COCODetectionDataset,
    label_field = "ground_truth",
    labels_path = "../annotations/instances_test2017.json"
)
voc_train.export(
    export_dir='/home/cclab/바탕화면/voc/test2017',
   # dataset_type = fo.types.COCODetectionDataset,
    label_field = "ground_truth",
    labels_path = "../annotations/instances_train2017.json"
)