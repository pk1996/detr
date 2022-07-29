'''
Integrating KITTI dataset into the detectron2 framework

References
1. https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
2. https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
3. https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html 

'''
import os.path as osp
import cv2, random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm
import json

CLASS = {'Car': 0, 'Pedastrain': 1, 'Cyclist': 2}

def parse_label(label):
    '''
    Helper function to parse through the label
    info from text file
    Refer - https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt
    '''
    lab_data = label.split(' ')
    lab_obj = {}
    lab_obj['class'] = CLASS.get(lab_data[0], -1)
    lab_obj['bbox'] = [float(lab_data[4]), float(lab_data[5]), float(lab_data[6]), float(lab_data[7])]
    lab_obj['dim'] = [float(lab_data[8]), float(lab_data[9]), float(lab_data[10])]
    lab_obj['loc'] = [float(lab_data[11]), float(lab_data[12]), float(lab_data[13])]
    lab_obj['roty'] = [float(lab_data[14])]
    return lab_obj

def get_kitti_dicts(split = 'train'):

    # bsae path
    base_path = "/srip-vol/datasets/KITTI3D/training"

    # label path
    label_path = "/srip-vol/datasets/KITTI3D/training/label_2"

    # image set
    image_set = 'train' if split == 'train' else 'val'   
    image_set = osp.join("/srip-vol/datasets/KITTI3D", 'ImageSets', image_set + '.txt')

    # Read indices
    indices = open(image_set).readlines()
    indices = sorted([index.rstrip() for index in indices])

    if osp.isfile('dd3d_kitti_data.json'):
        # Trying to cache it to avoid computing this dict repeatedly.
        print('Data dict already stored in file, loadin')
        dataset_dicts = json.load(open('dd3d_kitti_data.json', 'r'))
    else:
        dataset_dicts = []
        for idx in tqdm(indices):
            # Iterate through the dataset
            record = {}

            filename = osp.join(base_path, 'image_2', idx+'.png')
            # print(filename)
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            # iterate through label file
            labels = open(osp.join(label_path, idx+'.txt')).readlines()
            objs = []
            for label in labels:
                # Create label object
                label_data = parse_label(label)

                if(label_data['class'] == -1):
                    continue
                
                obj = {
                    "bbox": label_data['bbox'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": label_data['class'],
                    "depth": label_data['loc'][-1],
                }

                # TODO - Add depth and BEV annotation.

                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "val"]:
    DatasetCatalog.register("kitti_" + d, lambda d=d: get_kitti_dicts(split = d))
    MetadataCatalog.get("kitti_" + d).set(thing_classes=["Car", "Pedastrain", "Cyclist"])
# balloon_metadata = MetadataCatalog.get("balloon_train")

if __name__ == '__main__':
    # print('Here')
    dataset_dicts = get_kitti_dicts("val")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], scale=0.5) # metadata=balloon_metadata,
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])