import argparse
import os
import json
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='/media/islab/4tb/jehwan_연구/산업융합학회/crop_image_ori', help='json_folder를 입력하세요')
args = parser.parse_args()

CLASSES = ('tree, 나무', 'person, 사람', 'animal, 동물', 'house, 주택', 'apartment, building, 아파트, 빌딩', 
            'school, 학교', 'office, 관리소', 'traffic sign, 교통표지판', 'traffic light, 신호등', 'streetlamp, telephone pole, 가로등, 전신주', 
            'banner, 현수막','milestone, 정상표식', 'bridge, 다리', 'tower, 상징탑', 'car_vehicle, 승용차', 'bus_vehicle, 버스', 
            'truck_vehicle, 트럭', 'motorcycle, bike_vehicle, 오토바이, 자전거')
natural = ['tree, 나무', 'person, 사람', 'animal, 동물']
building = ['house, 주택', 'apartment, building, 아파트, 빌딩', 'school, 학교', 'office, 관리소']
structure = ['traffic sign, 교통표지판', 'traffic light, 신호등', 'streetlamp, telephone pole, 가로등, 전신주', 'banner, 현수막', 'milestone, 정상표식', 'bridge, 다리', 'tower, 상징탑']
vehicle = ['car_vehicle, 승용차', 'bus_vehicle, 버스', 'truck_vehicle, 트럭', 'motorcycle, bike_vehicle, 오토바이, 자전거']


folder = args.folder
class_names = sorted(os.listdir(folder))
for i,class_name in enumerate(class_names):
    class_path = os.path.join(folder, class_name)
    for j,image in enumerate(sorted(os.listdir(class_path))):
        try:
            only_image_name = image.split('.')[0]
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            height, width, _ = img.shape[0],img.shape[1],img.shape[2]
            print(height, width)
            if class_name in natural:
                LargeCategory = 'natural, 자연물'
                MediumCategory = class_name
                SmallCategory = ''
            else:
                LargeCategory = 'artifact, 인공물'
                if class_name in building:
                    MediumCategory = 'building, 건물'
                elif class_name in structure:
                    MediumCategory = 'structure, 구조물'
                elif class_name in vehicle:
                    MediumCategory = 'vehicle, 차량'
                SmallCategory = class_name
            temp_shape = {}
            temp_shape["info"] = {
                "width": width,
                "height": height,
                "image_name": only_image_name + '.jpg'
            }
            temp_shape["shapes"] = []
            temp_shape["shapes"].append({
            "occluded": 0,
            "LargeCategory": LargeCategory,
            "MediumCategory": MediumCategory,
            "SmallCategory": SmallCategory,
            "label": class_name,
            "points": (
            (0, 0),
            (width-1, height-1)
            ),
            "shape_type": "bbox"
            })
            with open(class_path + '/' + only_image_name + '.json', 'w', encoding='utf-8') as fp_temp:
                json.dump(temp_shape, fp_temp, ensure_ascii=False, indent=4, sort_keys=True)   
        except:
            pass