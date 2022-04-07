import argparse
import time
import json
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_txt, save_json, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.save_json, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'labels_json' if save_json else save_dir).mkdir(parents=True, exist_ok=True)  # make dir json

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # device = torch.device("cuda:1")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    for folder_name in sorted(os.listdir(source)):
            # Directories
            # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            # (save_dir / 'labels_{}'.format(folder_name) if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            # (save_dir / 'labels_json_{}'.format(folder_name) if save_json else save_dir).mkdir(parents=True, exist_ok=True)  # make dir json
        source_folder = os.path.join(source,folder_name)
        dataset = LoadImages(source_folder, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # print('p, s, im0, frame',p, s, im0, frame)
                # print('p', p)
                # print('s', s)
                # print('im0', im0)
                # print('frame', frame)
                p = Path(p)  # to Path
                # print('p_name', p.name)
                # save_path = str(save_dir / p.name)  # img.jpg
                # if not os.path.exists(str(save_dir / 'images' / folder_name)):
                #     os.mkdir(str(save_dir / 'images' / folder_name))
                (save_dir / 'images' / folder_name).mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / 'images' / folder_name / p.name)
                # print(save_path)  # img.jpg
                if not os.path.exists(str(save_dir / 'labels' / folder_name)):
                    os.mkdir(str(save_dir / 'labels' / folder_name))
                if not os.path.exists(str(save_dir / 'labels_json' / folder_name)):
                    os.mkdir(str(save_dir / 'labels_json' / folder_name))
                txt_path = str(save_dir / 'labels' / folder_name / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                json_path = str(save_dir / 'labels_json' / folder_name / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # if not os.path.exists(str(save_dir / 'labels_json' / p.stem)):
                #     os.mkdir(str(save_dir / 'labels_json' / p.stem))
                # json_path = os.path.join(str(save_dir / 'labels_json' / p.stem),('' if dataset.mode == 'image' else f'{frame}'))  # img.txt
                # print('json_path', json_path)
                # print('p.stem', p.stem)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    temp_shape = {}
                    temp_shape["info"] = {
                        "width": 3840,
                        "height": 2160,
                        "image_name": p.stem + '.jpg'
                    }
                    temp_shape["shapes"] = []
                    count = 1
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # print("xywh", xywh)
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            # print("line", line[0], line[1])
                            
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_json:
                            CLASSES = ('solar_pannel')
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            class_index = cls.int()
                            data_label = CLASSES[class_index]
                            
                            x1 = (xywh[0] - xywh[2] / 2)*3840
                            y1 = (xywh[1] - xywh[3] / 2)*2160
                            x2 = (xywh[0] + xywh[2] / 2)*3840
                            y2 = (xywh[1] + xywh[3] / 2)*2160
                            if x1 < 0:
                                x1 = 0.0
                            else:
                                x1 = x1
                            if y1 < 0:
                                y1 = 0.0
                            else:
                                y1 = y1
                            if x2 > 3840:
                                x2 = 3840.0
                            else:
                                x2 = x2
                            if y2 > 2160:
                                y2 = 2160.0
                            else:
                                y2 = y2
                            temp_shape["shapes"].append({
                                "object_id": count,
                                "label": data_label,
                                "points": (
                                (x1, y1),
                                (x2, y2)
                                ),
                                "shape_type": "bbox"
                            })
                            count += 1
                            with open(json_path + '.json', 'w', encoding='utf-8') as fp_temp:
                                json.dump(temp_shape, fp_temp, ensure_ascii=False, indent=4, sort_keys=True)
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # start = 1000
                # end = 1020
                # print(p.name.split('.')[0])
                if save_img:
                    # fps, w, h = 30, im0.shape[1], im0.shape[0]
                    # save_path += '.mp4'
                    # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if dataset.mode == 'image':
                        # if p.name.split('.')[0] >= start and p.name.split('.')[0] <= end:
                        cv2.imwrite(save_path, im0)
                        # fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # save_path += '.mp4'
                        # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        # vid_writer.write(im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-json', action='store_true', help='save results to *.json')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='/media/islab/4tb/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
