import json
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def hash(image_dir: str, output_dir: str):
    hashmap = dict()
    cnt = 0
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            hashmap[filename.split('.')[0]] = cnt
            cnt += 1
    with open(os.path.join(output_dir, 'hash.json'), 'w') as f:
        json.dump(hashmap, f)
    print('Hashing done!')

def yolo2coco(
        yolo_labels_dir: str,
        images_dir: str,
        output_dir: str,
        hash_file: str,
        score_or_not: bool = False,
        classes: list = [],
        ignore_classes: list = [],
        name: str = 'coco_labels.json'
):
    classes = [a for a in classes if a not in ignore_classes]

    # Initialize the COCO dataset
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }

    annotation_id = 0
    hashmap = dict()
    with open(hash_file, 'r') as f:
        hashmap = json.load(f)

    # For each image in the dataset
    for filename in os.listdir(yolo_labels_dir):
        if filename.endswith('.txt'):
            # Load YOLO labels
            with open(os.path.join(yolo_labels_dir, filename), 'r') as f:
                yolo_labels = f.readlines()
            with open(os.path.join(images_dir, filename.replace('.txt', '.jpg')), 'rb') as f:
                image = plt.imread(f)
                im_width, im_height = image.shape[1], image.shape[0]

            # Convert YOLO labels to COCO format
            for yolo_label in yolo_labels:
                if score_or_not:
                    class_id, x_center, y_center, width, height, score = map(float, yolo_label.strip().split())
                else:
                    class_id, x_center, y_center, width, height = map(float, yolo_label.strip().split())
                if class_id in ignore_classes:
                    continue
                if score_or_not:
                    coco_label = {
                        "id": annotation_id,
                        "image_id": hashmap[filename.split('.')[0]],
                        "category_id": int(class_id),
                        "bbox": [round((x_center - width / 2) * im_width), round((y_center - height / 2) * im_height), round(width * im_width), round(height * im_height)],
                        "area": round(width * height * im_width * im_height),
                        "iscrowd": 0,
                        "score": score
                    }
                else:
                    coco_label = {
                        "id": annotation_id,
                        "image_id": hashmap[filename.split('.')[0]],
                        "category_id": int(class_id),
                        "bbox": [round((x_center - width / 2) * im_width), round((y_center - height / 2) * im_height), round(width * im_width), round(height * im_height)],
                        "area": round(width * height * im_width * im_height),
                        "iscrowd": 0
                    }
                coco_data["annotations"].append(coco_label)
                annotation_id += 1

            # Add image information
            coco_data["images"].append({
                "id": hashmap[filename.split('.')[0]],
                "file_name": filename.replace('.txt', '.jpg'),  # replace with your image file extension
                "width": im_width, 
                "height": im_height
            })

    # Save COCO data to a JSON file
    with open(os.path.join(output_dir, name), 'w') as f:
        json.dump(coco_data, f)
    print(f'{yolo_labels_dir} Conversion done!')

def plot_PR_curve(coco_eval, output_dir):
    # Get precision and recall
    precision = coco_eval.eval['precision']
    # precision and recall are 4D arrays, we need to average over the last two dimensions
    precision = precision.mean(axis=(2, 3))
    recall = coco_eval.params.recThrs

    # Plot the P-R curve
    plt.figure()
    plt.plot(recall, precision, 'b-')
    plt.title('Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Save the P-R curve
    plt.savefig(os.path.join(output_dir, 'PR_curve.png'))

def plot_PR_curve_per_class(coco_eval, output_dir, classes):
    # Get precision and recall
    precision = coco_eval.eval['precision']
    # precision is a 4D array, we need to average over the last dimension
    precision_ = precision.mean(axis=(3,4))[0]
    recall = coco_eval.params.recThrs
    pre = np.transpose(precision_)

    # Plot the P-R curve for each class
    plt.figure()
    for i, class_name in enumerate(classes):
        plt.plot(recall, pre[i], label=class_name)
    plt.title('Precision-Recall curve per class')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Save the P-R curve
    plt.savefig(os.path.join(output_dir, 'PR_curve_per_class.png'))

def val(
        ground_truth_dir: str,
        predictions_dir: str,
        output_dir: str
):
    # Load the ground truth and detections in COCO format
    coco_gt = COCO(ground_truth_dir)
    coco_dt = COCO(predictions_dir)

    # Create a COCO Eval object
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Set the IoU threshold
    coco_eval.params.iouThrs = [0.5]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    category = coco_gt.dataset['categories']
    classes = [""] * len(category)
    for cat in category:
        classes[cat['id']] = cat['name']
    plot_PR_curve_per_class(coco_eval, output_dir, classes)

    print('finish')

def dataprocessing():
        hash(image_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/images', output_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/')
        yolo2coco(
            yolo_labels_dir='runs/detect/s_coco_e100_4Class_Vehicle3/labels',
            images_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/images',
            output_dir='runs/detect/s_coco_e100_4Class_Vehicle3',
            hash_file='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/hash.json',
            score_or_not=True,
            classes=['car', 'van', 'truck', 'bus']
        )
        yolo2coco(
            yolo_labels_dir='runs/detect/enhanced3/labels',
            images_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/images',
            output_dir='runs/detect/enhanced3',
            hash_file='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/hash.json',
            score_or_not=True,
            classes=['car', 'van', 'truck', 'bus', 'patch']
        )
        yolo2coco(
            yolo_labels_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/labels',
            images_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/images',
            output_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev',
            hash_file='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/hash.json',
            classes=['car', 'van', 'truck', 'bus', 'patch'],
            ignore_classes=[],
            name='patch.json'
        )
        yolo2coco(
            yolo_labels_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/labels',
            images_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/images',
            output_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev',
            hash_file='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/hash.json',
            classes=['car', 'van', 'truck', 'bus', 'patch'],
            ignore_classes=['patch'],
            name='label.json'
        )

def dataviewing():
    val(
        ground_truth_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/label.json',
        predictions_dir='runs/detect/s_coco_e100_4Class_Vehicle3/coco_labels.json',
        output_dir='runs/detect/s_coco_e100_4Class_Vehicle3'
    )
    val(
        ground_truth_dir='../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/patch.json',
        predictions_dir='runs/detect/enhanced3/coco_labels.json',
        output_dir='runs/detect/enhanced3'
    )

if __name__ == '__main__':
    dataprocessing()
    dataviewing()