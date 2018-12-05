import os
import numpy as np
import pandas as pd
import skimage.draw
from pathlib import Path
import random

# Import Mask RCNN
from models.maskrcnn.config import Config
from models.maskrcnn import model as modellib, utils, visualize
from datasets.spacenet import parse_and_save_target, parse_and_save_feature, ORIGINAL_SIZE, load_feature_by_imageid
# Path to trained weights file
COCO_WEIGHTS_PATH = "/e/data/model_images/mask_rcnn_coco.h5"

OUTPUT_DIR = Path('/e/data/output/AOI_4_Shanghai_Train_MRCNN_20181203')
OUTPUT_IMAGE_MASK_H5 = str(OUTPUT_DIR / 'image_mask.h5')
OUTPUT_IMAGE_ID_CSV = str(OUTPUT_DIR / 'image_id.csv')
OUTPUT_IMAGE_POLYGON_CSV = str(OUTPUT_DIR / 'image_polygon.csv')
OUTPUT_IMAGE_RGB_H5 = str(OUTPUT_DIR / 'image_rgb.h5')
OUTPUT_IMAGE_RGB_NORM_CSV = str(OUTPUT_DIR / 'image_rgb_norm.csv')
OUTPUT_IMAGE_RGB_SOLUTION_CSV = str(OUTPUT_DIR / 'image_rgb_solutions.csv')
DEFAULT_LOGS_DIR = str(OUTPUT_DIR / "logs")


PREPARE = False


############################################################
#  Configurations
############################################################


class SpacenetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "spacenet"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


############################################################
#  Dataset
############################################################

class SpacenetDataset(utils.Dataset):
    image_id_df = pd.read_csv(OUTPUT_IMAGE_ID_CSV)

    def load_spacenet(self, dataset_dir, subset, train_ratio=0.8):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("building", 1, "building")

        DATA_DIR = Path(dataset_dir)
        SUMMARY_DATA_DIR = DATA_DIR / 'summaryData'

        # Train or validation dataset?
        assert subset in ["train", "val"]

        if PREPARE:
            parse_and_save_target(SUMMARY_DATA_DIR, OUTPUT_IMAGE_ID_CSV,
                                  OUTPUT_IMAGE_MASK_H5, OUTPUT_IMAGE_POLYGON_CSV)

            parse_and_save_feature(DATA_DIR, OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_RGB_H5,
                                   OUTPUT_IMAGE_RGB_NORM_CSV, 'RGB-PanSharpen', 3)

        image_df = pd.read_csv(OUTPUT_IMAGE_ID_CSV)
        image_ids = image_df['id'].tolist()
        image_polygon_df = pd.read_csv(OUTPUT_IMAGE_POLYGON_CSV)

        validation_start = int(len(image_ids) * train_ratio)

        if subset == 'train':
            image_ids = image_ids[:validation_start]
        else:
            image_ids = image_ids[validation_start:]

        # Add images
        for image_id in image_ids:
            self.add_image(
                "building",
                image_id=image_polygon_df.loc[image_id, 'ImageId'],
                path=image_polygon_df.loc[image_id, 'ImageId'],
                width=ORIGINAL_SIZE, height=ORIGINAL_SIZE,
                polygons=image_polygon_df.loc[image_id, 'ImagePolygon'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        polygons = eval(info['polygons'])
        if len(polygons) == 0:
            return super(SpacenetDataset, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(polygons)],
                        dtype=np.uint8)
        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            exterior_points = p['exterior_coords']
            x = [round(float(pp[0])) for pp in exterior_points]
            y = [round(float(pp[1])) for pp in exterior_points]
            yy, xx = skimage.draw.polygon(y, x, (ORIGINAL_SIZE, ORIGINAL_SIZE))
            mask[yy, xx, i] = 1

            interior_points_list = p['interior_coords']
            for interior_points in interior_points_list:
                x = [round(float(pp[0])) for pp in interior_points]
                y = [round(float(pp[1])) for pp in interior_points]
                yy, xx = skimage.draw.polygon(y, x, (ORIGINAL_SIZE, ORIGINAL_SIZE))
                mask[yy, xx, i] = 0

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        if type(image_id) != str:
            # transfer id to ImageId
            image_id = self.image_id_df.loc[image_id, 'ImageId']
        image = load_feature_by_imageid(image_id, OUTPUT_IMAGE_RGB_H5)
        # change back to [0-255]
        image = image * 255
        return image


def train(model, args, config):
    """Train the model."""
    # Training dataset.
    dataset_train = SpacenetDataset()
    dataset_train.load_spacenet(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SpacenetDataset()
    dataset_val.load_spacenet(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads')


def command_line(input_list):
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN for Spacenet.')
    parser.add_argument("--command", required=False,
                        default="test",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        default="/e/data/AOI_4_Shanghai_Train",
                        metavar="/path/to/spacenet/dataset/",
                        help='Directory of the Spacenet dataset')
    parser.add_argument('--weights', required=False,
                        default="last",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    args = parser.parse_args(eval(input_list))

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SpacenetConfig()
    else:
        class InferenceConfig(SpacenetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args, config)
    else:
        dataset = SpacenetDataset()
        dataset.load_spacenet(args.dataset, "val")

        # Must call before using the dataset
        dataset.prepare()

        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

        if args.command == 'evaluate':
            image_id = random.choice(dataset.image_ids)
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            info = dataset.image_info[image_id]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))

            # Run object detection
            results = model.detect([image], verbose=1)

            # Display results
            r = results[0]
            print(r)
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        dataset.class_names, r['scores'], ax=None,
                                        title="Predictions")
            print("gt_class_id {}".format(gt_class_id))
            print("gt_bbox {}".format(gt_bbox))
            print("gt_mask {}".format(gt_mask))
        elif args.command == "evaluate_all":
            image_result_path = OUTPUT_DIR / 'image_result'
            if not image_result_path.exists():
                image_result_path.mkdir()

            for image_id in dataset.image_ids:
                image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
                info = dataset.image_info[image_id]
                print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                       dataset.image_reference(image_id)))

                # Run object detection
                results = model.detect([image], verbose=1)

                # Display results
                r = results[0]
                print(r)
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                            dataset.class_names, r['scores'], ax=None,
                                            title="Predictions",
                                            save_name="{}.jpg".format(str(image_result_path/info["id"])))
