from pathlib import Path
import pandas as pd
import numpy as np
import skimage.io
import shapely.geometry

from keras.callbacks import ModelCheckpoint, EarlyStopping, History
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from models.unet import get_unet
from datasets.spacenet import \
    parse_and_save_target, parse_and_save_feature, split_train_validation_batch, mask_to_poly, \
    ORIGINAL_SIZE, INPUT_SIZE, MIN_POLYGON_AREA
from utils.log import logger

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)

KTF.set_session(session)


if __name__ == "__main__":
    # input files
    # DATA_DIR = Path('/e/data/SpaceNet_Buildings_Competition_Round2_Sample/AOI_4_Shanghai_Train/')
    DATA_DIR = Path('/e/data/AOI_4_Shanghai_Train/')
    MUL_DATA_DIR = DATA_DIR / 'MUL'
    MUL_PANSHARPEN_DATA_DIR = DATA_DIR / 'MUL-PanSharpen'
    PAN_DATA_DIR = DATA_DIR / 'PAN'
    RGB_PANSHARPEN_DATA_DIR = DATA_DIR / 'RGB-PanSharpen'
    SUMMARY_DATA_DIR = DATA_DIR / 'summaryData'
    # output files
    OUTPUT_DIR = Path('/e/data/output/AOI_4_Shanghai_Train_20181126')
    OUTPUT_IMAGE_MASK_H5 = str(OUTPUT_DIR / 'image_mask.h5')
    OUTPUT_IMAGE_ID_CSV = str(OUTPUT_DIR / 'image_id.csv')
    OUTPUT_IMAGE_RGB_H5 = str(OUTPUT_DIR / 'image_rgb.h5')
    OUTPUT_IMAGE_RGB_NORM_CSV = str(OUTPUT_DIR / 'image_rgb_norm.csv')
    OUTPUT_IMAGE_RGB_SOLUTION_CSV = str(OUTPUT_DIR / 'image_rgb_solutions.csv')
    # model related
    checkpoints_path = OUTPUT_DIR / 'unet_checkpoints'
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()
    if not checkpoints_path.exists():
        checkpoints_path.mkdir()
    image_result_path = OUTPUT_DIR / 'image_result'
    if not image_result_path.exists():
        image_result_path.mkdir()
    OUTPUT_UNET_CHECKPOINTS_PATH = str(checkpoints_path / 'weights.epoch({epoch:02d})-val_loss({val_loss:.2f}).h5')
    OUTPUT_UNET_LAST_MODEL_PATH = str(OUTPUT_DIR / 'unet_last_model.h5')
    OUTPUT_UNET_TRAIN_HISTORY_CSV = str(OUTPUT_DIR / 'unet_train_history.csv')

    PREPARE = True

    if PREPARE:
        parse_and_save_target(SUMMARY_DATA_DIR, OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_MASK_H5)

        parse_and_save_feature(DATA_DIR, OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_RGB_H5,
                               OUTPUT_IMAGE_RGB_NORM_CSV, 'RGB-PanSharpen', 3)

    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 4
    TRAIN_RATIO = 0.8
    MODE = "train"

    # full size load, need a lot of RAM
    # targets = load_target(OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_MASK_H5)
    # features = load_feature(OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_RGB_H5)
    # logger.info("size of target : {}".format(targets.shape))
    # logger.info("size of features : {}".format(features.shape))
    # features_train, feature_val, targets_train, targets_val = split_train_validation(features, targets)

    if MODE == "test":
        train_data_generator, val_data_generator, train_pairs, val_pairs = \
            split_train_validation_batch(OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_RGB_H5, OUTPUT_IMAGE_MASK_H5,
                                         train_batch_size=TRAIN_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
                                         train_ratio=TRAIN_RATIO)

        model_checkpoint = ModelCheckpoint(
            OUTPUT_UNET_CHECKPOINTS_PATH,
            monitor='val_jaccard_coef_int',
            save_best_only=False)
        model_earlystop = EarlyStopping(
            monitor='val_jaccard_coef_int',
            patience=10,
            verbose=0,
            mode='max')
        model_history = History()

        logger.info("Fit")
        model = get_unet(feature_count=3)
        model.fit_generator(
            train_data_generator,
            steps_per_epoch=round((len(train_pairs) + len(val_pairs)) / TRAIN_BATCH_SIZE),
            epochs=20,
            verbose=1,
            validation_data=val_data_generator,
            validation_steps=round(len(val_pairs) / VAL_BATCH_SIZE),
            callbacks=[model_checkpoint, model_earlystop, model_history])
        model.save_weights(OUTPUT_UNET_LAST_MODEL_PATH)

        # Save evaluation history
        pd.DataFrame(model_history.history).to_csv(OUTPUT_UNET_TRAIN_HISTORY_CSV, index=False)
        logger.info("train unet down")
    else:
        train_data_generator, val_data_generator, train_pairs, val_pairs = \
            split_train_validation_batch(OUTPUT_IMAGE_ID_CSV, OUTPUT_IMAGE_RGB_H5, OUTPUT_IMAGE_MASK_H5,
                                         train_batch_size=TRAIN_BATCH_SIZE, val_batch_size=1,
                                         train_ratio=TRAIN_RATIO, use_shuffle=False, keep_target_dim=True)

        model = get_unet(feature_count=3)
        model.load_weights(OUTPUT_UNET_LAST_MODEL_PATH)

        with open(OUTPUT_IMAGE_RGB_SOLUTION_CSV, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            for idx, image_id in enumerate(val_pairs):
                x, y_target = next(val_data_generator)
                y_pred = model.predict(x, batch_size=1, verbose=1)
                logger.info("predicting {}".format(image_id))
                im_pred = skimage.transform.resize(x[0], (ORIGINAL_SIZE, ORIGINAL_SIZE))
                im_target = np.copy(im_pred)
                # add result mask
                result_mask = (skimage.transform.resize(y_pred[0], (ORIGINAL_SIZE, ORIGINAL_SIZE)) > 0.5).astype(np.uint8)
                result_mask = result_mask.reshape((ORIGINAL_SIZE, ORIGINAL_SIZE))

                target_mask = y_target[0].reshape((ORIGINAL_SIZE, ORIGINAL_SIZE))

                # switch to 2D and save to csv
                # np.savetxt("{}_x.csv".format(image_id), target_mask, delimiter=",")

                mask_locations_y, mask_locations_x = np.where(result_mask == 1)
                im_pred[mask_locations_y, mask_locations_x, :]\
                    = im_pred[mask_locations_y, mask_locations_x, :] * 0.5 + np.array([0, 0.5, 0]) * 0.5
                skimage.io.imsave(str(image_result_path/"{}_pred.jpg".format(image_id)), im_pred)

                mask_locations_y, mask_locations_x = np.where(target_mask == 1)
                im_target[mask_locations_y, mask_locations_x, :] \
                    = im_target[mask_locations_y, mask_locations_x, :] * 0.5 + np.array([0, 1, 1]) * 0.5
                skimage.io.imsave(str(image_result_path/"{}_target.jpg".format(image_id)), im_target)

                # gen result polygon
                df_poly = mask_to_poly(y_pred[0])

                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        f.write("{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio))

                        polygen_points = shapely.wkt.loads(row.wkt)
                        print(polygen_points)
                        points_x, points_y = polygen_points.exterior.xy

                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))
        del model
