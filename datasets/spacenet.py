import scipy
import tqdm
import tables as tb
import pandas as pd
import numpy as np
import skimage.transform
import skimage.io
import rasterio
import shapely.wkt
import rasterio.features
import shapely.wkt
import shapely.ops
import shapely.geometry

from utils.log import logger

ORIGINAL_SIZE = 650
INPUT_SIZE = 256

MIN_POLYGON_AREA = 30


def image_mask_from_summary(df, image_id):
    im_mask = np.zeros((ORIGINAL_SIZE, ORIGINAL_SIZE))

    if len(df[df.ImageId == image_id]) == 0:
        raise RuntimeError("ImageId not found on summaryData: {}".format(
            image_id))

    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = [round(float(pp[0])) for pp in coords]
            y = [round(float(pp[1])) for pp in coords]
            yy, xx = skimage.draw.polygon(y, x, (ORIGINAL_SIZE, ORIGINAL_SIZE))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (ORIGINAL_SIZE, ORIGINAL_SIZE))
                im_mask[yy, xx] = 0
    # dont resize here, since we want real image for output
    # im_mask = skimage.transform.resize(im_mask, (INPUT_SIZE, INPUT_SIZE))
    # im_mask = (im_mask > 0.5).astype(np.uint8)
    return im_mask


def parse_and_save_target(path_dir, image_id_csv, image_mask_h5):
    image_list = []
    total_image_df = pd.DataFrame()
    for csv_f in path_dir.rglob('*.csv'):
        df = pd.read_csv(csv_f)
        total_image_df = total_image_df.append(df)
        image_list.extend(df['ImageId'].tolist())

    image_list = list(set(image_list))
    # for reduce running time
    # image_list = image_list[:100]
    logger.info("image list: {}".format(image_list))
    logger.info("image list length: {}".format(len(image_list)))
    image_list_df = pd.DataFrame(columns=['id', 'ImageId'], data=[i for i in zip(range(len(image_list)), image_list)])
    image_list_df.to_csv(image_id_csv, encoding='utf-8')

    with tb.open_file(image_mask_h5, 'w') as f:
        for image_id in tqdm.tqdm(image_list):
            im_mask = image_mask_from_summary(total_image_df, image_id)
            atom = tb.Atom.from_dtype(im_mask.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, image_id, atom, im_mask.shape,
                                 filters=filters)
            ds[:] = im_mask
    return image_list


def get_image_path_based_type_imageid(path_dir, kind, image_id):
    return str(path_dir / kind / "{}_{}.tif".format(kind, image_id))


def calc_multiband_norm(path_dir, image_list, image_feature_norm_csv,
                        kind='RGB-PanSharpen', channel_count=3, max_sample=100):
    band_values = {k: [] for k in range(channel_count)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(channel_count)}
    #  first get all data, then use first part 0:1000 to calc threshold, then update all data
    for image_id in tqdm.tqdm(image_list[:max_sample]):
        image_loc = get_image_path_based_type_imageid(path_dir, kind, image_id)
        with rasterio.open(image_loc, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(channel_count):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    logger.info("Calc percentile point for normalization")
    for i_chan in range(channel_count):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)

    stat = dict()
    stat['path'] = path_dir
    for chan_i in band_cut_th.keys():
        stat['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        stat['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    pd.DataFrame(stat, index=[0]).to_csv(image_feature_norm_csv, index=False)


def parse_and_save_feature(path_dir, image_id_csv, image_feature_h5, image_feature_norm_csv, kind, channel_count):
    image_df = pd.read_csv(image_id_csv)
    image_list = image_df['ImageId'].tolist()

    calc_multiband_norm(path_dir, image_list, image_feature_norm_csv,
                        kind=kind, channel_count=channel_count, max_sample=500)

    bandstats = pd.read_csv(image_feature_norm_csv)
    _, bandstats = next(bandstats.iterrows())

    with tb.open_file(image_feature_h5, 'w') as h5_f:
        for image_id in tqdm.tqdm(image_list, total=len(image_list)):
            image_loc = get_image_path_based_type_imageid(path_dir, kind, image_id)
            with rasterio.open(image_loc, 'r') as img_f:
                values = img_f.read().astype(np.float32)

                for chan_i in range(channel_count):
                    min_val = bandstats['chan{}_min'.format(chan_i)]
                    max_val = bandstats['chan{}_max'.format(chan_i)]
                    values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                    values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                values = np.swapaxes(values, 0, 2)
                values = np.swapaxes(values, 0, 1)
                # dont resize here, since we want real image for output
                # im = skimage.transform.resize(values, (INPUT_SIZE, INPUT_SIZE))
                im = values
            atom = tb.Atom.from_dtype(im.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = h5_f.create_carray(h5_f.root, image_id, atom, im.shape,
                                    filters=filters)
            ds[:] = im


def load_target(image_id_csv, image_mask_h5):
    # note, only image ids in target make sense
    image_df = pd.read_csv(image_id_csv)
    image_list = image_df['ImageId'].tolist()

    y = []
    with tb.open_file(image_mask_h5, 'r') as f:
        for idx, image_id in enumerate(image_list):
            mask = np.array(f.get_node('/' + image_id))
            mask = (mask > 0.5).astype(np.uint8)
            y.append(mask)
    y = np.array(y)
    y = y.reshape((-1, INPUT_SIZE, INPUT_SIZE, 1))

    return y


def load_feature(image_id_csv, image_feature_h5):
    image_df = pd.read_csv(image_id_csv)
    image_list = image_df['ImageId'].tolist()

    features = []
    with tb.open_file(image_feature_h5, 'r') as f:
        for idx, image_id in enumerate(image_list):
            im = np.array(f.get_node('/' + image_id))
            features.append(im)
    features = np.array(features)

    return features


def load_batch_data(image_list, image_feature_h5, image_target_h5, batch_size, kind='train',
                    use_shuffle=True, keep_target_dim=False):
    x = []
    y = []

    with tb.open_file(image_feature_h5, 'r') as f_feature:
        with tb.open_file(image_target_h5, 'r') as f_target:
            while True:
                if use_shuffle:
                    np.random.shuffle(image_list)
                for idx, image_id in enumerate(image_list):
                    x_mask = np.array(f_feature.get_node('/' + image_id))
                    x_mask = skimage.transform.resize(x_mask, (INPUT_SIZE, INPUT_SIZE))
                    x.append(x_mask)
                    y_mask = np.array(f_target.get_node('/' + image_id))
                    # for validating or testing stage, we need original masks
                    if not keep_target_dim:
                        y_mask = skimage.transform.resize(y_mask, (INPUT_SIZE, INPUT_SIZE))
                    y_mask = (y_mask > 0.5).astype(np.uint8)
                    y.append(y_mask)
                    if len(x) >= batch_size:
                        x = np.array(x)
                        y = np.array(y)
                        if not keep_target_dim:
                            y = y.reshape((-1, INPUT_SIZE, INPUT_SIZE, 1))
                        else:
                            y = y.reshape((-1, ORIGINAL_SIZE, ORIGINAL_SIZE, 1))
                        yield x, y
                        x = []
                        y = []


def split_train_validation_batch(image_id_csv, image_feature_h5, image_target_h5,
                                 train_batch_size=20, val_batch_size=5, train_ratio=0.8,
                                 use_shuffle=True, keep_target_dim=False):
    image_df = pd.read_csv(image_id_csv)
    image_list = image_df['ImageId'].tolist()

    # pls dont shuffle here, since we want fixed training and validating set
    # if use_shuffle:
    #     np.random.shuffle(image_list)
    validation_start = int(len(image_list) * train_ratio)
    print("validation_start {}".format(validation_start))
    train_data_generator = load_batch_data(image_list[:validation_start], image_feature_h5, image_target_h5,
                                           train_batch_size, 'train', use_shuffle, keep_target_dim)
    val_data_generator = load_batch_data(image_list[validation_start:], image_feature_h5, image_target_h5,
                                         val_batch_size, 'val', use_shuffle, keep_target_dim)
    return train_data_generator, val_data_generator, image_list[:validation_start], image_list[validation_start:]


def split_train_validation(x, y, train_ratio=0.8):
    assert x.shape[0] == y.shape[0]
    seq = list(range(x.shape[0]))
    np.random.shuffle(seq)
    train_len = int(len(seq) * train_ratio)
    x = x[seq]
    y = y[seq]
    x_train = x[:train_len]
    x_val = x[train_len:]
    y_train = y[:train_len]
    y_val = y[train_len:]

    return x_train, x_val, y_train, y_val


def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
    """
    Convert from 256x256 mask to polygons on 650x650 image
    """
    mask = (skimage.transform.resize(mask, (ORIGINAL_SIZE, ORIGINAL_SIZE)) > 0.5).astype(np.uint8)
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    poly_list = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df


if __name__ == '__main__':
    logger.info("no actions now")
