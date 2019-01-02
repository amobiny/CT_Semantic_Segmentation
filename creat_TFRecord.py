import tensorflow as tf
import sys
from PIL import Image
import glob
import numpy as np
from config import args


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def createDataRecord(out_filename, input_paths, label_paths):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(input_paths)):
        # print how many images are saved every 100 images
        if not i % 100:
            print('Processed images: {}/{}'.format(i, len(input_paths)))
            sys.stdout.flush()
        # Load the image and annotation (label)
        img = np.array(Image.open(input_paths[i]))
        label = np.array(Image.open(label_paths[i]))

        if img is None:
            continue

        height = img.shape[0]
        width = img.shape[1]
        channel = img.shape[2]
        # Create a feature
        feature = {'image': _bytes_feature(img.tostring()),
                   'label': _bytes_feature(label.tostring()),
                   'height': _int64_feature(height),
                   'width': _int64_feature(width),
                   'channel': _int64_feature(channel)}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    print('Done and saved in {}'.format(out_filename))
    writer.close()
    sys.stdout.flush()


def valid_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([], tf.string),
                                                                     'label': tf.FixedLenFeature([], tf.string),
                                                                     'height': tf.FixedLenFeature([], tf.int64),
                                                                     'width': tf.FixedLenFeature([], tf.int64),
                                                                     'channel': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)
    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # channel = tf.cast(features['channel'], tf.int32)

    # image_shape = tf.stack([height, width, channel])
    # label_shape = tf.stack([height, width, 1])
    image_shape = [args.height, args.width, args.channel]
    label_shape = [args.height, args.width]
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Reshape from [height * width * depth] to [height, width, depth].
    image = tf.reshape(image, image_shape)
    label = tf.cast(tf.reshape(label, label_shape), tf.int64)

    return image, label


def train_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([], tf.string),
                                                                     'label': tf.FixedLenFeature([], tf.string),
                                                                     'height': tf.FixedLenFeature([], tf.int64),
                                                                     'width': tf.FixedLenFeature([], tf.int64),
                                                                     'channel': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label'], tf.uint8)

    image_shape = [args.height, args.width, args.channel]
    label_shape = [args.height, args.width]
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Reshape from [height * width * depth] to [height, width, depth].
    image = tf.reshape(image, image_shape)
    label = tf.cast(tf.reshape(label, label_shape), tf.int64)

    if args.data_augment:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# batch_size = 16
# dataset = tf.data.TFRecordDataset(args.train_tfrecords)
# dataset_1 = dataset.map(train_parser, num_parallel_calls=batch_size)
# dataset_2 = dataset.map(valid_parser, num_parallel_calls=batch_size)
# dataset_1 = dataset_1.batch(batch_size)
# dataset_2 = dataset_2.batch(batch_size)
# iterator1 = dataset_1.make_one_shot_iterator()
# iterator2 = dataset_2.make_one_shot_iterator()
# next_batch1 = iterator1.get_next()
# next_batch2 = iterator2.get_next()
#
# with tf.Session() as sess:
#     x_batch1, y_batch1 = sess.run(next_batch1)
#     x_batch2, y_batch2 = sess.run(next_batch2)
#     print()

# if __name__ == '__main__':
#     path_to_input_images = './data_preparation/CamVid/val/*.png'
#     path_to_output_labels = './data_preparation/CamVid/valannot/*.png'
#     input_address = glob.glob(path_to_input_images)
#     label_address = glob.glob(path_to_output_labels)
#     createDataRecord('valid.tfrecords', input_address, label_address)

