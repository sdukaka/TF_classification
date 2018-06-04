from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim
tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'test_list', '', 'Test image list.')
tf.app.flags.DEFINE_string(
    'test_dir', '.', 'Test image directory.')
tf.app.flags.DEFINE_string(
    'label_dir', '.', 'Label info directory.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'Batch size.')
tf.app.flags.DEFINE_integer(
    'num_classes', 5, 'Number of classes.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.test_list:
        raise ValueError('You must supply the test list with --test_list')
    tf.logging.set_verbosity(tf.logging.INFO)

    # TODO: map info for label-id modify by sdukaka
    label_info = "../../../data/chair/labels.txt"
    label_map = {}
    with open(label_info, 'r') as label_information:
        lines = label_information.readlines()
        for line in lines:
            line = line.strip('\n').split(':')
            label_map[line[0]] = line[1]
    # print("label map information: ", label_map)

    # create save dir
    SAVE_DIR = os.path.join(FLAGS.label_dir, "../label")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        test_image_size = FLAGS.test_image_size or network_fn.default_image_size
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        batch_size = FLAGS.batch_size
        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 5)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        test_ids = [line.strip() for line in open(FLAGS.test_list)]
        tot = len(test_ids)
        results = list()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            # TODO: input is the whole image for testing
            for idx in range(0, len(test_ids)):
                # image part
                image_id = test_ids[idx]
                image_path = os.path.join(FLAGS.test_dir, "img/{}.png".format(image_id))
                print("test image path: ", image_path)
                image = open(image_path, 'rb').read()
                image = tf.image.decode_png(image, channels=3)
                # wait for classification
                images = list()
                results = list()

                # label part
                label_path = os.path.join(FLAGS.label_dir, "{}.txt".format(image_id))
                print("test label path: ", label_path)
                with open(label_path, 'r') as label_record:
                    label_info = label_record.readlines()
                    for label_sub in label_info:
                        label_sub = label_sub.strip('\n').split('\t')
                        # Get chair bbox for visualization
                        if int(label_sub[15]) == 0:
                            x1 = max(int(label_sub[1]), 0)
                            y1 = max(int(label_sub[2]), 0)
                            x2 = min(int(label_sub[3]), 640)
                            y2 = min(int(label_sub[4]), 480)
                            # get chir infomation
                            image_patch = image[y1:y2, x1:x2, :]
                            processed_image = image_preprocessing_fn(image_patch, test_image_size, test_image_size)
                            processed_image = sess.run(processed_image)
                            images.append(processed_image)
                if len(images) != 0:
                    images = np.array(images)
                    predictions = sess.run(logits, feed_dict={tensor_input: images}).indices
                    for i in range(0, len(images)):
                        # print("get results for chair:", predictions[i].tolist())
                        label_list = [label_map[str(class_id)] for class_id in predictions[i].tolist()]
                        results.append(label_list)
                        # print("{} {}".format(image_id, label_list))
                save_new_file = os.path.join(SAVE_DIR, "{}.txt".format(image_id))
                original_file = open(label_path)
                result_count = 0
                with open(save_new_file, 'w') as modify_class:
                    for each_line in original_file:
                        line_info = each_line.strip('\n').split('\t')
                        if int(line_info[15]) == 0:
                            line_info[0] = "Model#{}".format(results[result_count][0])
                            for line_idx, line_s in enumerate(line_info):
                                if line_idx == 0:
                                    line_write = line_s
                                else:
                                    line_write = "{}\t{}".format(line_write, line_s)
                            for line_idx in range(5):
                                line_write = "{}\t{}".format(line_write, results[result_count][line_idx])
                            line_write = "{}\n".format(line_write)
                            modify_class.writelines(line_write)
                            result_count += 1
                        else:
                            modify_class.writelines(each_line)

            # for idx in range(0, tot, batch_size):
            #     images = list()
            #     idx_end = min(tot, idx + batch_size)
            #     print(idx)
            #     for i in range(idx, idx_end):
            #         image_id = test_ids[i]
            #         test_path = os.path.join(FLAGS.test_dir, image_id)
            #         print("test_image path: ", test_path)
            #         image = open(test_path, 'rb').read()
            #         image = tf.image.decode_jpeg(image, channels=3)
            #         processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
            #         processed_image = sess.run(processed_image)
            #         images.append(processed_image)
            #     images = np.array(images)
            #     predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices
            #     for i in range(idx, idx_end):
            #         print("{} {}".format(image_id, predictions[i - idx].tolist()))


if __name__ == '__main__':
    tf.app.run()