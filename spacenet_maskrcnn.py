import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

KTF.set_session(session)

if __name__ == "__main__":
    from models.maskrcnn.spacenet import command_line
    # for training
    # command_line("['--command', 'train', '--weights', 'coco']")
    # for testing one figure
    # command_line("['--command', 'evaluate', '--weights', 'last']")
    # for testing all val figures
    command_line("['--command', 'evaluate_all', '--weights', 'last']")
