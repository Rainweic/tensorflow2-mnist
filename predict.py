import tensorflow as tf
from src import utils
from model import resnet

# load model and weights
model = resnet.small_resnet()
model.load_weights("./weights/small_resnet")

# load test images
images_list = utils.get_images_from_csv("./datasets/test.csv", train=False)
images_list = utils.preprocess(images=images_list, train=False)

# get predict result
out = model.predict(images_list)
out = tf.argmax(out, axis=1).numpy().tolist()
out = [[i+1, data] for i, data in enumerate(out)]

# write result into csv
utils.write_to_csv("./datasets/result.csv", out)

