import tensorflow as tf
import json
import os
import argparse
from src import transform
from PIL import Image
import base64

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoint", help="path to checkpoint to export")
parser.add_argument("--export", default="export", help="path to directory where to export to")
parser.add_argument("--test_img_path", help="path to test image to generate with the model")
a = parser.parse_args()


def export(checkpoint, img_shape):

    if img_shape is None:
        img_shape = [256,256,3]

    # placeholder for base64 string decoded to an png image
    input = tf.placeholder(tf.string, shape=[1])
    input_data = tf.decode_base64(input[0])
    input_image = tf.image.decode_png(input_data)

    # remove alpha channel if present
    input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:, :, :3], lambda: input_image)
    # convert grayscale to RGB
    input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image),
                          lambda: input_image)

    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
    input_image.set_shape(img_shape)
    # expected shape is (1, img_shape) because of batches
    batch_input = tf.expand_dims(input_image, axis=0)

    # create network
    batch_output = transform.net(batch_input)

    # clip RGB values to the allowed range and cast to uint8
    batch_output = tf.clip_by_value(batch_output, 0, 255)
    batch_output = tf.bitcast(tf.cast(batch_output, tf.int8), tf.uint8)
    output_data = tf.image.encode_png(batch_output[0])
    output = tf.convert_to_tensor([tf.encode_base64(output_data)])

    # save inputs and outputs to collection
    key = tf.placeholder(tf.string, shape=[1])
    inputs = {
        "key": key.name,
        "input": input.name
    }
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {
        "key": tf.identity(key).name,
        "output": output.name,
    }
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    export_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        if os.path.isdir(checkpoint):
            ckpt = tf.train.get_checkpoint_state(checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                restore_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            restore_saver.restore(sess, checkpoint)
        print("exporting model")
        export_saver.export_meta_graph(filename= os.path.join(a.export, "export.meta"))
        export_saver.save(sess, os.path.join(a.export, "export"), write_meta_graph=False)

    return


def test_export():

    with open(a.test_img_path, 'rb') as f:
        data = f.read()

    data64 = base64.b64encode(data)
    img_in64 = data64.decode('utf-8')
    img_in64 = img_in64.replace("+", "-").replace("/", "_")

    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        saver = tf.train.import_meta_graph(os.path.join(a.export, "export.meta"))

        saver.restore(sess, os.path.join(a.export, "export"))

        input_vars = json.loads(tf.get_collection("inputs")[0].decode("utf-8"))
        output_vars = json.loads(tf.get_collection("outputs")[0].decode("utf-8"))
        input_img = graph.get_tensor_by_name(input_vars["input"])
        output = graph.get_tensor_by_name(output_vars["output"])

        _output = sess.run(output, feed_dict={input_img: [img_in64]})

    o = _output[0]
    out_data = tf.decode_base64(o)
    out_img = tf.image.decode_png(out_data)

    with tf.Session() as sess:
        out = sess.run(out_img)

    im = Image.fromarray(out)
    im.save(os.path.join('output', os.path.basename(a.test_img_path)))


def main():

    # image shape [height, width, channels]
    image_shape = [1000, 1500, 3]

    if not os.path.exists(a.export):
        os.makedirs(a.export)

    print("Exporting checkpoint {} to {}".format(a.checkpoint, a.export))
    # export(a.checkpoint, image_shape)

    if a.test_img_path is not None:
        im = Image.open(a.test_img_path)
        if [im.width, im.height] != [image_shape[1], image_shape[0]]:
            raise ValueError('Test image must have the same shape as the exported model')
        if im.format != "PNG":
            raise ValueError('Test image must be a .png')

        print("Testing exported checkpoint with {}".format(a.test_img_path))
        test_export()



if __name__ ==  "__main__":
    main()

