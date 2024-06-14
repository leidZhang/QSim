import torch
import tf2onnx
import onnx2pytorch
import tensorflow as tf
from keras.optimizers import Adam

from core.models.tf.cnn_auto_drive import get_model


def convert_model(tf_model_dir):
    optimizer = Adam(1e-4, decay=0.0)
    tf_model = get_model(optimizer)
    tf_model.load_weights(tf_model_dir)

    onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
    torch_model = onnx2pytorch.ConvertModel(onnx_model)

    print("TORCH MODEL")
    print(torch_model)

if __name__ == "__main__":
    convert_model(tf_model_dir="C:\\Users\\1111\\OneDrive\\Desktop\\acc_2024\\src\\model_best_fit.h5")