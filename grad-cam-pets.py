#http://www.hackevolve.com/where-cnn-is-looking-grad-cam/

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2
import sys
from PIL import Image
import PIL
import os
import numpy
model = VGG16(weights="imagenet")

# img_path = "/home/deepaknayak/Downloads/keras-grad-cam-master/1923.jpg"
img_path = "/home/deepaknayak/Documents/Reinforcement-learning/data-collector-master/AFFORDANCE_Dataset/3/"

def grad_cam_analysis(img):
    image_crop = img
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("block5_conv3")

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    open_cv_image = numpy.array(image_crop)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    heatmap = cv2.resize(heatmap, (open_cv_image.shape[1], open_cv_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(open_cv_image, 0.6, heatmap, 0.4, 0)
    cv2.imshow("Original", open_cv_image)
    cv2.imshow("GradCam", superimposed_img)
    cv2.waitKey(0)


for root, dirs, files in os.walk(img_path):
    for f in files:
        file_path = os.path.relpath(os.path.join(root, f), ".")
        img = image.load_img(file_path)  # , target_size=(224, 224)
        # img_converted = image.array_to_img(crop_img)
        cropped_TL = img.crop((427, 45, 625, 311))
        cropped_CAR = img.crop((218, 108, 562, 380))
        # cropped_CAR.show()
        # cropped.show()
        img_TL = cropped_TL.resize((224, 224), Image.ANTIALIAS)
        img_CAR = cropped_CAR.resize((224, 224), Image.ANTIALIAS)
        # img = image.load_img(img_converted, target_size=(224, 224))
        grad_cam_analysis(img_TL)
        grad_cam_analysis(img_CAR)












