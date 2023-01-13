# 1)The needed libaraies for the model
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

# 2)importing my dataset
# 2.1) read the images files
root_dir = "dataset/images/images"  # the image file
# files = os.path.join(root_dir)       # takes the path by os package and joins directories ( I do not need it )
File_names = os.listdir(root_dir)  # list the files in the directory ( now files_names has all the images )
print("This is the list of all the files present in the path given to us:\n")
print(File_names)
# 2.2)read the labels of the images
data_set = pd.read_csv("dataset/pokemon.csv")  # reads csv
print(data_set.head(5))

# 2.3) putting each image with its label in a dictionary
data_dict = {}
for key, val in zip(data_set["Name"], data_set["Type1"]):  # mapping the name of the image with its label from type 1
    data_dict[key] = val
print(data_dict)
# 2.4) Extracting the labels
labels = data_set["Type1"].unique()  # to avoid any duplications
# 2.5) mapping each label with a unique id
ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
labels_ids = dict(zip(labels, ids))
print(labels_ids)
# 2.6) creating the final version of images amd labels that I will work with
final_images = []
final_labels = []
count = 0
# files = os.path.join(root_dir)
for file in File_names:
    count += 1
    img = cv2.imread(os.path.join(root_dir, file),
                     cv2.COLOR_BGR2GRAY)  # it takes a path and a flag to read a colored image
    label = labels_ids[data_dict[file.split(".")[0]]]
    # 2.6.1) append img in final_images list
    final_images.append(np.array(img))
    # 2.6.2) append label in final_labels list
    final_labels.append(np.array(label))

# 2.7)converting lists into numpy array and normalizing and reshaping the data
final_images = np.array(final_images, dtype=np.float32) / 255.0
final_labels = np.array(final_labels, dtype=np.int8).reshape(809, 1)  # 809 no. of images

# 3) Constructing the ANN architecture
model = tf.keras.Sequential([  # sequential is to create a stack of layers
    tf.keras.layers.Flatten(input_shape=(120, 120, 3)),
    # flatten the input layer and transfer input to the hidden layer  120*120 size in 3 channels
    tf.keras.layers.Dense(100, activation='relu'),  # takes unit of output and activation function
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(18)
])
model.summary()

# 4) compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 5) fit model
history = model.fit(final_images, final_labels, epochs=50)  # iterations on training
# 6) prediction
predicted_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = predicted_model.predict(final_images)


def predict_output(img_path):
    img_tmp = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    chosen_img = np.array(img_tmp)
    final_image = np.array([chosen_img], dtype=np.float32) / 255.0
    print(predicted_model.predict(final_image))
    result = predicted_model.predict(final_image)
    result2 = np.array(result)
    result_id = result2.argmax()
    print(result2.argmax())
    for key in labels_ids:
        if result_id == labels_ids[key]:
            return key
            # print(key)


#x=predict_output("dataset/images/images/bulbasaur.png")
#print(x)
import GUI as gui
gui.mainWin.mainloop()