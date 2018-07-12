import numpy as np
import scipy.io as sio
import pickle
import datetime
import os


# padding_mode = {'reflect', 'mean'}
def get_data_in_window(window_size=3, features=None, padding_mode='reflect', three_classes=True):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd number but was {}!".format(window_size))
    if features is None:
        features = [20, 120]

    offset = int(window_size / 2)

    X, Y = read_img()
    Y = Y - 1
    data_size = X.shape[0] * X.shape[1]

    X_padded = np.pad(X, int(window_size / 2), padding_mode)
    Y_padded = np.pad(Y, int(window_size / 2), padding_mode)

    num_classes = window_size * window_size
    data = np.ndarray((data_size, num_classes + len(features) * num_classes))

    counter_index = 0
    for i, row in enumerate(Y_padded[offset:Y_padded.shape[0] - offset], start=offset):
        for j, column in enumerate(row[offset:Y_padded.shape[1] - offset], start=offset):
            img_data = list()
            class_data = list()
            # slinding window
            for k in range(j - offset, j + offset + 1):
                for l in range(i - offset, i + offset + 1):
                    # extract features
                    if three_classes:
                        class_data.append(Y_padded[l][k])
                    else:
                        class_data.append(0.0 if Y_padded[l][k] == 2.0 else Y_padded[l][k])
                    for m in features:
                        img_data.append(X_padded[l][k][m])
            data[counter_index][:] = np.append(img_data, class_data)
            counter_index += 1

    return data


def get_number_of_classes():
    return number_of_classes


def get_data(size=300, values=None):
    if values is None:
        values = [20, 120]

    vocab = {1.0: 'healthy', 2.0: 'diseased', 3.0: 'stem'}
    X, Y = read_img()
    counter_healthy = 0
    counter_disease = 0
    counter_stem = 0
    counter_data = 0
    data = np.ndarray((3 * size, len(values) + 1))
    for i, row in enumerate(Y):
        for j, col in enumerate(row):
            if vocab[col] == 'healthy' and counter_healthy < size:
                hyp_value = np.array(X[i, j][values])
                data[counter_data] = np.append(hyp_value, [int(col - 1)])
                counter_data += 1
                counter_healthy += 1
            elif vocab[col] == 'diseased' and counter_disease < size:
                hyp_value = np.array(X[i, j][values])
                data[counter_data] = np.append(hyp_value, [int(col - 1)])
                counter_data += 1
                counter_disease += 1
            elif vocab[col] == 'stem' and counter_stem < size:
                hyp_value = np.array(X[i, j][values])
                data[counter_data] = np.append(hyp_value, [int(col - 1)])
                counter_data += 1
                counter_stem += 1
            if counter_data >= 3 * size:
                return data
    raise ValueError("Not enough values in image (size: {})".format(size))


def read_img(src="cerc15dai175.mat"):

    try:
        data = np.load("{}.npz".format(src.split(".")[0]))
        return data["X"], data["Y"]
    except FileNotFoundError:
        data = sio.loadmat(src)

        dim = data["dim"].reshape(-1)
        input_image = data["counts"]
        input_image = input_image.T
        input_image = np.reshape(input_image, newshape=[dim[0], dim[1], dim[2]])

        input_image = input_image.astype(float)
        input_image -= np.min(input_image)
        input_image /= np.max(input_image)

        np.savez("{}.npz".format(src.split(".")[0]), X=input_image, Y=data["labels"].reshape(dim[0], dim[1]))

        return input_image, data["labels"].reshape(dim[0], dim[1])


def save_spn(spn, window_size, feature_list, min_instances_slice, number_of_classes, directory="/trainedSPNs"):
    if not os.path.exists(os.getcwd() + directory):
        os.makedirs(directory)
    path = os.getcwd()+directory
    time = datetime.datetime.now()
    spn_name = "spn_{}".format(time)
    dic = {"spn_name": spn_name, "window_size":window_size, "feature_list": feature_list,
           "min_instances_slice": min_instances_slice, "number_of_classes": number_of_classes}
    pickle.dump(spn, open("{}/{}.p".format(path, spn_name), "wb"))
    pickle.dump(dic, open("{}/dic_{}.p".format(path, time), "wb"))
    print("New trained SPN was saved!")


def load_spn(window_size, feature_list, min_instances_slice, number_of_classes, directory="/trainedSPNs"):
    path = os.getcwd()+ directory
    if not os.path.exists(path):
        return
    for file in os.listdir(path):
        if file.startswith("dic"):
            dic = pickle.load(open(path+"/"+file, "rb"))
            if dic["window_size"] == window_size and dic["feature_list"] == feature_list and \
                    dic["min_instances_slice"] == min_instances_slice and dic["number_of_classes"] == number_of_classes:
                print("Pretrained SPN was loaded!")
                try:
                    return pickle.load(open(path+"/"+dic["spn_name"]+".p", "rb"))
                except FileNotFoundError as e:
                    print("SPN not found!")
                    print(e)
                    return None
    print("No pretrained SPN was found!")


if __name__ == '__main__':
    #save_spn(4,3,[1,2,3,5,6], 1000, 3)
    print(load_spn(3,[1,2,3,5,6], 1000, 3))
