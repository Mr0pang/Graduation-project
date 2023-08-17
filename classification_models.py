from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold
from keras import Sequential
import pickle
import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Flatten, ConvLSTM2D, InputLayer, Reshape, SimpleRNN, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score, auc, precision_score, recall_score
import numpy as np

# 获取训练数据
def arrange_data(data, labels):
    output_data = list()
    output_labels = list()
    for idx in range(len(data)):
        for segment in data[idx]:
            output_data.append(np.expand_dims(segment, axis=2))
            if labels[idx][0] == 1:
                output_labels.append(0)
            else:
                output_labels.append(1)
    output_data = np.array(output_data)
    output_labels = np.array(output_labels)
    return output_data, output_labels

def build_model_mlp(size_y, size_x, dim=1):
    model = Sequential()
    model.add(InputLayer(input_shape=(size_y, size_x, dim)))
    model.add(Flatten())

    model.add(Dense(units=64, activation='softmax'))
    model.add(Dropout(0.25))
    model.add(Dense(units=32, activation='softmax'))
    model.add(Dropout(0.25))

    # 添加输出层
    model.add(Dense(units=2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def build_model_lstm(size_y, size_x, dim=1):
    model = Sequential()
    model.add(InputLayer(input_shape=(size_y, size_x, dim)))
    # 添加LSTM层
    model.add(Reshape((120,32)))
    model.add(LSTM(units=16, input_shape=(120, 32)))

    # 添加输出层
    model.add(Dense(units=2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def build_model_rnn(size_y, size_x, dim=1):
    model = Sequential()
    model.add(InputLayer(input_shape=(size_y, size_x, dim)))
    model.add(Reshape((120,32)))
    model.add(SimpleRNN(16))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))

    # 添加输出层
    model.add(Dense(units=2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model




def build_model_cnn(size_y, size_x, dim=1):
    # input layer
    img_input = keras.layers.Input(shape=(size_y, size_x, dim))
    x = keras.layers.Conv2D(filters=30, kernel_size=(size_y, 3), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0))(img_input)
    x = keras.layers.MaxPooling2D(1, 10)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    output = keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0))(x)

    model = keras.models.Model(img_input, output)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def plot_roc(y_scores, y_test_one_hot, subject):
    fpr = {}
    tpr = {}
    roc_auc = {}
    y_scores_onehot = np.zeros_like(y_scores)
    y_scores_onehot[np.arange(len(y_scores)), y_scores.argmax(1)] = 1

    n_classes = 2 # 类别数
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制每个类别的 ROC 曲线
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='type %s ROC curve (area = %0.2f)' % (i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{subject}ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def get_precision_recall(y_scores, y_test_one_hot):
    y_scores_onehot = np.zeros_like(y_scores)
    y_scores_onehot[np.arange(len(y_scores)), y_scores.argmax(1)] = 1
    return precision_score(y_test_one_hot, y_scores_onehot, average='macro'),recall_score(y_test_one_hot, y_scores_onehot, average='macro')

# 在验证集上计算准确率
def trial_evaluate(model, data, labels):
    acc = 0.0
    # labels_onehot = keras.utils.to_categorical(labels.reshape(labels.shape[0]), num_classes=2)
    for idx in range(len(data)):
        test_data, test_label = arrange_data(np.expand_dims(data[idx], axis=0), np.expand_dims(labels[idx], axis=0))
        test_label = keras.utils.to_categorical(test_label, num_classes=2)
        loss, accuracy = model.evaluate(test_data, test_label)
        res = model.predict(test_data)
        if accuracy > 0.5:
            acc += 1.0
    acc = acc/len(data)
    return acc


# run classification
def run_classification(data, labels, session=(1,2,3,4,5), model_type="cnn"):
    kf = KFold(n_splits=10, shuffle=True)
    classification_acc = pd.DataFrame()
    classification_precision = pd.DataFrame()
    classification_recall = pd.DataFrame()
    for subject in data:
        subject_acc = list()
        subject_precision = list()
        subject_recall = list()
        input_data = list()
        target_labels = list()
        # combine trials data of target session
        [input_data.extend(data[subject]["session" + str(idx)]['input data']) for idx in session]
        [target_labels.extend(labels[subject]["session" + str(idx)]) for idx in session]
        input_data = np.array(input_data)
        target_labels = np.array(target_labels)

        model = None
        # 10 fold
        count = 0
        for train_index, test_index in kf.split(input_data):
            count += 1
            train_data, train_labels = arrange_data(input_data[train_index], target_labels[train_index])
            test_data, test_labels = arrange_data(input_data[test_index], target_labels[test_index])

            size_y, size_x = train_data[0].shape[0:2]

            # train_data_size = train_data.shape[0]
            # test_data_size = test_data.shape[0]

            train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
            test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

            # build model
            if model_type == 'cnn' and model == None:
                model = build_model_cnn(size_y, size_x)
            elif model_type == 'lstm' and model == None:
                model = build_model_lstm(size_y, size_x)
            elif model_type == 'rnn' and model == None:
                model = build_model_rnn(size_y, size_x)
            elif model_type == 'mlp' and model == None:
                model = build_model_mlp(size_y, size_x)

            print(train_data.shape)
            print('Training ------------')
            log_dir = f'./logs/fit/{datetime.now().strftime("%Y%m%d")}/{model_type}/' + datetime.now().strftime("%Y%m%d-%H%M%S")
            # train the model
            callbacks = [
                EarlyStopping(monitor='val_acc', patience=10, mode='max', restore_best_weights=True),
                TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
            ]
            model.fit(train_data, train_labels, epochs=300, batch_size=40, validation_split=.1, callbacks=callbacks)

            print('\nTesting ------------')
            # Evaluate the model with the metrics we defined earlier
            loss, accuracy = model.evaluate(test_data, test_labels)
            pred_labels = model.predict(test_data)
            plot_roc(pred_labels, test_labels, subject)
            precision, recall = get_precision_recall(pred_labels, test_labels)
            trial_acc = trial_evaluate(model, input_data[test_index], target_labels[test_index])

            print(count, subject)
            print('test loss: ', loss)
            print('test accuracy: ', accuracy)
            print('test precision:', precision)
            print('test recall:', recall)
            print('trial to trial accuracy: ', trial_acc)
            subject_acc.append(trial_acc)
            subject_precision.append(precision)
            subject_recall.append(recall)
        classification_acc[subject] = subject_acc
        classification_precision[subject] = subject_precision
        classification_recall[subject] = subject_recall
    return classification_acc, classification_precision, classification_recall




if __name__ == '__main__':
    # '''
    data_src = r"../data/gdf"
    labels_src = r"../data/labels"
    #
    # test_data_src = r"../tempdata/gdf"
    # test_label_src = r"../tempdata/labels"

    #
    # data, labels = run_sig_processing(test_data_src, test_label_src, band_type=2)
    # print("save data")
    # Saving the data and labels:
    # with open('test_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([data, labels], f)
    # f.close()
    # '''
    # Getting back the data and labels:
    with open('temp_data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        data, labels = pickle.load(f)
    model_type = "rnn"

    acc, precision, recall = run_classification(data, labels, session=[1], model_type=model_type)
    # print(res)
    acc.to_csv(f'BP_acc_{model_type}.csv', encoding="utf-8")
    precision.to_csv(f'BP_precision_{model_type}.csv', encoding="utf-8")
    recall.to_csv(f'BP_recall_{model_type}.csv', encoding="utf-8")
    print("cnn classification")