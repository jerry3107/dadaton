# 數據集的位置
avatar_img_path = "./data"

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Dropout, Conv2D, Dense, LeakyReLU, Input, Reshape, Flatten, Conv2DTranspose
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


class SpectralNormalization(tf.keras.constraints.Constraint):
    def __init__(self, n_iter = 5):
        self.n_iter = n_iter
    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a = True)
            v /= tf.norm(v)
            u = tf.matmul(w, v)
            u /= tf.norm(u)
        spec_norm = tf.matmul(u, tf.matmul(w, v), transpose_a = True)
        return input_weights/spec_norm

def load_data():
    """
    加載數據集
    :return: 返回numpy數組
    """
    all_images = []
    # 從本地文件讀取圖片加載到images_data中。
    for image_name in os.listdir(avatar_img_path):
        try:
            image = cv2.cvtColor(
                cv2.resize(
                    cv2.imread(os.path.join(avatar_img_path, image_name), cv2.IMREAD_COLOR),
                    (64, 64)
                    ),cv2.COLOR_BGR2RGB
                )
            all_images.append(image)
        except:
            continue
        
    all_images = np.array(all_images)
    # 將圖片數值變成[-1,1]
    all_images = (all_images - 127.5) / 127.5
    # 將數據隨機排序
    np.random.shuffle(all_images)
    return all_images
img_dataset=load_data()
def show_images(images, index = -1):
    """
    展示並保存圖片
    :param images: 需要show的圖片
    :param index: 圖片名
    :return:
    """
    plt.figure()
    for i, image in enumerate(images):
        ax = plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(image)
    plt.savefig(f"E:\\training_pictures\data_{index}.png")
    plt.close()
show_images(img_dataset[0: 25])
# noise的维度
noise_dim = 100
# 圖片的shape
image_shape = (64,64,3)

def build_G():
    """
    構建生成器
    :return:
    """
    model = Sequential()
    model.add(Input(shape=noise_dim))

    model.add(Dense(128 * 32 * 32))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((32, 32, 128)))
    
    model.add(Conv2D(256, 5, padding='same', kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))

    model.add(Conv2DTranspose(256, 4, strides=2, padding='same', kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, 5, padding='same', kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(256, 5, padding='same', kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2D(3, 7, activation='tanh', padding='same'))

    return model

G = build_G()
def build_D():
    """
    構建判别器
    :return: 
    """
    model = Sequential()
    
    # 卷積層
    model.add(Conv2D(128, 3, input_shape = image_shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, strides=2, kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, strides=2, kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, strides=2, kernel_constraint=SpectralNormalization()))
    model.add(LeakyReLU(0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 0.0001, beta_1 = 0.5))
    
    return model
D = build_D()
def build_gan():
    """
    構建GAN網路
    :return:
    """
    # 暫停判别器，也就是在訓練的時候只優化G的網路權重，而對D保持不變
    D.trainable = False
    # GAN網路的輸入
    gan_input = Input(shape=(noise_dim))
    # GAN網路的輸出
    gan_out = D(G(gan_input))
    # 構建網路
    gan = Model(gan_input, gan_out)
    # 編譯GAN網路，使用Adam優化器，以及加上交叉熵損失函数（一般用於二分類）
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 0.0001, beta_1 = 0.5))
    return gan
GAN = build_gan()
def sample_noise(batch_size):
    """
    隨機產生正態分布（0，1）的noise
    :param batch_size:
    :return: 返回的shape為(batch_size,noise)
    """
    return np.random.normal(size=(batch_size, noise_dim))

def smooth_pos_labels(y):
    """
    使true label的值的範圍為[0.8,1]
    :param y:
    :return:
    """
    return y - (np.random.random(y.shape) * 0.2)

def smooth_neg_labels(y):
    """
    使fake label的值的範圍為[0.0,0.3]
    :param y:
    :return:
    """
    return y + np.random.random(y.shape) * 0.3
def load_batch(data, batch_size,index):
    """
    按批次加載圖片
    :param data: 圖片數據集
    :param batch_size: 批次大小
    :param index: 批次序號
    :return:
    """
    return data[index * batch_size: (index+1) * batch_size]

def train(epochs, batch_size):
    """
    訓練函數
    :param epochs: 訓練次數
    :param batch_size: 批尺寸
    :return:
    """
    # 生成器損失
    generator_loss = 0
    # img_dataset.shape[0] / batch_size 代表這個數據可以分為幾個批次進行訓練
    n_batches = int(img_dataset.shape[0] / batch_size)
    for i in range(epochs):
        for index in range(n_batches):
            # 按批次加載數據
            x = load_batch(img_dataset, batch_size, index)
            # 產生noise
            noise = sample_noise(batch_size)
            # G網路產生圖片
            generated_images = G.predict(noise)
            # 產生為1的標籤
            y_real = np.ones(batch_size)
            # 將1標籤的範圍變成[0.8 , 1.0]
            y_real = smooth_pos_labels(y_real)
            # 產生为0的標籤
            y_fake = np.zeros(batch_size)
            # 將0標籤的範圍變成[0.0 , 0.3]
            y_fake = smooth_neg_labels(y_fake)
            # 訓練真圖片loss
            d_loss_real = D.train_on_batch(x, y_real)
            # 訓練假圖片loss
            d_loss_fake = D.train_on_batch(generated_images, y_fake)
            # 產生為1的標籤
            y_real = np.ones(batch_size)
            # 訓練GAN網路，input = fake_img ,label = 1
            generator_loss = GAN.train_on_batch(noise, y_real)
        
        print(f'[Epoch {i}]. Discriminator real_img : {d_loss_real}. Discriminator fake_img : {d_loss_fake}. Generator_loss: {generator_loss}.')

        # 每個epoch保存一次。
        if i%1 == 0:
            # 隨機產生(25,100)的noise
            test_noise = sample_noise(25)
            # 使用G網路生成25張圖偏
            test_images = G.predict(test_noise)
            # show 預測 img
            show_images(test_images, i)
train(epochs = 100, batch_size = 128)

G.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 0.0001, beta_1 = 0.5))
D.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = 0.0001, beta_1 = 0.5))
G.save('AnimeFace_Generator')
D.save('AnimeFace_Discriminator')