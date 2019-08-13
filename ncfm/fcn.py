bm = 'valid'
tr = False


def fcn(mode=32):

    img_input = Input(shape=(3, 500, 500),name = 'img_input')

    # Vgg_Block_1
    pad_1 = ZeroPadding2D(padding=(100, 100))(img_input)
    conv1_1 = Convolution2D(64, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv1_1')(pad_1)   # 1
    conv1_2 = Convolution2D(64, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv1_2')(conv1_1)  # 3
    max_pool_1 =  MaxPooling2D((2, 2), strides=(2, 2),
                               name='max_pool_1')(conv1_2)   # 4

    # Vgg_Block_2
    conv2_1 = Convolution2D(128, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv2_1')(max_pool_1)  # 6
    conv2_2 = Convolution2D(128, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv2_2')(conv2_1)  # 8
    max_pool_2 = MaxPooling2D((2, 2), trainable=tr, strides=(2, 2),
                              name='max_pool_2')(conv2_2)  # 9

    # Vgg_Block_3
    conv3_1 = Convolution2D(256, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu',
                            name='conv3_1')(max_pool_2)  # 11
    conv3_2 = Convolution2D(256, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv3_2')(conv3_1)  # 13
    conv3_3 = Convolution2D(256, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv3_3')(conv3_2)  # 15
    max_pool_3 = MaxPooling2D((2, 2), strides=(2, 2),
                              name='max_pool_3')(conv3_3)  # 16

    # Vgg_Block_4
    conv4_1 = Convolution2D(512, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv4_1')(max_pool_3)  # 18
    conv4_2 = Convolution2D(512, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv4_2')(conv4_1)  # 20
    conv4_3 = Convolution2D(512, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv4_3')(conv4_2)  # 22
    max_pool_4 = MaxPooling2D((2, 2), strides=(2, 2),name='max_pool_4')(conv4_3)  # 23

    # Vgg_Block_5
    conv5_1 = Convolution2D(512, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv5_1')(max_pool_4)  # 25
    conv5_2 = Convolution2D(512, 3, 3, trainable=tr, border_mode=bm,
                            activation='relu', name='conv5_2')(conv5_1)  # 27
    conv5_3 = Convolution2D(512, 3, 3,trainable=tr, border_mode=bm,
                            activation='relu', name='conv5_3')(conv5_2)  # 29
    max_pool_5 = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_5')(conv5_3)  # 30

    # New_FCN_Block
    fc6 = Convolution2D(4096, 7, 7, init='he_normal', border_mode='same',
                        activation='relu', name='fc6')(max_pool_5)
    drop6 = Dropout(0.5, name='drop6')(fc6)
    fc7 = Convolution2D(4096, 1, 1, init='he_normal', activation='relu',
                        name='fc7')(drop6)
    drop7 = Dropout(0.5, name='drop7')(fc7)

    score_fr = Convolution2D(21, 1, 1, init='he_normal',
                             name='score_fr')(drop7)  # 16x16

    if mode == 32:
        upscore2  = Deconvolution2D(21, 64, 64,
                                    output_shape=(None, 21, 544, 544),
                                    subsample=(32, 32), bias=False,
                                    name='upscore2')(score_fr)
        crop = Cropping2D(cropping=((22, 22), (22, 22)))(upscore2)  # crop(n.upscore8, n.data)
        reshape = Reshape((21, 500 * 500))(crop)
        trans =  Permute((2, 1))(reshape)
        out = Activation('softmax')(trans)
        model = Model(input=[img_input], output=[out])

    if mode == 16:
        upscore2  = Deconvolution2D(21, 4, 4,
                                    output_shape=(None, 21, 34, 34),
                                    subsample=(2, 2), bias=False,
                                    name='upscore2_16')(score_fr)
        score_pool_4 = Convolution2D(21, 1, 1, init='he_normal')(max_pool_4)  # 38x38
        score_pool_4c = Cropping2D(cropping=((2, 2), (2, 2)))(score_pool_4)  # crop(n.score_pool4, n.upscore2)
        fuse_pool_4 = merge([upscore2, score_pool_4c], mode='sum', concat_axis=1)  # 34x34
        upscore_pool_4  = Deconvolution2D(21, 32, 32,
                                          output_shape=(None, 21, 560, 560),
                                          subsample=(16, 16), bias=False)(fuse_pool_4)
        crop = Cropping2D(cropping=((30, 30), (30, 30)))(upscore_pool_4)  # crop(n.upscore8, n.data)
        reshape = Reshape((21, 500 * 500))(crop)
        trans =  Permute((2, 1))(reshape)
        out = Activation('softmax')(trans)
        model = Model(input=[img_input], output=[out])
        # ---------------- LOAD mode=32 weights ---------------
        model.load_weights('mode_32',by_name=True)

    if mode == 8:
        upscore2  = Deconvolution2D(21, 4, 4,
                                    output_shape=(None, 21, 34, 34),
                                    subsample=(2, 2), bias=False)(score_fr)
        score_pool_4 = Convolution2D(21, 1, 1, init='he_normal')(max_pool_4)
        score_pool_4c = Cropping2D(cropping=((2, 2), (2, 2)))(score_pool_4)  # crop(n.score_pool4, n.upscore2)
        fuse_pool_4 = merge([upscore2, score_pool_4c], mode='sum', concat_axis=1)
        # --------------- LOAD mode=16 weights --------------
        model = Model(input=[img_input], output=[fuse_pool_4])
        model.load_weights('mode_16', by_name=True)
        # ===================================================
        upscore_pool_4  = Deconvolution2D(21, 4, 4,
                                          output_shape=(None, 21, 70, 70),
                                          subsample=(2, 2), bias=False)(fuse_pool_4)
        score_pool_3 = Convolution2D(21, 1, 1, init='he_normal')(max_pool_3)
        score_pool_3c = Cropping2D(cropping=((6, 7), (6, 7)))(score_pool_3)  # crop(n.score_pool3, n.upscore_pool4)
        fuse_pool_3 = merge([score_pool_3c, upscore_pool_4], mode='sum', concat_axis=1)
        upscore_8  = Deconvolution2D(21, 16, 16,
                                     output_shape=(None, 21, 568, 568),
                                     subsample=(8, 8), bias=False)(fuse_pool_3)
        crop = Cropping2D(cropping=((34, 34), (34, 34)))(upscore_8)  # crop(n.upscore8, n.data)
        reshape = Reshape((21, 500 * 500))(crop)
        trans =  Permute((2, 1))(reshape)
        out = Activation('softmax')(trans)
        model = Model(input=[img_input], output=[out])

    return model


model = fcn(mode=32)
model = load_vgg16_weights(model)
opti = Adam(lr=1e-3, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
for i in range(2):
    his = model.fit(Train_Data, Train_Targets,
                        batch_size=4,
                        nb_epoch=5,
                        verbose=1,
                        shuffle=True,
                        validation_data= (Val_Data, Val_Targets))

    preds = np.argmax(np.squeeze(np.reshape(model.predict(Val_Data[0:1, :, :, :]),
                                            (1, 500, 500, 21))), axis=2)
    gt = np.argmax(np.squeeze(np.reshape(Val_Targets[0:1, :, :],
                                         (1, 500, 500, 21))), axis=2)

    plt.figure(1), plt.imshow(preds)
    plt.figure(2), plt.imshow(gt)
    plt.pause(2)
