import librosa
import numpy as np
import tensorflow as tf
from openai.helpers.local_audio_player import SAMPLE_RATE


# 1. 数据预处理
def load_data(data_path):
    # 加载音频文件列表和对应语音转录
    audio_files, transcripts = load_metadata(data_path)

    # 提取MFCC特征
    mfcc_features = []
    for audio_file in audio_files:
        audio, rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(audio, sr=rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_features.append(mfcc.T)

    # 标记独热编码
    transcript_targets = np.array([to_categorical([char_to_index[c] for c in text.lower()], num_classes=NUM_CLASSES) for text in transcripts])

    return mfcc_features, transcript_targets

# 2. 模型构建
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. 模型训练
def train_model(x_train, y_train, x_test, y_test):
    model = build_model(x_train[0].shape)
    train_iterator = create_data_iterator(x_train, y_train, batch_size=BATCH_SIZE)
    validation_iterator = create_data_iterator(x_test, y_test, batch_size=BATCH_SIZE)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= MODEL_CHECKPOINT_DIR,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

    history = model.fit(train_iterator, epochs=NUM_EPOCHS, validation_data=validation_iterator,
                        callbacks=[model_checkpoint_callback, early_stop_callback])

    return model, history

# 4. 模型测试和声音克隆
def clone_sound(model, input_path):
    input_mfcc = extract_mfcc(input_path)
    predicted_transcript = predict_text(model, input_mfcc)
    synthesized_audio = synthesize_audio(predicted_transcript)
    save_audio(synthesized_audio)
