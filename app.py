import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- 1. KONFIGURASI & PARAMETER ---
# Sesuaikan path ini dengan lokasi folder dataset Anda
# Struktur folder yang diharapkan:
# dataset_padang_food/
#   ├── rendang/
#   │   ├── img1.jpg
#   │   └── ...
#   ├── sate_padang/
#   │   ├── img1.jpg
#   │   └── ...
#   └── ...
DATASET_PATH = "dataset_padang_food"

# Parameter Model
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 32
EPOCHS = 10  # Tambah jumlah epochs untuk hasil yang lebih baik
LEARNING_RATE = 0.001


def load_and_split_data(base_path):
    """
    Memuat path file, melakukan splitting data 80-20 (train_val-test)
    dan 80-20 (train-val) dari set train_val.
    Menggunakan stratify untuk menjaga proporsi kelas.
    """
    print(f"Membaca dataset dari: {base_path}")
    
    # Cari semua file gambar dengan ekstensi umum
    image_pattern = os.path.join(base_path, '*/*')
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    filepaths = []
    
    # Glob tidak case-insensitive, jadi kita cek manual
    for ext in allowed_extensions:
        filepaths.extend(glob.glob(image_pattern + ext))
        filepaths.extend(glob.glob(image_pattern + ext.upper()))

    if not filepaths:
        print(f"Error: Tidak ada file gambar ditemukan di {base_path}")
        print("Pastikan struktur folder Anda benar (cth: dataset_padang_food/nama_kelas/gambar.jpg)")
        return None, None, None, 0, []

    # Buat DataFrame untuk manajemen data yang mudah
    data = []
    for fp in filepaths:
        try:
            label = os.path.basename(os.path.dirname(fp))
            data.append({'filepath': fp, 'label': label})
        except Exception as e:
            print(f"Gagal memproses file {fp}: {e}")

    df = pd.DataFrame(data)

    if df.empty:
        print("Error: DataFrame kosong, tidak ada data yang berhasil diproses.")
        return None, None, None, 0, []

    print(f"\nTotal gambar ditemukan: {len(df)}")
    print("Distribusi kelas (sebelum split):")
    print(df['label'].value_counts())

    class_names = sorted(df['label'].unique())
    num_classes = len(class_names)

    if num_classes < 2:
        print(f"Error: Hanya ditemukan {num_classes} kelas. Dibutuhkan minimal 2 kelas untuk klasifikasi.")
        return None, None, None, 0, []

    # Split 1: 80% Training-Validasi, 20% Testing
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )

    # Split 2: 80% Training, 20% Validasi (dari set train_val_df)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.20,  # 0.20 dari 80% = 16% dari total
        random_state=42,
        stratify=train_val_df['label']
    )
    
    # 80% dari 80% = 64% dari total
    print(f"\nTotal data Training   : {len(train_df)} (64%)")
    print(f"Total data Validasi : {len(val_df)} (16%)")
    print(f"Total data Testing    : {len(test_df)} (20%)")
    
    return train_df, val_df, test_df, num_classes, class_names


def create_generators(train_df, val_df, test_df, class_names):
    """
    Membuat Keras ImageDataGenerators dari DataFrames.
    """
    print("\nMembuat Data Generators...")
    
    # Augmentasi untuk data training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,       # Zoom out
        shear_range=0.2,
        rotation_range=20     # Rotasi 0.2 radian ~ 11.4 derajat. 20 derajat lebih umum.
    )
    
    # TIDAK ADA augmentasi untuk validasi dan testing, hanya rescale
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    target_size = (IMG_HEIGHT, IMG_WIDTH)

    # Generator untuk Training
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names, # Pastikan urutan kelas konsisten
        shuffle=True
    )

    # Generator untuk Validasi
    validation_generator = test_val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )

    # Generator untuk Testing
    test_generator = test_val_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=class_names,
        shuffle=False  # PENTING: Jangan shuffle test data untuk evaluasi
    )
    
    return train_generator, validation_generator, test_generator


def build_model(num_classes):
    """
    Membangun model DenseNet121 untuk transfer learning.
    """
    print("\nMembangun model DenseNet121...")
    
    # Muat base model DenseNet121, pre-trained di ImageNet
    base_model = DenseNet121(
        weights='imagenet', 
        include_top=False,  # Buang layer klasifikasi asli
        input_shape=IMG_SHAPE
    )
    
    # Bekukan bobot base model agar tidak ikut terlatih
    base_model.trainable = False
    
    # Tambahkan layer kustom di atas base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Konversi fitur menjadi 1D vektor
    x = Dense(1024, activation='relu')(x) # Layer fully connected
    # Layer output (softmax untuk multi-kelas)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model berhasil dibangun.")
    return model


def plot_history(history):
    """
    Membuat plot akurasi dan loss training/validasi.
    """
    plt.figure(figsize=(12, 6))

    # Plot Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Akurasi Training vs Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Training vs Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_roc_auc(y_true, y_pred_proba, num_classes, class_labels):
    """
    Menghitung dan memplot Kurva ROC untuk setiap kelas (One-vs-Rest).
    """
    print("\nMembuat plot ROC/AUC...")
    
    # Binarize label asli
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Hitung ROC curve dan ROC area untuk setiap kelas
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color, label in zip(range(num_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve (area = {1:0.2f}) untuk kelas {0}'
                 ''.format(label, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Kurva ROC Multi-Kelas (One-vs-Rest)')
    plt.legend(loc="lower right", prop={'size': 10})
    plt.show()


def main():
    """
    Fungsi utama untuk menjalankan seluruh pipeline.
    """
    total_start_time = time.time()
    
    # 1. Muat dan Pisahkan Data
    train_df, val_df, test_df, num_classes, class_names = load_and_split_data(DATASET_PATH)
    
    if train_df is None:
        return  # Hentikan eksekusi jika data gagal dimuat

    # 2. Buat Generator
    train_generator, validation_generator, test_generator = create_generators(
        train_df, val_df, test_df, class_names
    )

    # 3. Bangun Model
    model = build_model(num_classes)
    model.summary()

    # 4. Latih Model
    print("\n--- Memulai Pelatihan Model ---")
    start_train_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        verbose=1
    )
    
    end_train_time = time.time()
    training_time = end_train_time - start_train_time
    print(f"--- Pelatihan Selesai (Durasi: {training_time:.2f} detik) ---")

    # 5. Evaluasi Model
    print("\n--- Memulai Evaluasi Model pada Data Test ---")
    start_test_time = time.time()

    # Dapatkan probabilitas prediksi
    # steps=None memastikan semua data test diprediksi
    y_pred_proba = model.predict(test_generator, steps=None, verbose=1) 
    
    # Konversi probabilitas ke label kelas
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Dapatkan label asli (ground truth)
    y_true = test_generator.classes
    
    end_test_time = time.time()
    testing_time = end_test_time - start_test_time
    print(f"--- Evaluasi Selesai (Durasi: {testing_time:.2f} detik) ---")

    # 6. Tampilkan Hasil Evaluasi
    print("\n=============================================")
    print("           HASIL EVALUASI MODEL          ")
    print("=============================================")

    # Hitung Metrik
    accuracy = accuracy_score(y_true, y_pred)
    # 'macro' menghitung metrik untuk setiap kelas, lalu mengambil rata-rata
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 'ovr' (One-vs-Rest) diperlukan untuk ROC/AUC multi-kelas
    try:
        y_true_bin_for_auc = label_binarize(y_true, classes=range(num_classes))
        roc_auc_macro = roc_auc_score(y_true_bin_for_auc, y_pred_proba, average='macro', multi_class='ovr')
    except ValueError as e:
        print(f"Peringatan: Gagal menghitung ROC AUC: {e}")
        roc_auc_macro = 0.0


    print("\n--- Metrik Keseluruhan (Macro-Average) ---")
    print(f"Akurasi    : {accuracy * 100:.2f} %")
    print(f"Presisi    : {precision * 100:.2f} %")
    print(f"Recall     : {recall * 100:.2f} %")
    print(f"F1 Score   : {f1 * 100:.2f} %")
    print(f"ROC/AUC    : {roc_auc_macro * 100:.2f} %")

    print("\n--- Laporan Klasifikasi (Per Kelas) ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # 7. Tampilkan Waktu Komputasi
    total_end_time = time.time()
    total_computation_time = total_end_time - total_start_time
    # total_computation_time_alt = training_time + testing_time 
    # (Hampir sama, tapi total_start_time mencakup pemrosesan data)

    print("\n--- Waktu Komputasi ---")
    print(f"Waktu Pelatihan (Training) : {training_time:.2f} detik")
    print(f"Waktu Pengujian (Testing)  : {testing_time:.2f} detik")
    print(f"Total Waktu (dari awal)    : {total_computation_time:.2f} detik")
    
    # 8. Tampilkan Plot
    print("\nMenampilkan plot...")
    plot_history(history)
    plot_roc_auc(y_true, y_pred_proba, num_classes, class_names)
    
    # Plot Confusion Matrix
    print("Menampilkan Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Label Asli (True)')
    plt.xlabel('Label Prediksi')
    plt.show()


# Menjalankan skrip utama
if __name__ == "__main__":
    # Pastikan TensorFlow menggunakan GPU jika tersedia
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Saat ini, set memory growth dibutuhkan untuk menghindari error CUDA
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs ditemukan.")
        except RuntimeError as e:
            print(e)
    else:
        print("Tidak ada GPU ditemukan, menggunakan CPU.")

    main()