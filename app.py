import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. KONFIGURASI ---
# GANTI INI: Sesuaikan dengan path ke folder dataset Anda
# Struktur folder yang diharapkan:
# /path/ke/dataset/
#   |- class_A/
#   |   |- img1.jpg
#   |   |- img2.jpg
#   ...
#   |- class_B/
#   |   |- img100.jpg
#   ...
DATASET_DIR = "./dataset_padang_food"

IMG_SIZE = (224, 224)  # Ukuran input standar untuk DenseNet
BATCH_SIZE = 32
EPOCHS = 10  # Ganti sesuai kebutuhan

# --- 2. MEMUAT DATASET & MEMBUAT SPLIT 80/20 ---
# Menggunakan 80% data untuk training+validasi, 20% untuk testing
# validation_split=0.2 berarti 20% untuk 'validation' (testing), sisanya 80% untuk 'training'
try:
    train_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
except FileNotFoundError:
    print(f"Error: Dataset directory not found at {DATASET_DIR}")
    print("Silakan ganti placeholder 'DATASET_DIR' dengan path yang benar.")
    exit()

class_names = train_val_ds.class_names
num_classes = len(class_names)
print(f"Ditemukan {num_classes} kelas: {class_names}")

# --- 3. MEMBUAT VALIDATION SPLIT (20% DARI DATA TRAINING) ---
# Mengambil 20% dari train_val_ds untuk validasi
# Dapatkan jumlah total batch sebagai Tensor int64
cardinality = tf.data.experimental.cardinality(train_val_ds)

# Ubah (cast) tensor int64 menjadi float32
cardinality_float = tf.cast(cardinality, tf.float32)

# Sekarang lakukan perkalian float * float (ini aman)
val_size_float = cardinality_float * 0.2

# Terakhir, ubah hasilnya kembali ke integer untuk .take()
val_size = int(val_size_float)
val_ds = train_val_ds.take(val_size)
train_ds = train_val_ds.skip(val_size)

print(f"Total batches (80%): {tf.data.experimental.cardinality(train_val_ds)}")
print(f"Batches Training: {tf.data.experimental.cardinality(train_ds)}")
print(f"Batches Validasi: {tf.data.experimental.cardinality(val_ds)}")
print(f"Batches Testing (20%): {tf.data.experimental.cardinality(test_ds)}")

# --- 4. OPTIMASI DATA PIPELINE ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- 5. AUGMENTASI DATA ---
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        
        # PERBAIKAN: Gunakan height_factor dan width_factor 
        # dengan rentang (-0.2, 0.0) untuk HANYA "zoom out"
        layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0)),
        
        # PERBAIKAN: Ganti 'factor' dengan 'x_factor' dan 'y_factor'
        layers.RandomShear(x_factor=0.2, y_factor=0.2),
        
        # Ini sudah benar, RandomRotation MENGGUNAKAN 'factor'
        layers.RandomRotation(factor=0.2), 
    ],
    name="data_augmentation",
)

# --- 6. MEMBANGUN MODEL (DENSENET) ---
def build_densenet_model(num_classes):
    # Load pre-trained DenseNet121
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,  # Jangan include layer Fully Connected di atas
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Bekukan (freeze) layer dari base model
    base_model.trainable = False

    # Buat model baru
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Terapkan augmentasi
    x = data_augmentation(inputs)
    
    # Preprocessing input (penting untuk model pre-trained)
    x = tf.keras.applications.densenet.preprocess_input(x)
    
    # Lewatkan ke base model
    # training=False penting karena kita membekukan layer
    x = base_model(x, training=False)
    
    # Tambahkan head kustom kita
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout untuk regularisasi
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

model = build_densenet_model(num_classes)
model.summary()

# --- 7. KOMPILASI MODEL ---
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# --- 8. MELATIH MODEL ---
print("\n--- Memulai Training ---")
start_train_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

end_train_time = time.time()
training_time = end_train_time - start_train_time

# --- TAMBAHKAN BLOK KODE PLOTTING DI SINI ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Dapatkan jumlah epoch yang sebenarnya dijalankan
epochs_range = range(len(acc))

plt.figure(figsize=(14, 6)) # Atur ukuran gambar

# Plot untuk Akurasi
plt.subplot(1, 2, 1) # 1 baris, 2 kolom, plot ke-1
plt.plot(epochs_range, acc, label='Train')
plt.plot(epochs_range, val_acc, label='Validation')
plt.legend(loc='upper left')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot untuk Loss
plt.subplot(1, 2, 2) # 1 baris, 2 kolom, plot ke-2
plt.plot(epochs_range, loss, label='Train')
plt.plot(epochs_range, val_loss, label='Validation')
plt.legend(loc='upper left')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Tampilkan kedua plot
plt.show()
# ---------------------------------------------

print(f"--- Training Selesai ---")


# --- 9. EVALUASI MODEL ---
print("\n--- Memulai Testing (Evaluasi) ---")
y_pred_probs = []
y_test_labels = []

start_test_time = time.time()

# Menggunakan model.predict() untuk efisiensi
# Ini mengukur waktu inferensi (testing) murni
y_pred_probs = model.predict(test_ds)

end_test_time = time.time()
testing_time = end_test_time - start_test_time

# Kumpulkan label asli dari test_ds
for _, labels in test_ds.unbatch():
    y_test_labels.append(labels.numpy())

y_test_labels = np.array(y_test_labels)

# Konversi probabilitas (softmax) ke label kelas (argmax)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_test_labels_indices = np.argmax(y_test_labels, axis=1) # Konversi dari one-hot ke index

print(f"--- Testing Selesai ---")

# --- 10. MENGHITUNG METRIK ---
print("\n--- Hasil Evaluasi ---")

# a. Akurasi
accuracy = accuracy_score(y_test_labels_indices, y_pred_labels)

# b. Presisi, Recall, F1-Score (Gunakan 'weighted' untuk multi-class)
precision = precision_score(y_test_labels_indices, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_test_labels_indices, y_pred_labels, average='weighted', zero_division=0)
f1 = f1_score(y_test_labels_indices, y_pred_labels, average='weighted', zero_division=0)

# c. ROC/AUC (Gunakan 'ovr' - One-vs-Rest untuk multi-class)
# y_test_labels (sudah one-hot) vs y_pred_probs (probabilitas)
try:
    roc_auc = roc_auc_score(y_test_labels, y_pred_probs, multi_class='ovr', average='weighted')
except ValueError as e:
    roc_auc = f"Tidak dapat dihitung (mungkin hanya ada 1 sampel per kelas): {e}"

# d. Computation Time
# Sudah dihitung: training_time dan testing_time

# --- 11. MENAMPILKAN HASIL ---
print("\n\n========================================================")
print("           LAPORAN EVALUASI AKHIR MODEL")
print("========================================================")
print(f"Model                 : DenseNet121 (Transfer Learning)")
print(f"Total Epochs          : {EPOCHS}")
print("-" * 56)

print("\n📊 HASIL PERFORMA KESELURUHAN (Weighted Average)")
print(f"Akurasi               : {accuracy:.4f}")
print(f"Presisi (Weighted)    : {precision:.4f}")
print(f"Recall (Weighted)     : {recall:.4f}")
print(f"F1-Score (Weighted)   : {f1:.4f}")
print(f"ROC/AUC (Weighted)    : {roc_auc if isinstance(roc_auc, str) else f'{roc_auc:.4f}'}")
print("-" * 56)

print("\n⏱️ WAKTU KOMPUTASI")
print(f"Waktu Training        : {training_time:.2f} detik")
print(f"Waktu Testing         : {testing_time:.2f} detik")
print("=" * 56)


print("\n\n--- Classification Report (Rincian per Kelas) ---")
# Menghitung dan mencetak laporan klasifikasi
report = classification_report(y_test_labels_indices, 
                               y_pred_labels, 
                               target_names=class_names, 
                               zero_division=0)
print(report)
print("-" * 56)


print("\n--- Menampilkan Visualisasi Confusion Matrix ---")
cm = confusion_matrix(y_test_labels_indices, y_pred_labels)

# Membuat visualisasi heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cm, 
            annot=True,     # Menampilkan angka di dalam kotak
            fmt='d',        # Format angka sebagai integer
            cmap='Blues',   # Skema warna
            xticklabels=class_names, 
            yticklabels=class_names)

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show() # Menampilkan plot

print(f"\nLabel Kelas: {list(enumerate(class_names))}")