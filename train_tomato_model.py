import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np

# === Paths ===
train_dir = r'C:\Users\user\kat\Final_Project\Tomato_Leaf_Disease\tomato\train'
val_dir = r'C:\Users\user\kat\Final_Project\Tomato_Leaf_Disease\tomato\val'

# === Image settings ===
img_size = (224, 224)
batch_size = 16

# === Data generators ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === Load MobileNetV2 ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train the model ===
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# === Save model ===
model.save('models/tomato_model.h5')
print("\n✅ Model saved as 'models/tomato_model.h5'")

# === Plot training results ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# === Evaluation ===
print("\n📊 Evaluating on validation set:")
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Accuracy: {accuracy * 100:.2f}%")

precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
print("\n🔍 Per-Class Performance:")
for idx, label in enumerate(class_labels):
    print(f"{label:<15} | Precision: {precision[idx]:.2f} | Recall: {recall[idx]:.2f} | F1-Score: {f1[idx]:.2f} | Support: {support[idx]}")
