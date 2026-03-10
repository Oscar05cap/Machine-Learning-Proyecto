from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

base_path = "/home/oscar/Documents/AM/Proyecto/Gatos (supervisado)/cat_v1"
data_root = Path(base_path)
img_size = (64, 64)          # dimensiones de la muestra 
use_grayscale = False         

images_info = []
X_vectors = []
y_labels = []

for breed_dir in data_root.iterdir():
    if not breed_dir.is_dir():
        continue
    breed = breed_dir.name
    for img_path in breed_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        stats = img_path.stat()
        try:
            with Image.open(img_path) as img:
                # Reescala y convierte
                if use_grayscale:
                    img_resized = img.resize(img_size).convert('L')
                else:
                    img_resized = img.resize(img_size).convert('RGB')
                img_array = np.array(img_resized).flatten()
                
                # Estadísticas de los píxeles
                mean_pixel = np.mean(img_array)
                std_pixel = np.std(img_array)
                
                # Metadatos que contiene la imagen
                width, height = img.size
                mode = img.mode
        except Exception as e:
            print(f"Error with {img_path}: {e}")
            continue

        images_info.append({
            'breed': breed,
            'file_name': img_path.name,
            'size_bytes': stats.st_size,
            'width': width,
            'height': height,
            'mode': mode,
            'extension': img_path.suffix,
            'mean_pixel': mean_pixel,
            'std_pixel': std_pixel
        })
        X_vectors.append(img_array)
        y_labels.append(breed)

# Para converitr en arrays
X_matrix = np.array(X_vectors)
y_labels = np.array(y_labels)

# Creación del dataframe
df_images = pd.DataFrame(images_info)
print(f"Total images: {len(df_images)}")
print(df_images.head())
print(df_images.info())
print(df_images.describe())
print(df_images['breed'].value_counts())

# Bar plot
plt.figure(figsize=(10,6))
df_images['breed'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel("Breed")
plt.ylabel("Number of images")
plt.title("Images per breed")
plt.show()

# Pie chart
counts = df_images['breed'].value_counts()
plt.figure(figsize=(10,6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
        startangle=140, colors=plt.cm.Paired.colors)
plt.title("Images per breed")
plt.show()

# COmprobar si exisen imágenes no RGB
non_rgb = df_images[df_images['mode'] != 'RGB']
if not non_rgb.empty:
    print("Non‑RGB images found:")
    print(non_rgb[['file_name', 'mode']])
else:
    print("All images are RGB.")

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_matrix)

le = LabelEncoder()
y_numeric = le.fit_transform(y_labels)

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_numeric,
                      cmap='tab10', alpha=0.6)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.title("PCA projection of images")
handles, _ = scatter.legend_elements(prop="colors")
plt.legend(handles, le.classes_, title="Breed")
plt.show()

print(f"Explained variance PC1: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Explained variance PC2: {pca.explained_variance_ratio_[1]:.2f}")

# Correlación entre características
corr_cols = ['width', 'height', 'size_bytes', 'mean_pixel', 'std_pixel']
corr_matrix = df_images[corr_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation between image features")
plt.show()

# Distribución de width's
plt.figure(figsize=(10,6))
sns.histplot(data=df_images, x='width', hue='breed', kde=True, bins=30)
plt.title("Distribution of image widths by breed")
plt.show()

# Brillo promedio que existe en las imágenes (por raza)
plt.figure(figsize=(10,6))
sns.boxplot(data=df_images, x='breed', y='mean_pixel')
plt.title("Average brightness by breed")
plt.xticks(rotation=45)
plt.show()

# Imprime una muestra por cada imagen
breeds = df_images['breed'].unique()
fig, axes = plt.subplots(2, 3, figsize=(10,7))
axes = axes.flatten()

for i, breed in enumerate(breeds):
    # Toma la primera imagen de cada raza
    img_row = df_images[df_images['breed'] == breed].iloc[0]
    img_path = data_root / breed / img_row['file_name']
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(breed)
    axes[i].axis('off')
    
plt.tight_layout()
plt.show()