import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd 
import shutil

def renombrar_imagenes(carpeta):
    # Obtener archivos de imagen (.jpg, .jpeg, .png) y ordenarlos
    archivos = sorted([f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Crear directorio temporal para evitar conflictos
    temp_dir = os.path.join(carpeta, "temp_rename")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Primero, mover todos los archivos a la carpeta temporal
    for archivo in archivos:
        shutil.move(os.path.join(carpeta, archivo), os.path.join(temp_dir, archivo))
    
    # Luego, mover de vuelta con los nuevos nombres
    for i, archivo in enumerate(archivos, start=1):
        extension = os.path.splitext(archivo)[1]  # Obtener la extensión
        nuevo_nombre = f"{i}{extension}"
        shutil.move(os.path.join(temp_dir, archivo), os.path.join(carpeta, nuevo_nombre))
    
    # Eliminar el directorio temporal
    os.rmdir(temp_dir)
    
    print(f"Se renombraron {len(archivos)} archivos correctamente.")
def load_datasettrain_with_subset(
    directory, 
    train_subset_percentage=1.0,  # Porcentaje de datos de entrenamiento a usar
    validation_split=0.2,  # Porcentaje para validación
    batch_size=16,
    image_size=(224, 224),
    seed=104,
    color_mode="rgb",
    label_mode="binary",
    class_names=["others", "radiography"]
):
    """
    Carga un dataset de imágenes con la opción de usar solo un subconjunto de los datos de entrenamiento.
    
    Parámetros:
    - directory: Ruta al directorio raíz del dataset
    - train_subset_percentage: Porcentaje de datos de entrenamiento a usar (0.0 - 1.0)
    - validation_split: Porcentaje de datos para validación
    - Otros parámetros similares a image_dataset_from_directory
    
    Retorna:
    - train_ds: Dataset de entrenamiento
    - val_ds: Dataset de validación
    """
    # Validar el porcentaje de subset
    if not 0 < train_subset_percentage <= 1.0:
        raise ValueError("train_subset_percentage debe estar entre 0 y 1")

    # Cargar dataset de entrenamiento con flujo de datos (sin cargar en memoria)
    full_train_ds = keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear",
        shuffle=True,
        verbose=False
    )

    # Cargar dataset de validación (también sin cargar en memoria)
    full_val_ds = keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation="bilinear",
        shuffle=True,
        verbose=False
    )

    # Contar total de muestras en el dataset de entrenamiento
    num_train_samples = sum(1 for _ in full_train_ds)
    subset_size = int(num_train_samples * train_subset_percentage)

    # Tomar solo el subconjunto necesario sin cargarlo en memoria
    train_ds = full_train_ds.take(subset_size)

    # Optimizar rendimiento con prefetching
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = full_val_ds.prefetch(tf.data.AUTOTUNE)

    # Imprimir información
    print(f"Total training batches: {num_train_samples}")
    print(f"Subset training batches: {subset_size}")
    print(f"Validation batches: {sum(1 for _ in full_val_ds)}")

    return train_ds, val_ds

def dataset_analysis(path):
    """
    Analyze image datasets in a given directory, providing comprehensive image format and characteristics statistics.
    
    Args:
        path (str): Root directory containing subfolders with image files
    """
    # Validate input path
    if not os.path.isdir(path):
        print(f"Error: {path} is not a valid directory.")
        return

    # Prepare overall dataset statistics
    total_dataset_stats = {
        'total_subfolders': 0,
        'total_images': 0,
        'image_formats': defaultdict(int),
        'bit_depths': defaultdict(int),
        'color_modes': defaultdict(int),
        'dimension_ranges': {
            'width': {'min': float('inf'), 'max': 0},
            'height': {'min': float('inf'), 'max': 0}
        }
    }

    # Iterate through subfolders
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    if not subfolders:
        print(f"No subfolders found in {path}")
        return

    total_dataset_stats['total_subfolders'] = len(subfolders)

    # Comprehensive analysis for each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        print(f"\n{'='*50}")
        print(f"Analyzing Subfolder: {subfolder}")
        print(f"{'='*50}")

        # Prepare subfolder-specific statistics
        subfolder_stats = {
            'total_images': 0,
            'image_formats': defaultdict(int),
            'bit_depths': defaultdict(int),
            'color_modes': defaultdict(int),
            'dimension_ranges': {
                'width': {'min': float('inf'), 'max': 0},
                'height': {'min': float('inf'), 'max': 0}
            }
        }

        # Get all files in the subfolder
        files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        
        if not files:
            print(f"No files found in subfolder {subfolder}")
            continue

        # Process each image file
        for file in files:
            try:
                file_path = os.path.join(subfolder_path, file)
                
                # Skip non-image files
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    continue

                with Image.open(file_path) as img:
                    # Image characteristics
                    image_type = img.format.upper() if img.format else "UNKNOWN"
                    width, height = img.size
                    image_mode = img.mode

                    # Determine bit depth
                    bit_depth_map = {
                        "1": 1,     # Black and white
                        "L": 8,     # Grayscale
                        "P": 8,     # Palette-based
                        "RGB": 24,  # True color
                        "RGBA": 32, # True color with alpha
                        "CMYK": 32  # Color separation
                    }
                    bit_depth = bit_depth_map.get(image_mode, "Unknown")

                    # Update subfolder statistics
                    subfolder_stats['total_images'] += 1
                    subfolder_stats['image_formats'][image_type] += 1
                    subfolder_stats['bit_depths'][bit_depth] += 1
                    subfolder_stats['color_modes'][image_mode] += 1
                    
                    # Update dimension ranges
                    subfolder_stats['dimension_ranges']['width']['min'] = min(subfolder_stats['dimension_ranges']['width']['min'], width)
                    subfolder_stats['dimension_ranges']['width']['max'] = max(subfolder_stats['dimension_ranges']['width']['max'], width)
                    subfolder_stats['dimension_ranges']['height']['min'] = min(subfolder_stats['dimension_ranges']['height']['min'], height)
                    subfolder_stats['dimension_ranges']['height']['max'] = max(subfolder_stats['dimension_ranges']['height']['max'], height)

                    # Update total dataset statistics
                    total_dataset_stats['total_images'] += 1
                    total_dataset_stats['image_formats'][image_type] += 1
                    total_dataset_stats['bit_depths'][bit_depth] += 1
                    total_dataset_stats['color_modes'][image_mode] += 1
                    total_dataset_stats['dimension_ranges']['width']['min'] = min(total_dataset_stats['dimension_ranges']['width']['min'], width)
                    total_dataset_stats['dimension_ranges']['width']['max'] = max(total_dataset_stats['dimension_ranges']['width']['max'], width)
                    total_dataset_stats['dimension_ranges']['height']['min'] = min(total_dataset_stats['dimension_ranges']['height']['min'], height)
                    total_dataset_stats['dimension_ranges']['height']['max'] = max(total_dataset_stats['dimension_ranges']['height']['max'], height)

            except Exception as e:
                print(f"Error processing '{file}' in '{subfolder}': {e}")

        # Print subfolder statistics
        print(f"\nSubfolder Summary - {subfolder}:")
        print(f"Total Images: {subfolder_stats['total_images']}")
        
        print("\nImage Formats:")
        for fmt, count in subfolder_stats['image_formats'].items():
            print(f"  - {fmt}: {count} images ({count/subfolder_stats['total_images']*100:.2f}%)")
        
        print("\nColor Modes:")
        for mode, count in subfolder_stats['color_modes'].items():
            print(f"  - {mode}: {count} images ({count/subfolder_stats['total_images']*100:.2f}%)")
        
        print("\nBit Depths:")
        for depth, count in subfolder_stats['bit_depths'].items():
            print(f"  - {depth} bit: {count} images ({count/subfolder_stats['total_images']*100:.2f}%)")
        
        print("\nDimension Ranges:")
        print(f"  - Width:  {subfolder_stats['dimension_ranges']['width']['min']} - {subfolder_stats['dimension_ranges']['width']['max']} pixels")
        print(f"  - Height: {subfolder_stats['dimension_ranges']['height']['min']} - {subfolder_stats['dimension_ranges']['height']['max']} pixels")

    # Print total dataset statistics
    print("\n\n" + "="*50)
    print("TOTAL DATASET OVERVIEW")
    print("="*50)
    print(f"Total Subfolders: {total_dataset_stats['total_subfolders']}")
    print(f"Total Images: {total_dataset_stats['total_images']}")
    
    print("\nOverall Image Formats:")
    for fmt, count in total_dataset_stats['image_formats'].items():
        print(f"  - {fmt}: {count} images ({count/total_dataset_stats['total_images']*100:.2f}%)")
    
    print("\nOverall Color Modes:")
    for mode, count in total_dataset_stats['color_modes'].items():
        print(f"  - {mode}: {count} images ({count/total_dataset_stats['total_images']*100:.2f}%)")
    
    print("\nOverall Bit Depths:")
    for depth, count in total_dataset_stats['bit_depths'].items():
        print(f"  - {depth} bit: {count} images ({count/total_dataset_stats['total_images']*100:.2f}%)")
    
    print("\nOverall Dimension Ranges:")
    print(f"  - Width:  {total_dataset_stats['dimension_ranges']['width']['min']} - {total_dataset_stats['dimension_ranges']['width']['max']} pixels")
    print(f"  - Height: {total_dataset_stats['dimension_ranges']['height']['min']} - {total_dataset_stats['dimension_ranges']['height']['max']} pixels")


def plot_image_counts(dataset_directory):
    """
    Grafica el número de imágenes en cada clase (subdirectorios) en los directorios 'train' y 'test' en un solo gráfico.
    
    Args:
    dataset_directory (str): Ruta principal donde están las carpetas 'train' y 'test'.
    """
    # Directorios de entrenamiento y prueba
    train_dir = os.path.join(dataset_directory, 'train')
    test_dir = os.path.join(dataset_directory, 'test')

    # Diccionario para almacenar el conteo de imágenes por clase
    class_counts = {'train': {}, 'test': {}}

    # Extensiones válidas de imágenes
    valid_extensions = ['.png', '.jpg','.JPEG']

    # Contamos imágenes en el directorio 'train'
    for subdir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, subdir)
        if os.path.isdir(class_path):
            class_counts['train'][subdir] = len([f for f in os.listdir(class_path) if any(f.endswith(ext) for ext in valid_extensions)])

    # Contamos imágenes en el directorio 'test'
    for subdir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, subdir)
        if os.path.isdir(class_path):
            class_counts['test'][subdir] = len([f for f in os.listdir(class_path) if any(f.endswith(ext) for ext in valid_extensions)])

    # Combinamos las clases de train y test
    all_classes = sorted(set(class_counts['train'].keys()).union(class_counts['test'].keys()))
    train_counts = [class_counts['train'].get(c, 0) for c in all_classes]
    test_counts = [class_counts['test'].get(c, 0) for c in all_classes]

    # Ancho de las barras para las clases de train y test
    width = 0.35  # Ancho de las barras
    x = np.arange(len(all_classes))  # Posiciones de las clases

    # Graficamos las barras agrupadas
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='blue')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test', color='red')

    # Añadimos etiquetas y título
    ax.set_title('Número de imágenes por clase en Train y Test')
    ax.set_xlabel('Clases')
    ax.set_ylabel('Número de imágenes')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()

    # Mejoramos el diseño
    plt.tight_layout()
    plt.show()


def show_random_images(dataset, num_images=20, images_per_row=5):
    """
    Muestra imágenes aleatorias de un dataset de TensorFlow, seleccionando de múltiples batches si es necesario.
    
    Parámetros:
    - dataset: Dataset de TensorFlow con imágenes y etiquetas.
    - num_images: Número total de imágenes a mostrar.
    - images_per_row: Número de imágenes por fila.
    """
    all_images = []
    all_labels = []
    
    # Recopilamos imágenes y etiquetas de los batches
    for images, labels in dataset.take((num_images // images_per_row) + 1):  # Tomamos suficientes batches
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
        
        # Si hemos recopilado suficientes imágenes, paramos
        if sum(len(imgs) for imgs in all_images) >= num_images:
            break
    
    # Concatenamos todas las imágenes y etiquetas en arrays
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    
    # Asegurarnos de que no excedamos el número solicitado de imágenes
    num_images = min(num_images, len(all_images))
    
    # Seleccionamos índices aleatorios sin repetición
    random_indices = np.random.choice(len(all_images), num_images, replace=False)
    
    # Definir el número de filas necesarias para mostrar las imágenes
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Calcular número de filas
    
    # Crear el gráfico
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 4, num_rows * 4))
    axes = axes.flatten()  # Convertir a una lista unidimensional para acceso más fácil
    
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(all_images[random_indices[i]].astype("uint8"))  # Mostrar la imagen seleccionada
            ax.set_title(f"Label: {int(all_labels[random_indices[i]])}")  # Mostrar la etiqueta
            ax.axis('off')  # Desactivar los ejes
        else:
            ax.axis('off')  # Desactivar los ejes para las celdas vacías
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataset, num_images=20, images_per_row=5, corte=0.5):
    all_images = []
    all_labels = []
    
    # Create a list from the dataset iterator
    dataset_list = list(dataset)
    
    # Determine how many batches to use based on batch size
    batch_count = min(len(dataset_list), num_images)
    
    # Get random indices for batches
    random_indices = np.random.choice(len(dataset_list), batch_count, replace=False)
    
    # Extract images and labels
    for idx in random_indices:
        batch_images, batch_labels = dataset_list[idx]
        
        # Handle either single image or batch
        if len(batch_images.shape) == 4:  # Batch of images
            for i in range(batch_images.shape[0]):
                all_images.append(batch_images[i].numpy())
                all_labels.append(batch_labels[i].numpy())
        else:  # Single image
            all_images.append(batch_images.numpy())
            all_labels.append(batch_labels.numpy())
        
        # Stop if we have enough images
        if len(all_images) >= num_images:
            all_images = all_images[:num_images]
            all_labels = all_labels[:num_images]
            break
    
    # Convert lists to arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Make predictions
    predictions = model.predict(all_images)
    predicted_classes = (predictions > corte).astype("int")
    
    # Ensure predictions are flattened properly
    if len(predicted_classes.shape) > 1:
        predicted_classes = predicted_classes.flatten()
    
    # Ensure labels are in the right format
    if len(all_labels.shape) > 1 and all_labels.shape[1] == 1:
        all_labels = all_labels.flatten()
    
    # Calculate rows needed
    num_rows = (len(all_images) + images_per_row - 1) // images_per_row
    
    # Create the figure
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))
    axes = axes.flatten() if num_rows * images_per_row > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < len(all_images):
            ax.imshow(all_images[i].astype("uint8"))
            
            # Get prediction and label
            pred = predicted_classes[i] if len(predicted_classes.shape) == 1 else predicted_classes[i][0]
            label = int(all_labels[i])
            pred_value = predictions[i] if len(predictions.shape) == 1 else predictions[i][0]
            
            # Set title color
            color = "green" if pred == label else "red"
            
            # Set title
            ax.set_title(f"Pred: {pred} / Real: {label}\n{pred_value:.4f}", color=color)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def make_model(input_shape, num_layers=2, num_neurons=64):
    model = keras.Sequential()
    
    # Normalización de entrada
    model.add(layers.Rescaling(1./255, input_shape=input_shape)) 
    model.add(layers.Flatten())
    
    # Capas ocultas con regularización
    for i in range(num_layers):
        model.add(layers.Dense(num_neurons, kernel_initializer='he_normal'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
    
    # Capa de salida
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Optimizador con tasa de aprendizaje explícita
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']  
    )
    model.summary()
    return model

def make_cnn_model(input_shape, num_classes=1):
    model = keras.Sequential()

    # Primera capa convolucional
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.2))


    # Capa densa
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Capa de salida (sigmoide para clasificación binaria)
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    # Compilación del modelo
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()]
    )

    model.summary()
    return model

def plot_training_history(history):
    """
    Genera gráficas de pérdida y precisión a partir del historial de entrenamiento con un estilo mejorado.
    
    :param history: Objeto History devuelto por model.fit().
    """
    sns.set_style("whitegrid")
    metrics = history.history.keys()
    
    plt.figure(figsize=(12, 5))
    
    # Gráfica de pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o', linestyle='dashed', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', linestyle='solid', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Gráfica de precisión (Accuracy), si está disponible
    if 'accuracy' in metrics:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linestyle='dashed', color='green')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linestyle='solid', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        
        # Ajustar el eje Y dinámicamente con un pequeño margen
        min_acc = min(min(history.history['accuracy']), min(history.history['val_accuracy']))
        max_acc = max(max(history.history['accuracy']), max(history.history['val_accuracy']))
        plt.ylim(min_acc - 0.1, max_acc + 0.1)
        
        plt.legend()
        plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataset, corte=0.5, dataset_name="dataset"):
    """
    Evalúa un modelo entrenado en un conjunto de datos (puede ser train, val o test)
    y genera métricas clave.
    
    :param model: Modelo de Keras entrenado.
    :param dataset: Conjunto de datos de evaluación (tf.data.Dataset).
    :param corte: Umbral para clasificación binaria.
    :param dataset_name: Nombre del conjunto de datos (train, val, test) para mostrar en los resultados.
    """
    # Obtener etiquetas y predicciones del conjunto de datos
    y_true = []
    y_pred = []
    
    for x, y in dataset:
        y_true.append(y.numpy())  # Etiquetas reales
        y_pred.append(model.predict(x,verbose=0))  # Predicciones del modelo
    
    y_true = np.concatenate(y_true, axis=0)  # Concatenar las etiquetas reales
    y_pred = np.concatenate(y_pred, axis=0)  # Concatenar las predicciones
    
    # Si es binario, aplicamos el umbral
    if y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()  # Asegurarse de que sea 1D
        y_pred_labels = (y_pred > corte).astype(int)
    else:
        y_pred_labels = np.argmax(y_pred, axis=1)  # Si es multiclase, tomamos el argmax
    
    y_true = y_true.flatten()  # Aplanar las etiquetas verdaderas
    
    # Imprimir reporte de clasificación
    report = classification_report(y_true, y_pred_labels, target_names=["others", "radiography"], output_dict=True)
    print(f"Reporte de clasificación para {dataset_name}:")
    print(pd.DataFrame(report).transpose())  # Formatear el reporte como tabla
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["others", "radiography"], 
                yticklabels=["others", "radiography"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.show()
    
    # Curva ROC y AUC (solo para clasificación binaria)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC for {dataset_name}")
    plt.legend()
    plt.show()
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['radiography']['precision'],
        'recall': report['radiography']['recall'],
        'f1': report['radiography']['f1-score'],
        'auc': roc_auc
    }

def visualize_incorrect_predictions(model, dataset, num_images=20, images_per_row=5,corte=0.5):
    """
    Visualiza las predicciones incorrectas de un modelo de clasificación binaria.
    
    Args:
        model: Modelo de TensorFlow/Keras entrenado
        dataset: Dataset de TensorFlow con imágenes y etiquetas
        num_images: Número máximo de imágenes incorrectas a mostrar
        images_per_row: Número de imágenes por fila en la visualización
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    incorrect_images = []
    incorrect_labels = []
    incorrect_predictions = []
    incorrect_probabilities = []
    
    num_images_collected = 0
    
    # Recorremos el dataset hasta encontrar suficientes predicciones incorrectas
    for images, labels in dataset:
        # Verificamos que las imágenes y etiquetas sean tensores y convertimos si es necesario
        if hasattr(images, 'numpy'):
            images_np = images.numpy()
        else:
            images_np = np.array(images)
            
        if hasattr(labels, 'numpy'):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        
        # Hacemos predicciones para el lote actual
        predictions = model.predict(images, verbose=0)
        
        # Aseguramos que las predicciones tengan la forma correcta
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            # Para modelos con una neurona de salida (clasificación binaria)
            predicted_classes = (predictions > corte).astype("int").flatten()
            probabilities = predictions.flatten()
        else:
            # Para modelos con dos neuronas de salida (softmax)
            predicted_classes = np.argmax(predictions, axis=1)
            probabilities = np.max(predictions, axis=1)
        
        # Aseguramos que las etiquetas tengan la forma correcta
        if len(labels_np.shape) > 1:
            labels_flat = labels_np.flatten()
        else:
            labels_flat = labels_np
            
        # Filtramos las predicciones incorrectas
        incorrect_indices = np.where(predicted_classes != labels_flat)[0]
        
        if len(incorrect_indices) > 0:
            # Usamos los índices incorrectos para extraer las imágenes y etiquetas
            incorrect_images_batch = images_np[incorrect_indices]
            incorrect_labels_batch = labels_flat[incorrect_indices]
            incorrect_predictions_batch = predicted_classes[incorrect_indices]
            incorrect_probabilities_batch = probabilities[incorrect_indices]
            
            # Agregamos las imágenes y etiquetas incorrectas al acumulador
            incorrect_images.append(incorrect_images_batch)
            incorrect_labels.append(incorrect_labels_batch)
            incorrect_predictions.append(incorrect_predictions_batch)
            incorrect_probabilities.append(incorrect_probabilities_batch)
            
            # Incrementamos el contador de imágenes recolectadas
            num_images_collected += len(incorrect_indices)
            
        # Detenerse si hemos alcanzado el número deseado de imágenes incorrectas
        if num_images_collected >= num_images:
            break
    
    # Convertimos las listas a arrays de numpy
    if incorrect_images:  # Solo hacemos la concatenación si hay imágenes incorrectas
        incorrect_images = np.concatenate(incorrect_images)
        incorrect_labels = np.concatenate(incorrect_labels)
        incorrect_predictions = np.concatenate(incorrect_predictions)
        incorrect_probabilities = np.concatenate(incorrect_probabilities)
    
    # Si no hay predicciones incorrectas, terminamos la función
    if len(incorrect_images) == 0:
        print("No se encontraron predicciones incorrectas.")
        return
    
    # Seleccionamos el número deseado de imágenes incorrectas
    num_incorrect_images = min(len(incorrect_images), num_images)
    
    # Calculamos el número de filas necesarias para la visualización
    num_rows = (num_incorrect_images + images_per_row - 1) // images_per_row
    
    # Creamos la figura para la visualización
    plt.figure(figsize=(images_per_row * 3, num_rows * 3))
    
    for i in range(num_incorrect_images):
        plt.subplot(num_rows, images_per_row, i + 1)
        
        # Nos aseguramos de que la imagen tenga el formato correcto para mostrar
        img = incorrect_images[i]
        
        # Normalización de la imagen si es necesario
        if img.max() > 1.0:
            img = img.astype("uint8")
        
        # Ajustamos el formato según la forma de la imagen
        if len(img.shape) == 2:  # Imagen en escala de grises
            plt.imshow(img, cmap='gray')
        elif img.shape[-1] == 1:  # Canal único pero con dimensión adicional
            plt.imshow(img.squeeze(), cmap='gray')
        else:  # Imagen a color (RGB)
            plt.imshow(img)
        
        # Definimos el título con información relevante
        title = f"Real: {int(incorrect_labels[i])}, Pred: {int(incorrect_predictions[i])}\n"
        title += f"Prob: {incorrect_probabilities[i]:.4f}"
        
        plt.title(title, color="red")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return incorrect_images, incorrect_labels, incorrect_predictions, incorrect_probabilities

def youden_index(model, test_ds):
    """
    Calcula el índice de Youden para un modelo dado y un conjunto de datos de prueba.
    
    :param model: Modelo entrenado de Keras.
    :param test_ds: Conjunto de datos de prueba (tf.data.Dataset).
    :return: Índice de Youden.
    """
    
    # Extraer etiquetas y predicciones
    y_true = []
    y_pred = []
    
    # Iterar sobre el dataset
    for x, y in test_ds:
        y_true.append(y.numpy())  # Convertir las etiquetas a numpy
        y_pred.append(model.predict(x,verbose=0))  # Obtener las predicciones del modelo
    
    # Concatenar las listas de etiquetas y predicciones
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Asegurarse de que y_pred esté en formato 1D
    y_pred = y_pred.flatten()
    
    # Convertir las probabilidades a etiquetas binarias (0 o 1) usando el umbral de 0.5
    y_pred_labels = (y_pred > 0.5).astype(int)
    
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    
    # Calcular sensibilidad y especificidad
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Índice de Youden
    J = sensitivity + specificity - 1
    
    return J


def show_class_distribution(dataset):
    """
    Muestra la distribución de clases en un dataset de TensorFlow.
    
    Parámetro:
    - dataset: Dataset de TensorFlow con imágenes y etiquetas.
    """
    class_counts = {0: 0, 1: 0}  # Suponiendo que tienes 2 clases: 0 para "others", 1 para "radiography"
    
    for _, labels in dataset:
        for label in labels.numpy():
            class_counts[int(label)] += 1  # Contamos las clases
    
    total_samples = sum(class_counts.values())
    
    print(f"Distribución de clases:")
    print(f"Clase 0 (others): {class_counts[0]} ({class_counts[0] / total_samples * 100:.2f}%)")
    print(f"Clase 1 (radiography): {class_counts[1]} ({class_counts[1] / total_samples * 100:.2f}%)")

def plot_training_metrics(train_metrics, val_metrics, title_prefix=''):
    """
    Plot training and validation metrics.
    
    Args:
    - train_metrics (dict): Dictionary with 'loss' and 'accuracy' lists for training
    - val_metrics (dict): Dictionary with 'loss' and 'accuracy' lists for validation
    - title_prefix (str, optional): Prefix for plot titles
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(train_metrics['loss'], label='Train Loss', color='blue')
    ax1.plot(val_metrics['loss'], label='Validation Loss', color='red')
    ax1.set_title(f'{title_prefix}Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(train_metrics['accuracy'], label='Train Accuracy', color='blue')
    ax2.plot(val_metrics['accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title(f'{title_prefix}Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    