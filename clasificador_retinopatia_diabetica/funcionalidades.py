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
import collections
from tensorflow.keras import layers, models, regularizers


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

def show_class_distribution(dataset, class_names=None):
    """
    Muestra la distribución de clases en un dataset de TensorFlow.

    Parámetros:
    - dataset: Dataset de TensorFlow con imágenes y etiquetas.
    - class_names (opcional): Lista de nombres de clases. Si no se proporciona, usará los índices.

    """

    class_counts = collections.Counter()

    for _, labels in dataset:
        for label in labels.numpy():
            class_counts[int(label)] += 1

    total_samples = sum(class_counts.values())

    print("Distribución de clases:")
    for class_idx, count in sorted(class_counts.items()):
        class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f"Clase {class_idx}"
        percentage = (count / total_samples) * 100
        print(f"{class_name}: {count} ({percentage:.2f}%)")


    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.30),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.35),

        # Flatten + Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def evaluate_model(model, dataset,class_names, dataset_name="test"):
    """
    Evalúa un modelo entrenado en un conjunto de datos multiclase (5 clases)
    y genera métricas clave.
    
    :param model: Modelo de Keras entrenado.
    :param dataset: Conjunto de datos de evaluación (tf.data.Dataset).
    :param dataset_name: Nombre del conjunto de datos para mostrar en los resultados.
    :return: Diccionario con las métricas principales.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    # Definir las clases
    n_classes = len(class_names)
    
    # Obtener etiquetas y predicciones del conjunto de datos
    y_true = []
    y_pred_probs = []
    
    for x, y in dataset:
        y_true.append(y.numpy())  # Etiquetas reales
        y_pred_probs.append(model.predict(x, verbose=0))  # Predicciones del modelo (probabilidades)
    
    y_true = np.concatenate(y_true, axis=0)  # Concatenar las etiquetas reales
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)  # Concatenar las predicciones
    
    # Para clasificación multiclase, tomamos el argmax de las probabilidades
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    # Verificar si las etiquetas son one-hot encoded o índices directos
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)  # Convertir de one-hot a índices
    else:
        y_true_labels = y_true.flatten()  # Usar directamente los índices
    
    # Imprimir reporte de clasificación
    report = classification_report(y_true_labels, y_pred_labels, 
                                  target_names=class_names, 
                                  output_dict=True)
    print(f"Reporte de clasificación para {dataset_name}:")
    print(pd.DataFrame(report).transpose())  # Formatear el reporte como tabla
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel("Etiquetas Predichas")
    plt.ylabel("Etiquetas Reales")
    plt.title(f"Matriz de Confusión para {dataset_name}")
    plt.tight_layout()
    plt.show()
    
    # Curvas ROC para cada clase (one-vs-rest)
    # Primero binarizamos las etiquetas para el enfoque one-vs-rest
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_bin = y_true  # Ya está en formato one-hot
    else:
        y_true_bin = label_binarize(y_true_labels, classes=range(n_classes))
    
    # Calcular curvas ROC y AUC para cada clase
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
    
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {class_name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curvas ROC para clasificación multiclase ({dataset_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calcular métricas generales y específicas por clase
    metrics = {
        'accuracy': report['accuracy'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }
    
    # Añadir métricas específicas por clase
    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = report[class_name]['precision']
        metrics[f'{class_name}_recall'] = report[class_name]['recall']
        metrics[f'{class_name}_f1'] = report[class_name]['f1-score']
        metrics[f'{class_name}_auc'] = roc_auc[i]
    
    return metrics

def visualize_predictions(model, dataset, num_images=20, images_per_row=5, class_names=None):
    """
    Visualiza las predicciones de un modelo en imágenes seleccionadas aleatoriamente de un dataset.
    
    :param model: Modelo de Keras entrenado.
    :param dataset: Dataset de Keras (tf.data.Dataset).
    :param num_images: Número de imágenes a visualizar.
    :param images_per_row: Número de imágenes por fila en la visualización.
    :param class_names: Lista con los nombres de las clases. Si es None, se usan índices numéricos.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Establecer nombres de clases predeterminados si no se proporcionan
    if class_names is None:
        class_names = [f"Clase {i}" for i in range(5)]
    
    # Lista para almacenar imágenes y etiquetas
    all_images = []
    all_labels = []
    
    # Convertir el dataset en una lista
    dataset_list = list(dataset)
    
    # Determinar cuántos lotes usar basado en el tamaño del lote
    batch_count = min(len(dataset_list), num_images)
    
    # Obtener índices aleatorios para los lotes
    random_indices = np.random.choice(len(dataset_list), batch_count, replace=False)
    
    # Extraer imágenes y etiquetas
    for idx in random_indices:
        batch_images, batch_labels = dataset_list[idx]
        
        # Manejar tanto imágenes individuales como lotes
        if len(batch_images.shape) == 4:  # Lote de imágenes
            for i in range(batch_images.shape[0]):
                all_images.append(batch_images[i].numpy())
                all_labels.append(batch_labels[i].numpy())
        else:  # Imagen individual
            all_images.append(batch_images.numpy())
            all_labels.append(batch_labels.numpy())
        
        # Detener si tenemos suficientes imágenes
        if len(all_images) >= num_images:
            all_images = all_images[:num_images]
            all_labels = all_labels[:num_images]
            break
    
    if len(all_images) == 0:
        print("No se pudieron extraer imágenes del dataset. Verifica el formato del dataset.")
        return
    
    # Convertir listas a arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    # Hacer predicciones
    predictions_prob = model.predict(all_images)
    
    # Obtener la clase predicha (índice con mayor probabilidad)
    predicted_classes = np.argmax(predictions_prob, axis=1)
    
    # Convertir etiquetas one-hot a índices si es necesario
    if len(all_labels.shape) > 1 and all_labels.shape[1] > 1:
        true_classes = np.argmax(all_labels, axis=1)
    else:
        true_classes = all_labels.flatten().astype(int)
    
    # Calcular filas necesarias
    num_rows = (len(all_images) + images_per_row - 1) // images_per_row
    
    # Crear la figura
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 3, num_rows * 3))
    axes = axes.flatten() if num_rows * images_per_row > 1 else [axes]
    
    # Generar colores distintos para cada clase
    cmap = cm.get_cmap('tab10', len(class_names))
    class_colors = [cmap(i) for i in range(len(class_names))]
    
    for i, ax in enumerate(axes):
        if i < len(all_images):
            # Mostrar imagen
            img = all_images[i]
            
            # Normalizar la imagen si es necesario
            if img.max() > 1.0:
                img = img.astype("uint8")
            
            # Si la imagen es en escala de grises (1 canal), convertirla a RGB
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            # Obtener predicción y etiqueta real
            pred_class = predicted_classes[i]
            true_class = true_classes[i]
            
            # Obtener probabilidades para cada clase
            probs = predictions_prob[i]
            
            # Determinar colores para título y barra
            title_color = "green" if pred_class == true_class else "red"
            
            # Configurar título
            ax.set_title(
                f"Pred: {class_names[pred_class]}\nReal: {class_names[true_class]}", 
                color=title_color, 
                fontsize=10
            )
            
            # Quitar ejes
            ax.axis('off')
            
            # Añadir barras de probabilidad debajo de la imagen
            bar_height = 0.1
            bar_positions = np.arange(len(probs)) * bar_height
            
            # Posición para las barras de probabilidad
            bar_ax = ax.inset_axes([0.1, -0.3, 0.8, 0.2])
            
            # Dibujar barras para cada clase
            for j, (prob, color) in enumerate(zip(probs, class_colors)):
                bar_ax.barh(j, prob, height=bar_height*0.8, color=color, alpha=0.7)
                # Añadir texto de probabilidad
                bar_ax.text(prob + 0.01, j, f"{prob:.2f}", va='center', fontsize=8)
            
            # Configurar ejes para las barras
            bar_ax.set_yticks(range(len(class_names)))
            bar_ax.set_yticklabels(class_names, fontsize=8)
            bar_ax.set_xlim(0, 1.1)
            bar_ax.set_title("Probabilidades", fontsize=9)
            bar_ax.tick_params(axis='both', which='major', labelsize=7)
            
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    # Mostrar estadísticas de predicción
    correct = np.sum(predicted_classes == true_classes)
    print(f"Precisión en las {len(all_images)} imágenes mostradas: {correct/len(all_images):.2%}")
    
    # Mostrar distribución de clases en la muestra
    unique_classes, counts = np.unique(true_classes, return_counts=True)
    print("\nDistribución de clases en la muestra:")
    for cls, count in zip(unique_classes, counts):
        print(f"  {class_names[cls]}: {count} imágenes ({count/len(all_images):.1%})")