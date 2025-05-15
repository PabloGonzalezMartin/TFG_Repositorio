import matplotlib.pyplot as plt
import os
from collections import defaultdict
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras import backend as K
import pandas as pd
from tensorflow.keras.layers import Conv2D



def dataset_analysis(path):
    subfolders = os.listdir(path)

    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            format_dimensions_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

            for file in files:
                try:
                    file_path = os.path.join(subfolder_path, file)
                    with Image.open(file_path) as img:
                        image_type = img.format.upper()  # Format (e.g., JPEG, PNG)
                        image_dimensions = img.size  # (width, height)
                        image_mode = img.mode  # Mode (e.g., RGB, L)


                        # Calculate bit depth
                        if image_mode == "1":  # 1-bit pixels, black and white, stored with one pixel per byte
                            bit_depth = 1
                        elif image_mode == "L":  # 8-bit pixels, grayscale
                            bit_depth = 8
                        elif image_mode == "P":  # 8-bit pixels, mapped to any other mode using a color palette
                            bit_depth = 8
                        elif image_mode == "RGB":  # 8-bit pixels, true color
                            bit_depth = 24  # 8 bits per channel
                        elif image_mode == "RGBA":  # 8-bit pixels, true color with transparency mask
                            bit_depth = 32  # 8 bits per channel
                        elif image_mode == "CMYK":  # 8-bit pixels, color separation
                            bit_depth = 32  # 8 bits per channel
                        else:
                            bit_depth = "Unknown"

                        format_dimensions_counts[image_type][(image_dimensions, bit_depth)][image_mode] += 1
                except Exception as e:
                    print(f"Exception processing '{file}' in '{subfolder}': {e}")

            print('--------'*10)
            print(f"Subfolder '{subfolder}' contains ({len(files)} files):")
            for format, dimensions_counts in format_dimensions_counts.items():
                print(f"- {sum(sum(counts.values()) for counts in dimensions_counts.values())} images of format {format}:")
                for (dimensions, bit_depth), counts in dimensions_counts.items():
                    for mode, count in counts.items():
                        print(f"  - {count} images with dimensions {dimensions}, bit depth {bit_depth}, mode {mode}")


def plot_directory_image_counts(path, colors=['lightskyblue', 'mediumseagreen', 'indianred', 'orange']):
    """
    Count and plot the number of files in each subdirectory of the given path.
    
    Parameters:
    path (str): The main directory path to analyze
    colors (list): List of colors to use for the bars in the plot
    
    Returns:
    tuple: A tuple containing (subfolders, image_counts) for further use if needed
    """
    # Get all subdirectories
    subfolders = os.listdir(path)
    
    # Count files in each subdirectory
    image_counts = []
    valid_subfolders = []
    
    for directory in subfolders:
        sub_dir = os.path.join(path, directory)
        if os.path.isdir(sub_dir):
            file_count = len(os.listdir(sub_dir))
            image_counts.append(file_count)
            valid_subfolders.append(directory)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_subfolders, image_counts, color=colors[:len(valid_subfolders)])
    
    # Add value counts on each bar
    for i, count in enumerate(image_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    # Set labels and title
    plt.xlabel('Directory')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Número de imágenes')
    plt.title('Número de imágenes por directorio')
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return valid_subfolders, image_counts

def dataset_size_analysis(path):
    format_dimensions_counts = defaultdict(int)

    subfolders = os.listdir(path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)

            for file in files:
                try:
                    file_path = os.path.join(subfolder_path, file)
                    with Image.open(file_path) as img:
                        image_dimensions = img.size
                        image_mode = img.mode

                        # Calculate bit depth
                        bit_depth = {
                            "1": 1,
                            "L": 8,
                            "P": 8,
                            "RGB": 24,
                            "RGBA": 32,
                            "CMYK": 32
                        }.get(image_mode, "Unknown")

                        # Update counts
                        format_dimensions_counts[(image_dimensions, bit_depth)] += 1

                except Exception as e:
                    print(f"Exception processing '{file}' in '{subfolder}': {e}")

    # Plotting dimensions and bit depths
    plt.figure(figsize=(10, 5))
    labels = [f"{dims}, {depth} bit" for (dims, depth) in format_dimensions_counts]
    sizes = list(format_dimensions_counts.values())
    total = sum(sizes)
    bars = plt.bar(labels, sizes, color='blue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('Number of Images')
    plt.title('Image Distribution by Dimensions and Bit Depth')

    # Adding percentage labels above the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{100 * yval/total:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def dataset_size_analysis_by_class(path):
    # Dictionary to store counts: {subfolder: {image_size: count}}
    folder_size_counts = defaultdict(lambda: defaultdict(int))

    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                try:
                    file_path = os.path.join(subfolder_path, file)
                    with Image.open(file_path) as img:
                        dims = img.size
                        folder_size_counts[subfolder][dims] += 1
                except Exception as e:
                    print(f"Exception processing '{file}' in '{subfolder}': {e}")

    # Create a single plot
    plt.figure(figsize=(15, 7))
    
    # Determine unique image sizes across all folders for consistent coloring and grouping
    all_sizes = set(size for sizes in folder_size_counts.values() for size in sizes)
    all_sizes = sorted(all_sizes, key=lambda s: (s[0] * s[1]))  # Sort by area

    subfolder_names = list(folder_size_counts.keys())
    bar_width = 0.15  # Width of bars
    indices = range(len(subfolder_names))

    for i, size in enumerate(all_sizes):
        counts = [folder_size_counts[subfolder].get(size, 0) for subfolder in subfolder_names]
        plt.bar([index + i * bar_width for index in indices], counts, bar_width, label=f'{size[0]}x{size[1]}')

    plt.xticks([index + (len(all_sizes) - 1) * bar_width / 2 for index in indices], subfolder_names, rotation=45, ha="right")
    plt.ylabel('Number of Images')
    plt.title('Image Size Distribution by Subfolder')
    plt.legend(title="Image Size")
    plt.tight_layout()
    plt.show()



def random_photos_from_folders(base_folder):
    # Walk through all directories and files in the base_folder
    for root, dirs, files in os.walk(base_folder):
        # Filter to get only files that are images
        images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(images) >= 4:  # Ensure there are at least 4 images
            selected_images = random.sample(images, 4)  # Randomly select 4 images

            # Display selected images
            fig, axs = plt.subplots(1, 4, figsize=(12, 2))  # Create a 1x4 grid of plots
            for idx, img_name in enumerate(selected_images):
                img_path = os.path.join(root, img_name)
                img = Image.open(img_path)
                axs[idx].imshow(img)
                axs[idx].axis('off')  # Hide axes

                # Extract sub-folder name from the root path
                subfolder_name = os.path.basename(root)
                # Set the title to include image name and sub-folder name
                axs[idx].set_title(f"{img_name}\n({subfolder_name})")

            plt.show()

def load_and_preprocess_image(path, label, image_size=(256, 256), data_augmentation=True):
    # Read the image file
    image = tf.io.read_file(path)

    # Extract file extension
    file_extension = tf.strings.split(path, '.')[-1]

    # Decode based on file extension using tf.cond
    def decode_jpeg():
        return tf.image.decode_jpeg(image, channels=3)

    def decode_png():
        return tf.image.decode_png(image, channels=3)

    def decode_bmp():
        return tf.image.decode_bmp(image, channels=3)

    def decode_gif():
        # Decode GIF and take the first frame
        return tf.squeeze(tf.image.decode_gif(image), axis=0)

    # Handle each format
    image = tf.cond(tf.math.equal(file_extension, 'jpg'), decode_jpeg,
            lambda: tf.cond(tf.math.equal(file_extension, 'jpeg'), decode_jpeg,
            lambda: tf.cond(tf.math.equal(file_extension, 'png'), decode_png,
            lambda: tf.cond(tf.math.equal(file_extension, 'bmp'), decode_bmp,
            lambda: tf.cond(tf.math.equal(file_extension, 'gif'), decode_gif,
            decode_jpeg)))))

    # Resize and normalize
    image = tf.image.resize(image, image_size)

    # Convertir explícitamente a float32 aquí
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1] range

    # Apply data augmentation if in training mode
    if data_augmentation == True:
        # Randomly flip the image horizontally
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Randomly zoom in
        zoom_size = (int(image_size[0] * 1.04), int(image_size[1] * 1.04))  # Zoom in slightly
        image = tf.image.resize_with_crop_or_pad(image, zoom_size[0], zoom_size[1])
        image = tf.image.random_crop(image, size=[image_size[0], image_size[1], 3])

        # Randomly adjust contrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Asegurar que siga siendo float32 después del data augmentation
        image = tf.cast(image, tf.float32)

    return image, label

def plot_training_history(history, figsize=(13, 5)):
    """
    Plot the training and validation accuracy and loss from a model history.
    
    Parameters:
    history: The history object returned from model.fit()
    figsize (tuple): Size of the figure for each plot (width, height)
    
    Returns:
    None: Displays the plots
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=figsize)
    plt.plot(history.history['accuracy'], color="#E74C3C", marker='o')
    plt.plot(history.history['val_accuracy'], color='#641E16', marker='h')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], color="#E74C3C", marker='o')
    plt.plot(history.history['val_loss'], color='#641E16', marker='h')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def analizar_modelo_densenet(modelo):
    """
    Analiza y muestra las características principales de un modelo DenseNet121

    Args:
        modelo: Modelo keras cargado (DenseNet121)

    Returns:
        dict: Diccionario con información detallada del modelo
    """
    import numpy as np

    # Obtener información básica
    info_modelo = {
        "nombre": modelo.name,
        "numero_capas": len(modelo.layers),
        "parametros_totales": modelo.count_params(),
        "parametros_entrenables": int(np.sum([K.count_params(w) for w in modelo.trainable_weights])),
        "parametros_no_entrenables": int(np.sum([K.count_params(w) for w in modelo.non_trainable_weights])),
        "tipo_entrada": str(modelo.input.shape),
        "tipo_salida": str(modelo.output.shape),
    }

    # Obtener estructura detallada
    bloques_densenet = []
    capas_por_tipo = {}

    for capa in modelo.layers:
        # Contar tipos de capas
        tipo_capa = capa.__class__.__name__
        if tipo_capa in capas_por_tipo:
            capas_por_tipo[tipo_capa] += 1
        else:
            capas_por_tipo[tipo_capa] = 1

        # Identificar bloques densos
        if 'dense_block' in capa.name:
            bloques_densenet.append({
                "nombre": capa.name,
                "forma_salida": str(capa.output_shape),
                "parametros": capa.count_params()
            })

    info_modelo["capas_por_tipo"] = capas_por_tipo
    info_modelo["bloques_densos"] = bloques_densenet

    # Imprimir un resumen
    print(f"Análisis del modelo: {info_modelo['nombre']}")
    print("-" * 50)
    print(f"Total parámetros: {info_modelo['parametros_totales']:,}")
    print(f"Parámetros entrenables: {info_modelo['parametros_entrenables']:,}")
    print(f"Parámetros no entrenables: {info_modelo['parametros_no_entrenables']:,}")
    print(f"Forma de entrada: {info_modelo['tipo_entrada']}")
    print(f"Forma de salida: {info_modelo['tipo_salida']}")
    print("\nDistribución de capas:")
    for tipo, cantidad in capas_por_tipo.items():
        print(f"  - {tipo}: {cantidad}")

    print("\nBloques densos:")
    for bloque in bloques_densenet:
        print(f"  - {bloque['nombre']} | Salida: {bloque['forma_salida']} | Parámetros: {bloque['parametros']:,}")

    return info_modelo

def plot_enhanced_confusion_matrix(conf_mat, classes, title='Matriz de Confusión Detallada',
                                   plot_type='all', figsize_single=(10, 8), figsize_all=(18, 12)):
    """
    Visualización mejorada de la matriz de confusión con análisis detallado.

    Args:
        conf_mat: Matriz de confusión
        classes: Nombres de las clases
        title: Título del gráfico
        plot_type: Tipo de visualización a generar ('all', 'absolute', 'percentage', 'errors', 'metrics')
        figsize_single: Tamaño de figura para gráficas individuales
        figsize_all: Tamaño de figura para la visualización completa

    Returns:
        Figura de matplotlib
    """
    # Convertir a numpy array si no lo es
    conf_mat = np.array(conf_mat)

    # Normalizar matriz para obtener porcentajes
    conf_mat_percent = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis] * 100

    # OPCIÓN 1: Crear la visualización completa con todas las gráficas
    if plot_type == 'all':
        # Configuración de la figura principal
        fig = plt.figure(figsize=figsize_all)

        # Crear grid para múltiples visualizaciones
        gs = fig.add_gridspec(2, 3)

        # 1. Matriz de confusión absoluta (números)
        ax1 = fig.add_subplot(gs[0, 0:2])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
        disp.plot(include_values=True, cmap='viridis', ax=ax1, xticks_rotation=45,
                values_format='d', colorbar=False)
        ax1.set_title('Matriz de Confusión (valores absolutos)', fontsize=14)
        ax1.set_xlabel('Predicción', fontsize=12)
        ax1.set_ylabel('Valor Real', fontsize=12)

        # 2. Matriz de confusión normalizada (porcentajes)
        ax2 = fig.add_subplot(gs[0, 2])
        sns.heatmap(conf_mat_percent, annot=True, fmt='.1f', cmap='coolwarm',
                    xticklabels=classes, yticklabels=classes, cbar=True, ax=ax2)
        ax2.set_title('Matriz de Confusión (porcentajes)', fontsize=14)
        ax2.set_xlabel('Predicción', fontsize=12)
        ax2.set_ylabel('Valor Real', fontsize=12)

        # 3. Análisis de principales errores
        ax3 = fig.add_subplot(gs[1, 0])

        # Identificar los principales errores (fuera de la diagonal)
        error_matrix = conf_mat.copy()
        np.fill_diagonal(error_matrix, 0)  # Eliminar la diagonal (aciertos)

        # Ordenar los errores de mayor a menor
        error_indices = np.dstack(np.unravel_index(np.argsort(error_matrix.ravel())[::-1],
                                                error_matrix.shape))[0]

        # Mostrar los 10 principales errores o menos si hay menos errores
        top_n = min(10, np.count_nonzero(error_matrix))

        # Preparar datos para el gráfico
        top_errors = []
        error_labels = []
        error_percents = []

        for i in range(top_n):
            true_class_idx = error_indices[i][0]
            pred_class_idx = error_indices[i][1]
            error_count = error_matrix[true_class_idx, pred_class_idx]

            if error_count == 0:  # No hay más errores
                break

            true_class = classes[true_class_idx]
            pred_class = classes[pred_class_idx]
            error_labels.append(f"{true_class} → {pred_class}")
            top_errors.append(error_count)

            # Calcular el porcentaje de error respecto al total de la clase real
            class_total = conf_mat[true_class_idx].sum()
            error_percents.append((error_count / class_total) * 100)

        # Gráfico de barras para los principales errores
        bars = ax3.barh(range(len(top_errors)), top_errors, color='salmon')
        ax3.set_yticks(range(len(top_errors)))
        ax3.set_yticklabels(error_labels)
        ax3.set_title('Principales Errores de Clasificación', fontsize=14)
        ax3.set_xlabel('Número de Errores', fontsize=12)

        # Añadir valores en las barras
        for i, (bar, percent) in enumerate(zip(bars, error_percents)):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{percent:.1f}%", va='center', fontsize=10)

        # 4. Métricas por clase
        ax4 = fig.add_subplot(gs[1, 1:])

        # Calcular precision, recall y f1-score para cada clase
        precision, recall, f1, support = precision_recall_fscore_support(
            np.repeat(np.arange(len(classes)), np.sum(conf_mat, axis=1)),
            np.concatenate([np.repeat(i, col.sum()) for i, col in enumerate(conf_mat.T)])
        )

        # Calcular la exactitud por clase (accuracy per class)
        accuracy_per_class = np.diag(conf_mat) / np.sum(conf_mat, axis=1)

        # Crear un DataFrame de métricas para mejor visualización
        metrics_df = pd.DataFrame({
            'Precisión': precision,
            'Exhaustividad': recall,
            'F1-Score': f1,
            'Exactitud': accuracy_per_class,
            'Support': support
        }, index=classes)

        # Ordenar por F1-Score ascendente para ver las clases más problemáticas primero
        metrics_df = metrics_df.sort_values('F1-Score')

        # Gráfico de métricas por clase
        metrics_df[['Precisión', 'Exhaustividad', 'F1-Score', 'Exactitud']].plot(
            kind='barh', ax=ax4, figsize=(10, 8), width=0.8)

        ax4.set_title('Métricas por Clase', fontsize=14)
        ax4.set_xlabel('Puntuación', fontsize=12)
        ax4.set_ylabel('Clase', fontsize=12)
        ax4.set_xlim([0, 1])
        ax4.grid(axis='x', linestyle='--', alpha=0.7)
        ax4.legend(loc='lower right')

        # Añadir información de support
        for i, v in enumerate(metrics_df['Support']):
            ax4.text(0.01, i, f"n={v}", va='center', fontsize=9, color='black')

        # Ajustar el diseño para evitar solapamientos
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, y=0.98)
        fig.subplots_adjust(top=0.93)

        return fig

    # OPCIÓN 2: Crear visualizaciones individuales
    elif plot_type == 'absolute':
        # Matriz de confusión absoluta (números)
        fig, ax = plt.subplots(figsize=figsize_single)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
        disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation=45,
                values_format='d', colorbar=True)
        ax.set_title('Matriz de Confusión (valores absolutos)', fontsize=14)
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Valor Real', fontsize=12)
        plt.tight_layout()
        return fig

    elif plot_type == 'percentage':
        # Matriz de confusión normalizada (porcentajes)
        fig, ax = plt.subplots(figsize=figsize_single)
        sns.heatmap(conf_mat_percent, annot=True, fmt='.1f', cmap='coolwarm',
                    xticklabels=classes, yticklabels=classes, cbar=True, ax=ax)
        ax.set_title('Matriz de Confusión (porcentajes)', fontsize=14)
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Valor Real', fontsize=12)
        plt.tight_layout()
        return fig

    elif plot_type == 'errors':
        # Análisis de principales errores
        fig, ax = plt.subplots(figsize=figsize_single)

        # Identificar los principales errores (fuera de la diagonal)
        error_matrix = conf_mat.copy()
        np.fill_diagonal(error_matrix, 0)  # Eliminar la diagonal (aciertos)

        # Ordenar los errores de mayor a menor
        error_indices = np.dstack(np.unravel_index(np.argsort(error_matrix.ravel())[::-1],
                                                error_matrix.shape))[0]

        # Mostrar los 10 principales errores o menos si hay menos errores
        top_n = min(10, np.count_nonzero(error_matrix))

        # Preparar datos para el gráfico
        top_errors = []
        error_labels = []
        error_percents = []

        for i in range(top_n):
            true_class_idx = error_indices[i][0]
            pred_class_idx = error_indices[i][1]
            error_count = error_matrix[true_class_idx, pred_class_idx]

            if error_count == 0:  # No hay más errores
                break

            true_class = classes[true_class_idx]
            pred_class = classes[pred_class_idx]
            error_labels.append(f"{true_class} → {pred_class}")
            top_errors.append(error_count)

            # Calcular el porcentaje de error respecto al total de la clase real
            class_total = conf_mat[true_class_idx].sum()
            error_percents.append((error_count / class_total) * 100)

        # Gráfico de barras para los principales errores
        bars = ax.barh(range(len(top_errors)), top_errors, color='salmon')
        ax.set_yticks(range(len(top_errors)))
        ax.set_yticklabels(error_labels)
        ax.set_title('Principales Errores de Clasificación', fontsize=14)
        ax.set_xlabel('Número de Errores', fontsize=12)

        # Añadir valores en las barras
        for i, (bar, percent) in enumerate(zip(bars, error_percents)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                  f"{percent:.1f}%", va='center', fontsize=10)

        plt.tight_layout()
        return fig

    elif plot_type == 'metrics':
        # Métricas por clase
        fig, ax = plt.subplots(figsize=figsize_single)

        # Calcular precision, recall y f1-score para cada clase
        precision, recall, f1, support = precision_recall_fscore_support(
            np.repeat(np.arange(len(classes)), np.sum(conf_mat, axis=1)),
            np.concatenate([np.repeat(i, col.sum()) for i, col in enumerate(conf_mat.T)])
        )

        # Calcular la exactitud por clase (accuracy per class)
        accuracy_per_class = np.diag(conf_mat) / np.sum(conf_mat, axis=1)

        # Crear un DataFrame de métricas para mejor visualización
        metrics_df = pd.DataFrame({
            'Precisión': precision,
            'Exhaustividad': recall,
            'F1-Score': f1,
            'Exactitud': accuracy_per_class,
            'Support': support
        }, index=classes)

        # Ordenar por F1-Score ascendente para ver las clases más problemáticas primero
        metrics_df = metrics_df.sort_values('F1-Score')

        # Gráfico de métricas por clase
        metrics_df[['Precisión', 'Exhaustividad', 'F1-Score', 'Exactitud']].plot(
            kind='barh', ax=ax, width=0.8)

        ax.set_title('Métricas por Clase', fontsize=14)
        ax.set_xlabel('Puntuación', fontsize=12)
        ax.set_ylabel('Clase', fontsize=12)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')

        # Añadir información de support
        for i, v in enumerate(metrics_df['Support']):
            ax.text(0.01, i, f"n={v}", va='center', fontsize=9, color='black')

        plt.tight_layout()
        return fig

    else:
        raise ValueError("Tipo de gráfico no válido. Opciones: 'all', 'absolute', 'percentage', 'errors', 'metrics'")

def analizar_resultados_modelo(y_true, y_pred, label_encoder, conf_mat=None, plot_type='all', save_individual=False):
    """
    Analiza y visualiza los resultados del modelo.

    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones del modelo
        label_encoder: Codificador de etiquetas
        conf_mat: Matriz de confusión (opcional, se calcula si no se proporciona)
        plot_type: Tipo de visualización a generar ('all', 'absolute', 'percentage', 'errors', 'metrics')
        save_individual: Si es True, guarda cada gráfica individualmente además de mostrarlas
    """
    # Obtener las clases
    class_names = label_encoder.classes_

    # Calcular matriz de confusión si no se proporciona
    if conf_mat is None:
        conf_mat = confusion_matrix(y_true, y_pred)

    # Imprimir estadísticas
    print(f"Total de muestras de prueba: {len(y_true)}")
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Crear visualización según el tipo solicitado
    fig = plot_enhanced_confusion_matrix(conf_mat, class_names, plot_type=plot_type)

    # Si se solicita guardar individuales y se está generando la visualización completa
    if save_individual and plot_type == 'all':
        # Generar y guardar cada gráfica individual
        fig_abs = plot_enhanced_confusion_matrix(conf_mat, class_names, plot_type='absolute')
        fig_abs.savefig('confusion_matrix_absolute.png', dpi=300, bbox_inches='tight')

        fig_pct = plot_enhanced_confusion_matrix(conf_mat, class_names, plot_type='percentage')
        fig_pct.savefig('confusion_matrix_percentage.png', dpi=300, bbox_inches='tight')

        fig_err = plot_enhanced_confusion_matrix(conf_mat, class_names, plot_type='errors')
        fig_err.savefig('confusion_matrix_errors.png', dpi=300, bbox_inches='tight')

        fig_met = plot_enhanced_confusion_matrix(conf_mat, class_names, plot_type='metrics')
        fig_met.savefig('confusion_matrix_metrics.png', dpi=300, bbox_inches='tight')

        print("\nGráficas individuales guardadas con éxito.")

    # Identificar las clases más difíciles de clasificar
    accuracy_per_class = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    class_difficulty = pd.DataFrame({
        'Clase': class_names,
        'Exactitud': accuracy_per_class,
        'Total Muestras': np.sum(conf_mat, axis=1),
        'Errores': np.sum(conf_mat, axis=1) - np.diag(conf_mat)
    })
    class_difficulty['Tasa de Error'] = class_difficulty['Errores'] / class_difficulty['Total Muestras']

    # Ordenar por tasa de error (descendente)
    difficult_classes = class_difficulty.sort_values('Tasa de Error', ascending=False)

    print("\nClases más difíciles de clasificar:")
    print(difficult_classes.to_string(index=False))

    # Recomendaciones basadas en el análisis
    print("\nRecomendaciones:")
    worst_classes = difficult_classes.head(3)['Clase'].values
    print(f"- Considerar aumentar los datos para las clases: {', '.join(worst_classes)}")
    print("- Revisar posibles similitudes entre clases con alta confusión")
    print("- Evaluar aplicar técnicas de balanceo de clases si hay clases minoritarias")

    return fig, difficult_classes



def plot_test_predictions(model, test_dataset, class_labels, num_images=10, random_selection=True, cols=5):
    """
    Plots the predictions of a model on the test dataset, selecting images randomly or sequentially.

    Parameters:
    - model: Trained Keras model to be used for prediction.
    - test_dataset: TensorFlow dataset containing the test images and labels.
    - class_labels: List of class labels.
    - num_images: Number of test images to plot (default is 10).
    - random_selection: If True, select images randomly from the dataset. If False, take the first ones.
    - cols: Number of columns in the grid layout (default is 5).
    """
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import random

    # Convert the dataset to a list to enable random sampling
    all_images = []
    all_true_labels = []

    # Extract all images and labels from the dataset
    for batch_images, batch_labels in test_dataset:
        # Convert batch tensors to numpy arrays and add to our lists
        if isinstance(batch_images, tf.Tensor):
            batch_images = batch_images.numpy()
        if isinstance(batch_labels, tf.Tensor):
            batch_labels = batch_labels.numpy()

        # Handle both single-example and batch cases
        if len(batch_images.shape) == 3:  # Single image (height, width, channels)
            all_images.append(batch_images)
            all_true_labels.append(batch_labels)
        else:  # Batch of images
            for i in range(len(batch_images)):
                all_images.append(batch_images[i])
                all_true_labels.append(batch_labels[i])

    total_available = len(all_images)
    if total_available < num_images:
        print(f"Warning: Only {total_available} images available, using all of them.")
        num_images = total_available

    # Select images (randomly or sequentially)
    if random_selection:
        # Random selection without replacement
        max_idx = min(total_available, 1000)  # Limit to first 1000 images for memory efficiency
        selected_indices = random.sample(range(max_idx), num_images)
    else:
        # Sequential selection
        selected_indices = list(range(num_images))

    # Get selected images and labels
    selected_images = [all_images[i] for i in selected_indices]
    selected_labels = [all_true_labels[i] for i in selected_indices]

    # Make predictions on the selected images
    selected_images_array = np.array(selected_images)
    pred_probs = model.predict(selected_images_array)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Get the probability of the predicted class
    max_probs = np.max(pred_probs, axis=1) * 100

    # Calculate rows and columns for the grid
    rows = (num_images + cols - 1) // cols  # Ceiling division

    # Create a figure with proper dimensions
    # Adjust figure size based on number of rows and columns
    fig_width = min(15, cols * 3)
    fig_height = min(20, rows * 3)

    plt.figure(figsize=(fig_width, fig_height))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)

        # Handle grayscale images (1 channel)
        if len(selected_images[i].shape) == 2 or selected_images[i].shape[-1] == 1:
            plt.imshow(selected_images[i], cmap='gray')
        else:
            plt.imshow(selected_images[i])

        actual_label = class_labels[np.argmax(selected_labels[i]) if len(selected_labels[i].shape) > 0 else selected_labels[i]]
        predicted_label = class_labels[pred_labels[i]]
        probability = max_probs[i]  # Probability of the predicted class

        # Color code - green for correct, red for incorrect predictions
        color = 'green' if actual_label == predicted_label else 'red'

        # More compact title with adequate font size
        title_fontsize = min(12, max(10, 12 - 0.4*cols))
        plt.title(f"Real: {actual_label}\nPred: {predicted_label}\n{probability:.1f}%",
                 color=color, fontsize=title_fontsize)
        plt.axis('off')

    plt.tight_layout()

# Ejemplo de uso:
"""
# Usar la función con selección aleatoria (por defecto)
plot_test_predictions(trained_model, test_ds, class_labels=class_names, num_images=15)

# Usar la función con selección secuencial
plot_test_predictions(trained_model, test_ds, class_labels=class_names,
                     num_images=15, random_selection=False)
"""

def plot_misclassified_predictions(model, test_dataset, class_labels, num_images=20):
    """
    Plots only the misclassified images from the test dataset.

    Parameters:
    - model: Trained Keras model to be used for prediction.
    - test_dataset: TensorFlow dataset containing the test images and labels.
    - class_labels: List of class labels.
    - num_images: Maximum number of misclassified images to plot.
    """
    # Initialize lists to accumulate images and labels
    all_images = []
    all_true_labels = []

    # Extract all images and labels from the dataset
    for batch_images, batch_labels in test_dataset:
        # Convert batch tensors to numpy arrays
        if isinstance(batch_images, tf.Tensor):
            batch_images = batch_images.numpy()
        if isinstance(batch_labels, tf.Tensor):
            batch_labels = batch_labels.numpy()

        # Handle both single-example and batch cases
        if len(batch_images.shape) == 3:  # Single image
            all_images.append(batch_images)
            all_true_labels.append(batch_labels)
        else:  # Batch of images
            for i in range(len(batch_images)):
                all_images.append(batch_images[i])
                all_true_labels.append(batch_labels[i])

        # Limit the number of evaluated images to prevent memory issues
        if len(all_images) >= 1000:
            break

    # Convert to numpy arrays for prediction
    all_images_array = np.array(all_images)
    all_true_labels_array = np.array(all_true_labels)

    # Make predictions
    pred_probs = model.predict(all_images_array)
    pred_labels = np.argmax(pred_probs, axis=1)

    # Find misclassified images
    misclassified_indices = np.where(pred_labels != all_true_labels_array)[0]

    if len(misclassified_indices) == 0:
        print("No misclassified images found in the first 1000 samples.")
        return None

    # Randomly select up to num_images from misclassified
    if len(misclassified_indices) > num_images:
        selected_indices = random.sample(list(misclassified_indices), num_images)
    else:
        selected_indices = misclassified_indices
        print(f"Found {len(selected_indices)} misclassified images.")

    # Plot the misclassified images
    plt.figure(figsize=(15, 20))
    rows = (len(selected_indices) // 5) + (1 if len(selected_indices) % 5 != 0 else 0)

    for i, idx in enumerate(selected_indices):
        plt.subplot(rows, 5, i + 1)
        plt.imshow(all_images[idx])

        actual_label = class_labels[all_true_labels[idx]]
        predicted_label = class_labels[pred_labels[idx]]
        probability = np.max(pred_probs[idx]) * 100  # Probability of the predicted class

        plt.title(f"Real: {actual_label}\nPred: {predicted_label}\nProb: {probability:.1f}%",
                  color='red', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Imágenes Mal Clasificadas", fontsize=16, y=1.02)
    return plt.gcf()

# Ejemplo de uso:
"""
# Visualizar imágenes mal clasificadas
plot_misclassified_predictions(trained_model, test_ds, class_labels=class_names, num_images=15)
"""

def plot_predictions_by_class(model, test_dataset, class_labels, target_class=None, num_images=20, random_selection=True):
    """
    Plots predictions for a specific class or shows examples from all classes.

    Parameters:
    - model: Trained Keras model to be used for prediction.
    - test_dataset: TensorFlow dataset containing the test images and labels.
    - class_labels: List of class labels.
    - target_class: Index of the class to filter by (None = all classes).
    - num_images: Number of test images to plot.
    - random_selection: If True, select images randomly.
    """
    # Initialize lists to accumulate images and labels
    all_images = []
    all_true_labels = []

    # Extract all images and labels from the dataset
    for batch_images, batch_labels in test_dataset:
        # Convert batch tensors to numpy arrays
        if isinstance(batch_images, tf.Tensor):
            batch_images = batch_images.numpy()
        if isinstance(batch_labels, tf.Tensor):
            batch_labels = batch_labels.numpy()

        # Handle both single-example and batch cases
        if len(batch_images.shape) == 3:  # Single image
            all_images.append(batch_images)
            all_true_labels.append(batch_labels)
        else:  # Batch of images
            for i in range(len(batch_images)):
                all_images.append(batch_images[i])
                all_true_labels.append(batch_labels[i])

        # Limit the number of processed images
        if len(all_images) >= 1000:
            break

    # Filter by target class if specified
    if target_class is not None:
        class_indices = [i for i, label in enumerate(all_true_labels) if label == target_class]
        if len(class_indices) == 0:
            print(f"No images found for class '{class_labels[target_class]}'.")
            return None

        title = f"Predicciones para la Clase: {class_labels[target_class]}"
    else:
        class_indices = list(range(len(all_images)))
        title = "Predicciones por Clase"

    # Select images
    if random_selection and len(class_indices) > num_images:
        selected_indices = random.sample(class_indices, num_images)
    else:
        selected_indices = class_indices[:min(num_images, len(class_indices))]
        if len(selected_indices) < num_images:
            print(f"Warning: Only {len(selected_indices)} images available for the selected class.")

    # Get selected images and labels
    selected_images = [all_images[i] for i in selected_indices]
    selected_labels = [all_true_labels[i] for i in selected_indices]

    # Make predictions
    selected_images_array = np.array(selected_images)
    pred_probs = model.predict(selected_images_array)
    pred_labels = np.argmax(pred_probs, axis=1)
    max_probs = np.max(pred_probs, axis=1) * 100

    # Plot
    plt.figure(figsize=(15, 20))
    rows = (len(selected_indices) // 5) + (1 if len(selected_indices) % 5 != 0 else 0)

    for i in range(len(selected_indices)):
        plt.subplot(rows, 5, i + 1)
        plt.imshow(selected_images[i])

        actual_label = class_labels[selected_labels[i]]
        predicted_label = class_labels[pred_labels[i]]
        probability = max_probs[i]

        color = 'green' if actual_label == predicted_label else 'red'
        plt.title(f"Real: {actual_label}\nPred: {predicted_label}\nProb: {probability:.1f}%",
                  color=color, fontsize=20)
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.02)
    return plt.gcf()

def visualize_convolution(dataset, kernel_size=(3, 3), filters=1, strides=(1, 1),
                         padding='VALID', activation=None, use_bias=True):
    """
    Visualiza el efecto de una convolución sobre una imagen aleatoria del dataset.

    Args:
        dataset: Dataset de TensorFlow que contiene imágenes
        kernel_size: Tamaño del kernel (tuple de 2 enteros)
        filters: Número de filtros a aplicar
        strides: Tamaño del stride (tuple de 2 enteros)
        padding: Tipo de padding ('VALID' o 'SAME')
        activation: Función de activación (None por defecto para ver la convolución pura)
        use_bias: Si se debe usar bias en la convolución

    Returns:
        None (muestra las visualizaciones)
    """
    dataset_shuffled = dataset.shuffle(buffer_size=1000)

    for image_batch, _ in dataset_shuffled.take(1):
        # Si el batch tiene varias imágenes, elegimos una aleatoria del batch
        if len(image_batch.shape) == 4:
            idx = tf.random.uniform(shape=[], minval=0, maxval=image_batch.shape[0], dtype=tf.int32)
            image = image_batch[idx]
        else:
            image = image_batch
    # Asegurarnos de que la imagen tiene la forma correcta [height, width, channels]
    if len(image.shape) < 3:
        image = tf.expand_dims(image, axis=-1)

    # Crear una capa de convolución con los parámetros especificados
    conv_layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        activation=activation,
        use_bias=use_bias
    )

    # Aplicar la convolución (necesitamos añadir la dimensión de batch)
    input_img = tf.expand_dims(image, axis=0)
    conv_output = conv_layer(input_img)

    # También calcular la convolución sin activación si se especificó una
    conv_output_raw = None
    if activation is not None:
        conv_layer_raw = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding.lower(),
            activation=None,  # Sin activación
            use_bias=use_bias
        )
        # Copiar los pesos del modelo con activación
        conv_layer_raw.build(input_img.shape)
        conv_layer_raw.set_weights(conv_layer.get_weights())
        conv_output_raw = conv_layer_raw(input_img)

    # Obtener los pesos del kernel para visualizarlos
    weights = conv_layer.get_weights()
    kernel_weights = weights[0]
    bias = None
    if use_bias and len(weights) > 1:
        bias = weights[1]

    # Información de depuración
    print(f"Forma de la imagen: {image.shape}")
    print(f"Forma del kernel: {kernel_weights.shape}")
    print(f"Forma de la salida: {conv_output[0].shape}")

    # Número de canales de entrada a mostrar para los kernels
    num_channels_to_show = min(3, kernel_weights.shape[2])

    # Función para crear un heat map normalizado para mejor visualización
    def create_normalized_heatmap(data):
        data_min = tf.reduce_min(data)
        data_max = tf.reduce_max(data)
        if data_max - data_min < 1e-5:  # Si los valores son muy cercanos
            return data - data_min  # Centrar alrededor de 0
        return (data - data_min) / (data_max - data_min + 1e-8)  # Normalizar entre 0 y 1

    # CASO 1: Un solo filtro
    if filters == 1:
        # Crear figura principal
        fig = plt.figure(figsize=(12, 8))
        grid_spec = plt.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        # Panel 1: Imagen original
        ax1 = fig.add_subplot(grid_spec[0, 0])
        if image.shape[-1] == 1:  # Imagen en escala de grises
            ax1.imshow(tf.squeeze(image), cmap='gray')
        else:  # Imagen a color
            if isinstance(image, tf.Tensor):
                ax1.imshow(image.numpy())
            else:
                ax1.imshow(image)
        ax1.set_title('Imagen Original')
        ax1.axis('off')

        # Panel 2: Información
        ax2 = fig.add_subplot(grid_spec[0, 1:])
        info_text = (
            f"Parámetros de la Convolución:\n"
            f"- Kernel size: {kernel_size}\n"
            f"- Filters: {filters}\n"
            f"- Strides: {strides}\n"
            f"- Padding: {padding}\n"
            f"- Activación: {activation.__name__ if hasattr(activation, '__name__') else activation}\n"
            f"- Dimensiones entrada: {image.shape}\n"
            f"- Dimensiones salida: {conv_output[0].shape}\n"
        )
        if bias is not None:
            info_text += f"- Bias: {bias[0]:.4f}\n"

        ax2.text(0.05, 0.5, info_text, fontsize=12, va='center')
        ax2.axis('off')

        # Panel 3: Kernel
        ax3 = fig.add_subplot(grid_spec[1, 0])
        # Si hay múltiples canales de entrada, promediar para la visualización
        if kernel_weights.shape[2] > 1:
            kernel_to_display = np.mean(kernel_weights[:, :, :, 0], axis=2)
            ax3.set_title('Kernel (Promedio de canales)')
        else:
            kernel_to_display = kernel_weights[:, :, 0, 0]
            ax3.set_title('Kernel')

        # Crear heatmap normalizado para mejor visualización
        ax3.imshow(kernel_to_display, cmap='viridis')

        # Mostrar los valores numéricos del kernel
        height, width = kernel_to_display.shape
        for y in range(height):
            for x in range(width):
                value = kernel_to_display[y, x]
                # Color del texto basado en el valor para mejor contraste
                text_color = 'white' if abs(value) > 0.5 * np.max(np.abs(kernel_to_display)) else 'black'
                ax3.text(x, y, f'{value:.2f}',
                          ha='center', va='center',
                          color=text_color,
                          fontsize=10)
        ax3.axis('off')

        # Panel 4: Resultado de la convolución con activación si especificada
        ax4 = fig.add_subplot(grid_spec[1, 1])
        conv_result = conv_output[0]
        # Normalizar para mejor visualización
        conv_result_display = create_normalized_heatmap(conv_result)
        ax4.imshow(tf.squeeze(conv_result_display), cmap='viridis')
        title = 'Resultado de Convolución'
        if activation is not None:
            act_name = activation.__name__ if hasattr(activation, '__name__') else str(activation)
            title += f' con {act_name}'
        ax4.set_title(title)
        ax4.axis('off')

        # Panel 5: Convolución pura (sin activación) si se especificó activación
        if activation is not None and conv_output_raw is not None:
            ax5 = fig.add_subplot(grid_spec[1, 2])
            conv_result_raw = conv_output_raw[0]
            # Normalizar para mejor visualización
            conv_result_raw_display = create_normalized_heatmap(conv_result_raw)
            ax5.imshow(tf.squeeze(conv_result_raw_display), cmap='viridis')
            ax5.set_title('Convolución Pura (sin activación)')
            ax5.axis('off')

        plt.tight_layout()
        plt.show()

        # Si hay múltiples canales, mostrar kernels de cada canal por separado
        if kernel_weights.shape[2] > 1:
            plt.figure(figsize=(4 * min(3, kernel_weights.shape[2]), 4))
            for i in range(min(3, kernel_weights.shape[2])):
                plt.subplot(1, min(3, kernel_weights.shape[2]), i+1)
                kernel = kernel_weights[:, :, i, 0]
                plt.imshow(kernel, cmap='viridis')
                plt.title(f'Kernel (Canal {i})')

                # Mostrar valores
                height, width = kernel.shape
                for y in range(height):
                    for x in range(width):
                        value = kernel[y, x]
                        text_color = 'white' if abs(value) > 0.5 * np.max(np.abs(kernel)) else 'black'
                        plt.text(x, y, f'{value:.2f}',
                              ha='center', va='center',
                              color=text_color,
                              fontsize=10)
                plt.axis('off')
            plt.tight_layout()
            plt.show()

    # CASO 2: Múltiples filtros
    else:
        # Número de filtros a mostrar (limitar a 8 máximo)
        num_filters_to_show = min(8, filters)

        # Figura principal con información general
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if image.shape[-1] == 1:  # Imagen en escala de grises
            plt.imshow(tf.squeeze(image), cmap='gray')
        else:  # Imagen a color
            if isinstance(image, tf.Tensor):
                plt.imshow(image.numpy())
            else:
                plt.imshow(image)
        plt.title('Imagen Original')
        plt.axis('off')

        # Panel informativo
        plt.subplot(1, 2, 2)
        info_text = (
            f"Parámetros de la Convolución:\n"
            f"- Kernel size: {kernel_size}\n"
            f"- Filters: {filters}\n"
            f"- Strides: {strides}\n"
            f"- Padding: {padding}\n"
            f"- Activación: {activation.__name__ if hasattr(activation, '__name__') else activation}\n"
            f"- Dimensiones entrada: {image.shape}\n"
            f"- Dimensiones salida: {conv_output[0].shape}\n"
        )
        plt.text(0.1, 0.5, info_text, fontsize=12, va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Figura para mostrar los kernels
        fig_kernels = plt.figure(figsize=(16, 4))
        plt.suptitle('Kernels de Convolución')

        for i in range(num_filters_to_show):
            plt.subplot(1, num_filters_to_show, i+1)
            # Promediar si hay múltiples canales
            if kernel_weights.shape[2] > 1:
                kernel = np.mean(kernel_weights[:, :, :, i], axis=2)
            else:
                kernel = kernel_weights[:, :, 0, i]

            plt.imshow(kernel, cmap='viridis')
            plt.title(f'Kernel {i+1}')

            # Mostrar valores si el kernel es pequeño
            if kernel_size[0] <= 5 and kernel_size[1] <= 5:
                height, width = kernel.shape
                for y in range(height):
                    for x in range(width):
                        value = kernel[y, x]
                        text_color = 'white' if abs(value) > 0.5 * np.max(np.abs(kernel)) else 'black'
                        plt.text(x, y, f'{value:.2f}',
                              ha='center', va='center',
                              color=text_color,
                              fontsize=9)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Figura para mostrar los resultados de cada filtro
        plt.figure(figsize=(16, 8))
        plt.suptitle('Resultados de los Filtros de Convolución', fontsize=16)

        for i in range(num_filters_to_show):
            plt.subplot(2, 4, i+1)
            output_slice = conv_output[0, :, :, i]
            # Normalizar para mejor visualización
            output_display = create_normalized_heatmap(output_slice)
            plt.imshow(output_display, cmap='viridis')
            plt.title(f'Salida Filtro {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        # Si tenemos activación, mostrar también los resultados sin activación
        if activation is not None and conv_output_raw is not None:
            plt.figure(figsize=(16, 8))
            plt.suptitle('Resultados sin Activación (Convolución Pura)', fontsize=16)

            for i in range(num_filters_to_show):
                plt.subplot(2, 4, i+1)
                output_slice = conv_output_raw[0, :, :, i]
                # Normalizar para mejor visualización
                output_display = create_normalized_heatmap(output_slice)
                plt.imshow(output_display, cmap='viridis')
                act_name = activation.__name__ if hasattr(activation, '__name__') else str(activation)
                plt.title(f'Filtro {i+1} sin {act_name}')
                plt.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()