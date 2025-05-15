#  Fundamentos de las Redes Neuronales Convolucionales y su Aplicación en la Detección de Enfermedades Oculares

- **Autor:** Pablo González Martín  
- **Director:** Carlos Javier Pérez González  
- **Grado en Matemáticas**  
- **Universidad de La Laguna**  
- **Curso académico:** 2024/25

--- Contacto:

- pablo.glez.mar@gmail.com


## ⚙️ Instalación y uso

1. Clona este repositorio (será necesario tener git descargado):
   ```bash
   git clone https://github.com/usuario/repositorio.git
   cd repositorio
2. Crea un entorno virtual a través de la terminal de bash (opcional pero recomendado):

python -m venv venv

.\venv\Scripts\Activate.ps1 # Windows

source venv/bin/activate  # macOS/Linux

3. Instala las dependencias:

pip install -r requirements.txt

4. Accede a los siguientes enlaces para descargar los datsets. Dos formas:

- Descargarlos y subir las diferentes carpetas a MiUnidad en el Drive

- Descargarlos en local y colocarlos en las carpetas de sus proyectos tal y como se indica en (Resumen de contenidos más abajo).

!pip install gdown
!gdown https://drive.google.com/uc?id=ID_DEL_ARCHIVO   (especificados debajo)

DESCARGA:

📁 dataset_clasificador_binario    https://drive.google.com/drive/folders/1s6GWY313yyX3W-JFVfXYmeEeiNzL_CEY?usp=sharing

📁 dataset_enfermedades_oculares   https://drive.google.com/drive/folders/1avldzS-aNxJBJqYYYgorylWAQoN0ck_O?usp=sharing

📁 dataset_retinopatia_diabetica   https://drive.google.com/drive/folders/1KrMutkM8Cihpu47csOO5G-JExAgaoe8g?usp=sharing


5. Ejecuta el notebook:
Descarga o abre notebook.ipynb con Jupyter Lab, VSCode o Google Colab. Es importante tener el archivo de func.py respectivo a cada proyecto, en caso de tenerlo, en el mismo directorio.


## 🧪 Resumen de contenidos

├── 📓 test_cpu_vs_gpu.ipynb         # Comparativas de rendimiento entre CPU y GPU

├── 📓 redneuronal_cap1.ipynb        # Fundamentos teóricos y visualizaciones

│

│

├── 📁 clasificador_binario/

│   ├── 📁 dataset_clasificador_binario/   # Datos para clasificación binaria

│   ├── 📓 clasificadorbinario.ipynb       # Implementación del clasificador binario

│   └── 📄 func1.py                        # Funciones auxiliares

│

│

├── 📁 clasificador_enfermedades_oculares/

│   ├── 📁 dataset_enfermedades_oculares/                        # Dataset de enfermedades oculares

│   ├── 📓 clasificadorbinario.ipynb       # Clasificador multiclase


│

│

├── 📁 clasificador_retinopatia_diabetica/

│    ├── 📁 dataset_retinopatia_diabetica/                        # Dataset de retinopatía diabética

│    ├── 📓 clasificador_retinopatia.ipynb  # Clasificador de gravedad

│    └── 📄 funcionalidades.py              # Funciones auxiliares