README — Redes Neuronales con NumPy, Keras y PyTorch
Este repositorio reúne una serie de laboratorios orientados al estudio de las redes neuronales artificiales, abordando desde implementaciones manuales con NumPy hasta el uso de frameworks de alto nivel como TensorFlow/Keras y PyTorch. Los ejercicios cubren problemas de clasificación binaria, multiclase y visión por computadora (MNIST).

Contenido
Laboratorio A — Clasificación binaria con NumPy

Laboratorio B — Red multicapa con NumPy

Laboratorio C — Red multicapa con TensorFlow + Keras

Laboratorio D — Clasificación multiclase con Keras

Laboratorio E — MNIST con TensorFlow + Keras

Laboratorio F — MNIST con PyTorch

Requisitos

Conclusiones

Laboratorio A — Clasificación binaria con NumPy
Objetivo: diferenciar entre manzanas y bananos a partir de tres características (peso, azúcar y madurez).

Pasos realizados:

Construcción manual de la matriz X (20 muestras × 3 características) y el vector de etiquetas y (0 = manzana, 1 = banano).

División de los datos en entrenamiento (80%) y prueba (20%) con train_test_split, usando stratify para conservar la proporción de clases.

Estandarización de las características con StandardScaler.

Definición de la función sigmoide y de la pérdida entropía cruzada binaria.

Inicialización aleatoria de pesos y sesgo.

Entrenamiento mediante descenso del gradiente con propagación hacia adelante y retropropagación.

Evaluación sobre el conjunto de prueba usando un umbral de 0.5.

Clasificación de una fruta nueva.

Resultado: accuracy de 1.0 en el conjunto de prueba.

Laboratorio B — Red multicapa con NumPy
Objetivo: resolver el mismo problema binario usando una red con una capa oculta.

Arquitectura: 3 entradas → 4 neuronas ocultas (ReLU) → 1 salida (Sigmoide).

Pasos realizados:

Definición de ReLU y su derivada, además de la sigmoide.

Inicialización de pesos con inicialización He (√(2/n)).

Implementación de la propagación hacia adelante en dos capas.

Cálculo de gradientes mediante retropropagación en ambas capas.

Entrenamiento durante 3000 épocas con tasa de aprendizaje 0.05.

Resultado: el error descendió de 0.9529 a 0.0096, con accuracy de 1.0 en prueba.

Laboratorio C — Red multicapa con TensorFlow + Keras
Objetivo: replicar la red del Laboratorio B usando la API Sequential de Keras.

Arquitectura: 3 entradas → 4 neuronas (ReLU) → 1 salida (Sigmoide).

Pasos realizados:

Construcción del modelo con keras.Sequential.

Configuración con optimizador Adam y pérdida binary_crossentropy.

Entrenamiento durante 200 épocas.

Visualización de la curva de aprendizaje.

Evaluación sobre el conjunto de prueba y clasificación de una nueva fruta.

Resultado: error final de 0.094 y accuracy de 1.0, confirmando la equivalencia con la implementación manual.

Laboratorio D — Clasificación multiclase con Keras
Objetivo: extender el problema a tres clases (manzana, banano y naranja).

Arquitectura: 3 entradas → 6 neuronas (ReLU) → 3 salidas (Softmax).

Pasos realizados:

Construcción del conjunto X_multi con 30 muestras (10 por clase).

División y estandarización de los datos.

Configuración del modelo con sparse_categorical_crossentropy.

Entrenamiento durante 300 épocas.

Predicción usando argmax sobre las probabilidades.

Resultado: accuracy de 1.0 y clasificación correcta de una fruta nueva como NARANJA.

Laboratorio E — MNIST con TensorFlow + Keras
Objetivo: clasificar dígitos manuscritos (0–9) usando el conjunto MNIST.

Arquitectura: entrada 28×28 → Flatten (784) → 128 neuronas (ReLU) → 10 salidas (Softmax).

Pasos realizados:

Carga del conjunto MNIST (60 000 entrenamiento / 10 000 prueba).

Normalización de píxeles al rango [0, 1].

Construcción y compilación del modelo (Adam + sparse_categorical_crossentropy).

Entrenamiento durante 5 épocas con validation_split=0.1.

Visualización de las curvas de pérdida y precisión.

Análisis de imágenes mal clasificadas.

Resultado: accuracy de entrenamiento 98.7% y de validación 97.5%.

Laboratorio F — MNIST con PyTorch
Objetivo: resolver el mismo problema de MNIST usando PyTorch.

Arquitectura: 784 entradas → 128 neuronas (ReLU) → 10 salidas.

Pasos realizados:

Carga de MNIST con torchvision.datasets y transformación ToTensor().

Creación de DataLoader con lotes de 64.

Definición del modelo como subclase de nn.Module.

Configuración con CrossEntropyLoss y optimizador Adam.

Entrenamiento durante 3 épocas con ciclo explícito (zero_grad → forward → loss → backward → step).

Evaluación sobre el conjunto de prueba y predicción de una imagen individual.

Resultado: accuracy de 0.97 en prueba.
