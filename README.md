# Taller 4: Arquitecturas SMP y SIMD

**Curso:** Infraestructuras Paralelas y Distribuidas
**Código:** 750023C
**Periodo:** 2025-2
**Profesor:** PhD (C) Manuel Alejandro Pastrana

## Descripción

Este repositorio contiene la implementación de tres ejercicios que exploran las arquitecturas de computación paralela SMP (Symmetric Multiprocessing) y SIMD (Single Instruction, Multiple Data), demostrando sus ventajas en el procesamiento de grandes volúmenes de datos.

## Contenido

- **Ejercicio 1 (30%)**: Arquitectura SMP - Suma paralela de matrices con hilos
- **Ejercicio 2 (30%)**: Arquitectura SIMD - Comparación NumPy vs bucles tradicionales
- **Ejercicio Integrador (40%)**: Sistema híbrido SMP-SIMD

## Requisitos

### Dependencias

```bash
pip install numpy matplotlib
```

O usando el archivo de requisitos:

```bash
pip install -r requirements.txt
```

### Versiones recomendadas
- Python 3.8 o superior
- NumPy 1.20 o superior
- Matplotlib 3.3 o superior

## Estructura del Proyecto

```
taller_4/
├── ejercicio1_smp.py              # Ejercicio 1: Arquitectura SMP
├── ejercicio2_simd.py             # Ejercicio 2: Arquitectura SIMD
├── ejercicio_integrador.py        # Ejercicio Integrador: Híbrido SMP-SIMD
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Este archivo
├── Taller 4.pdf                   # Enunciado del taller
└── resultados/                    # Gráficos generados (creado al ejecutar)
```

## Ejercicios

### Ejercicio 1: Arquitectura SMP

**Objetivo:** Simular un sistema SMP con hilos en Python para calcular la suma de una matriz grande.

**Características:**
- Matriz de 1000×1000 con números aleatorios
- División en bloques de 100×100 (100 bloques totales)
- Cada bloque procesado por un hilo independiente
- Comparación entre versión secuencial y paralela

**Ejecución:**
```bash
python ejercicio1_smp.py
```

**Salida esperada:**
- Suma total de la matriz (secuencial y paralela)
- Tiempos de ejecución de ambos métodos
- Speedup obtenido
- Explicación detallada de la implementación

**Factores clave considerados:**
1. **Threading en Python**: Uso de `threading.Thread` para simular múltiples procesadores
2. **Sincronización**: `threading.Lock` para evitar race conditions
3. **División del trabajo**: Distribución equitativa de bloques entre hilos
4. **GIL (Global Interpreter Lock)**: Limitación de Python para paralelismo real en CPU-bound tasks

### Ejercicio 2: Arquitectura SIMD

**Objetivo:** Comparar el rendimiento de la multiplicación de matrices utilizando NumPy (SIMD) versus bucles tradicionales.

**Características:**
- Dos matrices de 1000×1000 con números aleatorios
- Multiplicación con NumPy (aprovecha instrucciones SIMD)
- Multiplicación con bucles tradicionales de Python
- Visualización de resultados en gráfico de barras

**Ejecución:**
```bash
python ejercicio2_simd.py
```

**Salida esperada:**
- Tiempos de ejecución de ambos métodos
- Gráfico comparativo guardado como `ejercicio2_comparacion.png`
- Tabla de comparación con speedup
- Verificación de correctitud de resultados
- Explicación detallada del rendimiento SIMD

**Factores clave considerados:**
1. **SIMD en NumPy**: Uso de librerías optimizadas (BLAS/LAPACK)
2. **Vectorización**: Procesamiento de múltiples datos con una instrucción
3. **Overhead del intérprete**: Python interpretado vs código compilado
4. **Cache efficiency**: Mejor uso de la jerarquía de memoria

**Resultados típicos:**
- NumPy es entre 100x y 1000x más rápido que bucles tradicionales
- La diferencia aumenta con el tamaño de las matrices
- Demuestra la importancia de usar bibliotecas optimizadas

### Ejercicio Integrador: Sistema Híbrido SMP-SIMD

**Objetivo:** Simular un sistema que combina SMP y SIMD para procesar una gran cantidad de datos.

**Características:**
- Matriz de 10000×10000 con números aleatorios (~763 MB)
- División en bloques de 1000×1000 (100 bloques)
- Procesamiento paralelo con hilos (SMP)
- Operaciones vectorizadas con NumPy dentro de cada hilo (SIMD)
- Comparación con tres métodos: híbrido, secuencial con SIMD, y puro secuencial

**Ejecución:**
```bash
python ejercicio_integrador.py
```

**Salida esperada:**
- Tiempos de los tres métodos
- Gráficos comparativos guardados como `ejercicio_integrador_comparacion.png`
- Análisis de speedup y eficiencia paralela
- Explicación detallada de la arquitectura híbrida

**Factores clave considerados:**
1. **Paralelismo multinivel**:
   - Nivel 1 (SMP): Hilos procesando bloques independientes
   - Nivel 2 (SIMD): Operaciones vectorizadas dentro de cada hilo

2. **Escalabilidad**:
   - Aprovecha múltiples núcleos (SMP)
   - Aprovecha unidades SIMD de cada núcleo

3. **Eficiencia**:
   - Minimiza overhead de sincronización
   - Maximiza localidad de caché

4. **Limitaciones**:
   - GIL de Python puede limitar paralelismo
   - Overhead de creación de hilos
   - Balance entre número de hilos y tamaño de bloque

## Conceptos Clave

### SMP (Symmetric Multiprocessing)

**Definición:** Arquitectura donde múltiples procesadores comparten la misma memoria y sistema operativo.

**Características:**
- Múltiples CPUs o núcleos
- Memoria compartida
- Cada procesador puede ejecutar tareas independientes
- Requiere sincronización para acceso a datos compartidos

**Ventajas:**
- Aprovecha múltiples núcleos
- Escalabilidad en sistemas multi-core
- Reduce tiempo de procesamiento para tareas paralelas

**Desafíos:**
- Sincronización entre hilos
- Race conditions
- Overhead de cambio de contexto

### SIMD (Single Instruction, Multiple Data)

**Definición:** Arquitectura que permite ejecutar la misma operación sobre múltiples datos simultáneamente.

**Características:**
- Una instrucción procesa múltiples elementos
- Procesamiento vectorizado
- Común en CPUs modernas (SSE, AVX, NEON)

**Ventajas:**
- Alto rendimiento en operaciones de datos
- Eficiencia energética
- Ideal para álgebra lineal y procesamiento de imágenes

**Aplicaciones:**
- Machine Learning
- Procesamiento de imágenes/video
- Simulaciones científicas
- Procesamiento de señales

### Sistema Híbrido SMP-SIMD

**Definición:** Combinación de paralelismo a nivel de hilos (SMP) y paralelismo a nivel de datos (SIMD).

**Ventajas:**
- Maximiza uso de recursos del sistema
- Paralelismo en múltiples niveles
- Mejor rendimiento para grandes volúmenes de datos

**Casos de uso:**
- Procesamiento masivo de datos
- High Performance Computing (HPC)
- Deep Learning
- Análisis de Big Data

## Resultados y Análisis

### Ejercicio 1: SMP

**Resultados típicos:**
```
Tiempo secuencial:  0.045000 segundos
Tiempo paralelo:    0.025000 segundos
Speedup:            1.80x
Mejora porcentual:  44.44%
```

**Análisis:**
- El speedup depende del número de núcleos disponibles
- El GIL de Python limita el paralelismo real
- Para mejor rendimiento en CPU-bound tasks, considerar `multiprocessing`

### Ejercicio 2: SIMD

**Resultados típicos:**
```
NumPy (SIMD):           0.15 segundos
Bucles Tradicionales:   150.00 segundos
Speedup:                1000x
```

**Análisis:**
- Diferencia dramática demuestra poder de SIMD
- NumPy usa código compilado optimizado
- Esencial usar bibliotecas optimizadas para rendimiento

### Ejercicio Integrador

**Resultados típicos:**
```
Híbrido SMP-SIMD:    2.5 segundos
Secuencial (SIMD):   8.0 segundos
Puro Secuencial:     800.0 segundos (estimado)

Speedup vs Secuencial: 3.2x
Speedup vs Puro:       320x
```

**Análisis:**
- Combinar SMP y SIMD multiplica beneficios
- Escalabilidad limitada por número de núcleos
- Overhead de threading reduce eficiencia teórica

## Conclusiones

1. **SMP** es efectivo para tareas que pueden dividirse en subtareas independientes
2. **SIMD** es crucial para operaciones de álgebra lineal y procesamiento de datos
3. **Sistemas híbridos** maximizan el rendimiento combinando ambas técnicas
4. **NumPy** es fundamental para computación científica en Python
5. Las limitaciones de Python (GIL) pueden mitigarse con:
   - Bibliotecas optimizadas (NumPy, que libera el GIL)
   - `multiprocessing` para verdadero paralelismo
   - Extensiones en C/Cython

## Mejoras Futuras

1. **Usar multiprocessing** en lugar de threading para evitar el GIL
2. **Ajustar tamaño de bloques** dinámicamente según hardware
3. **Implementar pool de workers** para reutilizar hilos
4. **Añadir profiling** para identificar cuellos de botella
5. **Considerar GPU computing** con CUDA/OpenCL para mayor paralelismo

## Referencias

- [NumPy Documentation](https://numpy.org/doc/)
- [Python Threading](https://docs.python.org/3/library/threading.html)
- [SIMD Programming](https://en.wikipedia.org/wiki/SIMD)
- [SMP Architecture](https://en.wikipedia.org/wiki/Symmetric_multiprocessing)

## Autor

Implementación realizada para el curso de Infraestructuras Paralelas y Distribuidas, Universidad del Valle.

## Licencia

Este proyecto es de uso académico para el curso 750023C.
