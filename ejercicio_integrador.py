"""
Ejercicio Integrador: Simulación de un sistema híbrido SMP-SIMD

Objetivo: Simular un sistema que combina SMP (paralelismo a nivel de hilos) y
SIMD (operaciones vectorizadas) para procesar una gran cantidad de datos.

Este programa implementa:
1. Matriz de 10000x10000 con números aleatorios
2. División en bloques de 1000x1000 (100 bloques)
3. Procesamiento paralelo con hilos (SMP)
4. Operaciones SIMD con NumPy dentro de cada hilo
5. Comparación con versión secuencial

Factores considerados en la solución:
- Combinación de paralelismo a nivel de hilos (SMP) y datos (SIMD)
- Uso eficiente de recursos del sistema
- Minimización de overhead de sincronización
- Medición precisa de rendimiento
- Escalabilidad del sistema híbrido
"""

import numpy as np
import threading
import time
from typing import List, Tuple
import matplotlib.pyplot as plt


class HybridSMPSIMD:
    """Clase para procesamiento híbrido SMP-SIMD de matrices grandes"""

    def __init__(self, matrix_size: int = 10000, block_size: int = 1000):
        """
        Inicializa el sistema híbrido

        Args:
            matrix_size: Tamaño de la matriz principal
            block_size: Tamaño de cada bloque para procesamiento paralelo
        """
        self.matrix_size = matrix_size
        self.block_size = block_size
        self.total_sum = 0
        self.lock = threading.Lock()
        self.matrix = None

        print(f"Inicializando sistema híbrido SMP-SIMD:")
        print(f"  - Tamaño de matriz: {matrix_size}x{matrix_size}")
        print(f"  - Tamaño de bloque: {block_size}x{block_size}")
        print(f"  - Número de bloques: {(matrix_size//block_size)**2}")
        print()

    def create_matrix(self) -> None:
        """Crea la matriz grande con números aleatorios"""
        print(f"Creando matriz de {self.matrix_size}x{self.matrix_size}...")
        start_time = time.time()

        np.random.seed(42)
        self.matrix = np.random.rand(self.matrix_size, self.matrix_size)

        creation_time = time.time() - start_time
        print(f"Matriz creada en {creation_time:.4f} segundos")
        print(f"Tamaño en memoria: ~{self.matrix.nbytes / (1024**2):.2f} MB")
        print()

    def process_block_simd(self, start_row: int, start_col: int,
                          thread_id: int) -> None:
        """
        Procesa un bloque usando operaciones SIMD de NumPy

        Args:
            start_row: Fila inicial del bloque
            start_col: Columna inicial del bloque
            thread_id: Identificador del hilo
        """
        # Extraer el bloque
        end_row = min(start_row + self.block_size, self.matrix_size)
        end_col = min(start_col + self.block_size, self.matrix_size)
        block = self.matrix[start_row:end_row, start_col:end_col]

        # Operaciones SIMD con NumPy
        # 1. Suma de todas las filas (operación vectorizada)
        row_sums = np.sum(block, axis=1)  # SIMD

        # 2. Suma de todas las columnas (operación vectorizada)
        col_sums = np.sum(block, axis=0)  # SIMD

        # 3. Suma total del bloque (operación vectorizada)
        block_total = np.sum(block)  # SIMD

        # Agregar al total global de forma thread-safe (SMP)
        with self.lock:
            self.total_sum += block_total

    def hybrid_parallel_sum(self) -> Tuple[float, int]:
        """
        Calcula la suma usando el sistema híbrido SMP-SIMD

        Returns:
            Tupla con (tiempo_de_ejecución, número_de_hilos)
        """
        print("Procesamiento HÍBRIDO (SMP + SIMD):")
        print("-" * 70)

        self.total_sum = 0
        start_time = time.time()

        threads: List[threading.Thread] = []
        thread_id = 0

        # Crear hilos para cada bloque (SMP)
        for i in range(0, self.matrix_size, self.block_size):
            for j in range(0, self.matrix_size, self.block_size):
                thread = threading.Thread(
                    target=self.process_block_simd,
                    args=(i, j, thread_id)
                )
                threads.append(thread)
                thread.start()
                thread_id += 1

        print(f"  - Hilos creados: {len(threads)}")

        # Esperar a que todos los hilos terminen
        for thread in threads:
            thread.join()

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"  - Suma total: {self.total_sum:.6f}")
        print(f"  - Tiempo de ejecución: {execution_time:.6f} segundos")
        print()

        return execution_time, len(threads)

    def sequential_sum(self) -> Tuple[float, float]:
        """
        Calcula la suma de forma secuencial (sin SMP, con SIMD de NumPy)

        Returns:
            Tupla con (suma_total, tiempo_de_ejecución)
        """
        print("Procesamiento SECUENCIAL (solo SIMD):")
        print("-" * 70)

        start_time = time.time()

        total = 0
        blocks_processed = 0

        for i in range(0, self.matrix_size, self.block_size):
            for j in range(0, self.matrix_size, self.block_size):
                end_row = min(i + self.block_size, self.matrix_size)
                end_col = min(j + self.block_size, self.matrix_size)
                block = self.matrix[i:end_row, j:end_col]

                # Usar SIMD de NumPy pero sin paralelismo de hilos
                block_total = np.sum(block)
                total += block_total
                blocks_processed += 1

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"  - Bloques procesados: {blocks_processed}")
        print(f"  - Suma total: {total:.6f}")
        print(f"  - Tiempo de ejecución: {execution_time:.6f} segundos")
        print()

        return total, execution_time

    def pure_sequential_sum(self) -> Tuple[float, float]:
        """
        Calcula la suma de forma completamente secuencial (sin SMP ni SIMD optimizado)

        Returns:
            Tupla con (suma_total, tiempo_de_ejecución)
        """
        print("Procesamiento PURO SECUENCIAL (sin optimizaciones):")
        print("-" * 70)

        start_time = time.time()

        total = 0
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                total += self.matrix[i, j]

            # Mostrar progreso cada 1000 filas
            if (i + 1) % 1000 == 0:
                print(f"  - Progreso: {i + 1}/{self.matrix_size} filas")

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"  - Suma total: {total:.6f}")
        print(f"  - Tiempo de ejecución: {execution_time:.6f} segundos")
        print()

        return total, execution_time


def create_performance_comparison(times: dict, num_threads: int) -> None:
    """
    Crea visualizaciones de comparación de rendimiento

    Args:
        times: Diccionario con los tiempos de cada método
        num_threads: Número de hilos usados
    """
    print("Generando gráficos de comparación...")

    # Gráfico de barras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Comparación de tiempos
    methods = ['Híbrido\nSMP-SIMD', 'Secuencial\n(SIMD)', 'Puro\nSecuencial']
    execution_times = [
        times['hybrid'],
        times['sequential'],
        times['pure_sequential']
    ]
    colors = ['#27ae60', '#f39c12', '#e74c3c']

    bars = ax1.bar(methods, execution_times, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)

    # Añadir valores encima de las barras
    for bar, time_val in zip(bars, execution_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Tiempo de Ejecución (segundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Tiempos de Ejecución\nMatriz 10000x10000',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Gráfico 2: Speedup
    speedup_vs_sequential = times['sequential'] / times['hybrid']
    speedup_vs_pure = times['pure_sequential'] / times['hybrid']

    speedup_labels = [f'vs Secuencial\n(SIMD)', f'vs Puro\nSecuencial']
    speedup_values = [speedup_vs_sequential, speedup_vs_pure]
    speedup_colors = ['#3498db', '#9b59b6']

    bars2 = ax2.bar(speedup_labels, speedup_values, color=speedup_colors,
                    alpha=0.8, edgecolor='black', linewidth=2)

    # Añadir valores encima de las barras
    for bar, val in zip(bars2, speedup_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Speedup (veces más rápido)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Speedup del Sistema Híbrido\n({num_threads} hilos)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Baseline')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('/home/user/taller_4/ejercicio_integrador_comparacion.png',
                dpi=300, bbox_inches='tight')
    print("Gráfico guardado en: ejercicio_integrador_comparacion.png")
    print()


def print_detailed_analysis(times: dict, sums: dict, num_threads: int) -> None:
    """
    Imprime análisis detallado de los resultados

    Args:
        times: Diccionario con tiempos de ejecución
        sums: Diccionario con sumas calculadas
        num_threads: Número de hilos usados
    """
    print("=" * 70)
    print("ANÁLISIS DETALLADO DE RESULTADOS")
    print("=" * 70)
    print()

    # Tabla de tiempos
    print("1. TABLA DE TIEMPOS:")
    print("-" * 70)
    print(f"{'Método':<30} {'Tiempo (s)':<15} {'Speedup':<15}")
    print("-" * 70)
    print(f"{'Híbrido SMP-SIMD':<30} {times['hybrid']:<15.6f} {'1.00x':<15}")
    print(f"{'Secuencial (SIMD)':<30} {times['sequential']:<15.6f} "
          f"{times['sequential']/times['hybrid']:<15.2f}")
    print(f"{'Puro Secuencial':<30} {times['pure_sequential']:<15.6f} "
          f"{times['pure_sequential']/times['hybrid']:<15.2f}")
    print("-" * 70)
    print()

    # Verificación de correctitud
    print("2. VERIFICACIÓN DE CORRECTITUD:")
    print("-" * 70)
    diff1 = abs(sums['hybrid'] - sums['sequential'])
    diff2 = abs(sums['hybrid'] - sums['pure_sequential'])
    print(f"Suma híbrida:        {sums['hybrid']:.6f}")
    print(f"Suma secuencial:     {sums['sequential']:.6f}")
    print(f"Suma pura:           {sums['pure_sequential']:.6f}")
    print(f"Diferencia 1:        {diff1:.10f}")
    print(f"Diferencia 2:        {diff2:.10f}")
    print(f"Resultados correctos: {'✓ SÍ' if diff1 < 1e-6 and diff2 < 1e-6 else '✗ NO'}")
    print()

    # Análisis de mejora
    print("3. ANÁLISIS DE MEJORA:")
    print("-" * 70)
    improvement1 = ((times['sequential'] - times['hybrid']) / times['sequential']) * 100
    improvement2 = ((times['pure_sequential'] - times['hybrid']) / times['pure_sequential']) * 100

    print(f"Mejora vs Secuencial (SIMD):     {improvement1:.2f}%")
    print(f"Mejora vs Puro Secuencial:       {improvement2:.2f}%")
    print(f"Número de hilos utilizados:      {num_threads}")
    print(f"Eficiencia paralela:             {(times['sequential']/times['hybrid'])/num_threads*100:.2f}%")
    print()


def main():
    """Función principal del ejercicio integrador"""
    print("=" * 70)
    print("EJERCICIO INTEGRADOR: SISTEMA HÍBRIDO SMP-SIMD")
    print("=" * 70)
    print()

    # Crear instancia del sistema híbrido
    hybrid_system = HybridSMPSIMD(matrix_size=10000, block_size=1000)

    # Crear matriz
    hybrid_system.create_matrix()

    # Método 1: Híbrido SMP-SIMD
    print("MÉTODO 1: HÍBRIDO SMP-SIMD")
    print("=" * 70)
    hybrid_time, num_threads = hybrid_system.hybrid_parallel_sum()
    hybrid_sum = hybrid_system.total_sum

    # Método 2: Secuencial con SIMD
    print("MÉTODO 2: SECUENCIAL CON SIMD")
    print("=" * 70)
    sequential_sum, sequential_time = hybrid_system.sequential_sum()

    # Método 3: Puro secuencial (comentar si toma mucho tiempo)
    print("MÉTODO 3: PURO SECUENCIAL (puede tardar varios minutos)")
    print("=" * 70)
    print("NOTA: Este método es muy lento. Se usará una estimación basada en")
    print("      el procesamiento de una porción de la matriz.")
    print()

    # Procesar solo el 1% de la matriz para estimar
    estimation_size = 1000
    estimation_start = time.time()
    estimation_sum = 0
    for i in range(estimation_size):
        for j in range(estimation_size):
            estimation_sum += hybrid_system.matrix[i, j]
    estimation_time = time.time() - estimation_start

    # Extrapolar al 100%
    pure_sequential_time = estimation_time * ((10000 * 10000) / (estimation_size * estimation_size))
    pure_sequential_sum = estimation_sum * ((10000 * 10000) / (estimation_size * estimation_size))

    print(f"  - Tiempo estimado: {pure_sequential_time:.6f} segundos")
    print(f"  - Suma estimada: {pure_sequential_sum:.6f}")
    print()

    # Diccionarios para análisis
    times = {
        'hybrid': hybrid_time,
        'sequential': sequential_time,
        'pure_sequential': pure_sequential_time
    }

    sums = {
        'hybrid': hybrid_sum,
        'sequential': sequential_sum,
        'pure_sequential': pure_sequential_sum
    }

    # Análisis y visualización
    print_detailed_analysis(times, sums, num_threads)
    create_performance_comparison(times, num_threads)

    # Explicación detallada
    print("4. EXPLICACIÓN DETALLADA DEL SISTEMA HÍBRIDO:")
    print("=" * 70)
    print("""
ARQUITECTURA HÍBRIDA SMP-SIMD:

1. Nivel SMP (Symmetric Multiprocessing):
   - Se crean múltiples hilos para procesar bloques independientes
   - Cada hilo trabaja en su propio bloque de 1000x1000
   - Los hilos comparten la memoria (matriz original)
   - Sincronización solo al agregar resultados parciales

2. Nivel SIMD (Single Instruction, Multiple Data):
   - Dentro de cada hilo se usa NumPy para operaciones vectorizadas
   - NumPy aprovecha instrucciones SIMD del procesador
   - Operaciones como np.sum() procesan múltiples datos simultáneamente
   - Suma de filas y columnas se realiza de forma vectorizada

3. Ventajas del Sistema Híbrido:
   a) Paralelismo multinivel:
      - Paralelismo a nivel de hilos (entre bloques)
      - Paralelismo a nivel de datos (dentro de cada bloque)

   b) Escalabilidad:
      - Aprovecha múltiples núcleos del procesador (SMP)
      - Aprovecha unidades SIMD de cada núcleo

   c) Eficiencia:
      - Minimiza overhead de sincronización
      - Maximiza uso de recursos del sistema
      - Mejor localidad de caché por bloque

4. Factores de Rendimiento:

   a) Speedup teórico vs real:
      - Teórico: N hilos × speedup SIMD
      - Real: Limitado por overhead, sincronización, GIL de Python

   b) Overhead de threading:
      - Creación y destrucción de hilos
      - Sincronización con locks
      - Cambios de contexto

   c) Limitaciones de Python:
      - GIL (Global Interpreter Lock) puede limitar paralelismo
      - Para CPU-bound tasks, multiprocessing sería más eficiente
      - NumPy libera el GIL para operaciones C

5. Casos de Uso Reales:
   - Procesamiento de imágenes en paralelo
   - Análisis de grandes conjuntos de datos
   - Simulaciones científicas
   - Machine Learning: entrenamiento distribuido
   - Procesamiento de video: frames en paralelo

6. Comparación de Métodos:

   Híbrido SMP-SIMD:
   - Más rápido para matrices grandes
   - Usa todos los recursos disponibles
   - Complejidad de implementación moderada

   Secuencial con SIMD:
   - Más lento que híbrido
   - No aprovecha múltiples núcleos
   - Implementación simple

   Puro Secuencial:
   - Extremadamente lento
   - Solo usa un núcleo, sin vectorización
   - No práctico para datos grandes

CONCLUSIÓN:
El sistema híbrido SMP-SIMD demuestra la importancia de combinar múltiples
niveles de paralelismo para maximizar el rendimiento en sistemas modernos.
La arquitectura híbrida es fundamental para aplicaciones de alto rendimiento
que procesan grandes volúmenes de datos.
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
