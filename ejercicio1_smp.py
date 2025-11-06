"""
Ejercicio 1: Arquitectura SMP (Symmetric Multiprocessing)

Objetivo: Simular un sistema SMP con hilos en Python para calcular la suma de una matriz grande.

Este programa implementa:
1. Una matriz de 1000x1000 con números aleatorios
2. División de la matriz en bloques de 100x100 (100 bloques en total)
3. Procesamiento paralelo usando threading
4. Comparación de tiempos: versión secuencial vs paralela

Factores considerados en la solución:
- Threading en Python para simular arquitectura SMP
- Uso de Lock para evitar condiciones de carrera al sumar resultados
- Medición precisa de tiempos de ejecución
- División eficiente de trabajo entre hilos
"""

import numpy as np
import threading
import time
from typing import List


class MatrixSumSMP:
    """Clase para calcular la suma de una matriz usando arquitectura SMP"""

    def __init__(self, matrix: np.ndarray, block_size: int = 100):
        """
        Inicializa la clase con la matriz y tamaño de bloque

        Args:
            matrix: Matriz NumPy a procesar
            block_size: Tamaño de cada bloque (por defecto 100x100)
        """
        self.matrix = matrix
        self.block_size = block_size
        self.total_sum = 0
        self.lock = threading.Lock()  # Para evitar race conditions
        self.matrix_size = matrix.shape[0]

    def calculate_block_sum(self, start_row: int, start_col: int) -> None:
        """
        Calcula la suma de un bloque de la matriz y la agrega al total

        Args:
            start_row: Fila inicial del bloque
            start_col: Columna inicial del bloque
        """
        # Determinar los límites del bloque
        end_row = min(start_row + self.block_size, self.matrix_size)
        end_col = min(start_col + self.block_size, self.matrix_size)

        # Extraer el bloque
        block = self.matrix[start_row:end_row, start_col:end_col]

        # Calcular suma del bloque
        block_sum = np.sum(block)

        # Agregar al total de forma thread-safe
        with self.lock:
            self.total_sum += block_sum

    def parallel_sum(self) -> float:
        """
        Calcula la suma total usando múltiples hilos (SMP)

        Returns:
            Tiempo de ejecución en segundos
        """
        start_time = time.time()

        threads: List[threading.Thread] = []

        # Crear hilos para cada bloque
        for i in range(0, self.matrix_size, self.block_size):
            for j in range(0, self.matrix_size, self.block_size):
                thread = threading.Thread(
                    target=self.calculate_block_sum,
                    args=(i, j)
                )
                threads.append(thread)
                thread.start()

        # Esperar a que todos los hilos terminen
        for thread in threads:
            thread.join()

        end_time = time.time()
        execution_time = end_time - start_time

        return execution_time

    def sequential_sum(self) -> tuple:
        """
        Calcula la suma total de forma secuencial

        Returns:
            Tupla con (suma_total, tiempo_de_ejecución)
        """
        start_time = time.time()

        total = 0
        for i in range(0, self.matrix_size, self.block_size):
            for j in range(0, self.matrix_size, self.block_size):
                end_row = min(i + self.block_size, self.matrix_size)
                end_col = min(j + self.block_size, self.matrix_size)
                block = self.matrix[i:end_row, j:end_col]
                total += np.sum(block)

        end_time = time.time()
        execution_time = end_time - start_time

        return total, execution_time


def main():
    """Función principal para ejecutar el ejercicio"""
    print("=" * 70)
    print("EJERCICIO 1: ARQUITECTURA SMP (Symmetric Multiprocessing)")
    print("=" * 70)
    print()

    # Crear matriz de 1000x1000 con números aleatorios
    print("1. Creando matriz de 1000x1000 con números aleatorios...")
    np.random.seed(42)  # Para reproducibilidad
    matrix = np.random.rand(1000, 1000)
    print(f"   Matriz creada: {matrix.shape}")
    print(f"   Rango de valores: [{matrix.min():.4f}, {matrix.max():.4f}]")
    print()

    # Crear instancia de la clase
    smp_calculator = MatrixSumSMP(matrix, block_size=100)

    # Cálculo secuencial
    print("2. Calculando suma de forma SECUENCIAL...")
    seq_sum, seq_time = smp_calculator.sequential_sum()
    print(f"   Suma total (secuencial): {seq_sum:.6f}")
    print(f"   Tiempo de ejecución: {seq_time:.6f} segundos")
    print()

    # Reiniciar para cálculo paralelo
    smp_calculator.total_sum = 0

    # Cálculo paralelo
    print("3. Calculando suma de forma PARALELA (SMP con hilos)...")
    num_blocks = (1000 // 100) * (1000 // 100)
    print(f"   Número de bloques: {num_blocks} (bloques de 100x100)")
    print(f"   Número de hilos: {num_blocks}")
    parallel_time = smp_calculator.parallel_sum()
    print(f"   Suma total (paralela): {smp_calculator.total_sum:.6f}")
    print(f"   Tiempo de ejecución: {parallel_time:.6f} segundos")
    print()

    # Comparación de resultados
    print("4. ANÁLISIS DE RESULTADOS:")
    print("-" * 70)
    print(f"   Tiempo secuencial:  {seq_time:.6f} segundos")
    print(f"   Tiempo paralelo:    {parallel_time:.6f} segundos")
    print(f"   Speedup:            {seq_time/parallel_time:.2f}x")
    print(f"   Mejora porcentual:  {((seq_time-parallel_time)/seq_time)*100:.2f}%")
    print()

    # Verificación de correctitud
    difference = abs(seq_sum - smp_calculator.total_sum)
    print(f"   Diferencia entre resultados: {difference:.10f}")
    print(f"   Resultados correctos: {'✓ SÍ' if difference < 1e-6 else '✗ NO'}")
    print()

    # Explicación detallada
    print("5. EXPLICACIÓN DE LA SOLUCIÓN:")
    print("-" * 70)
    print("""
   FACTORES CONSIDERADOS:

   a) Arquitectura SMP simulada:
      - Se utilizan hilos de Python (threading) para simular múltiples
        procesadores trabajando en memoria compartida
      - Cada hilo procesa un bloque independiente de 100x100 elementos

   b) Sincronización:
      - Se usa un Lock (threading.Lock) para evitar condiciones de carrera
      - Solo se sincroniza al agregar el resultado parcial al total
      - Esto minimiza el overhead de sincronización

   c) División del trabajo:
      - Matriz de 1000x1000 dividida en 100 bloques de 100x100
      - Cada bloque es procesado por un hilo independiente
      - División equitativa del trabajo entre hilos

   d) Limitaciones de Python:
      - El GIL (Global Interpreter Lock) puede limitar el paralelismo real
      - Para operaciones CPU-bound, multiprocessing podría ser más eficiente
      - Sin embargo, threading muestra el concepto de SMP

   e) Medición de rendimiento:
      - time.time() para medir tiempos de ejecución
      - Comparación directa entre versión secuencial y paralela
      - Cálculo de speedup y mejora porcentual
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
