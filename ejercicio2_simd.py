"""
Ejercicio 2: Arquitectura SIMD (Single Instruction, Multiple Data)

Objetivo: Comparar el rendimiento de la multiplicación de matrices utilizando
NumPy (que aprovecha instrucciones SIMD) versus un bucle tradicional en Python.

Este programa implementa:
1. Dos matrices de 1000x1000 con números aleatorios
2. Multiplicación usando NumPy (optimizado con SIMD)
3. Multiplicación usando bucles tradicionales
4. Comparación de tiempos y visualización de resultados

Factores considerados en la solución:
- NumPy utiliza librerías optimizadas (BLAS/LAPACK) con instrucciones SIMD
- Los bucles tradicionales en Python son interpretados y muy lentos
- SIMD permite procesar múltiples datos con una sola instrucción
- Comparación justa entre ambos métodos
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple


class MatrixMultiplicationSIMD:
    """Clase para comparar multiplicación de matrices con y sin SIMD"""

    def __init__(self, size: int = 1000):
        """
        Inicializa las matrices para la multiplicación

        Args:
            size: Tamaño de las matrices (size x size)
        """
        self.size = size
        print(f"Creando matrices de {size}x{size}...")
        np.random.seed(42)  # Para reproducibilidad
        self.matrix_a = np.random.rand(size, size)
        self.matrix_b = np.random.rand(size, size)
        print(f"Matrices creadas exitosamente")
        print()

    def multiply_numpy(self) -> Tuple[np.ndarray, float]:
        """
        Multiplica matrices usando NumPy (SIMD optimizado)

        Returns:
            Tupla con (resultado, tiempo_de_ejecución)
        """
        print("Multiplicando con NumPy (SIMD)...")
        start_time = time.time()

        # NumPy usa operaciones SIMD optimizadas a través de BLAS
        result = np.dot(self.matrix_a, self.matrix_b)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
        return result, execution_time

    def multiply_traditional(self) -> Tuple[np.ndarray, float]:
        """
        Multiplica matrices usando bucles tradicionales (sin SIMD)

        Returns:
            Tupla con (resultado, tiempo_de_ejecución)
        """
        print("Multiplicando con bucles tradicionales...")
        start_time = time.time()

        # Crear matriz resultado
        result = np.zeros((self.size, self.size))

        # Triple bucle anidado - forma tradicional
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    result[i][j] += self.matrix_a[i][k] * self.matrix_b[k][j]

            # Mostrar progreso cada 100 filas
            if (i + 1) % 100 == 0:
                print(f"  Progreso: {i + 1}/{self.size} filas completadas")

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
        return result, execution_time

    def verify_results(self, result1: np.ndarray, result2: np.ndarray) -> bool:
        """
        Verifica que ambos métodos produzcan el mismo resultado

        Args:
            result1: Primera matriz resultado
            result2: Segunda matriz resultado

        Returns:
            True si los resultados son equivalentes
        """
        # Usar np.allclose para comparar con tolerancia
        return np.allclose(result1, result2, rtol=1e-5, atol=1e-8)


def create_comparison_table(numpy_time: float, traditional_time: float) -> None:
    """
    Crea una tabla de comparación de resultados

    Args:
        numpy_time: Tiempo de ejecución de NumPy
        traditional_time: Tiempo de ejecución tradicional
    """
    print("\n" + "=" * 70)
    print("TABLA DE COMPARACIÓN DE RESULTADOS")
    print("=" * 70)
    print(f"{'Método':<25} {'Tiempo (s)':<15} {'Velocidad Relativa':<20}")
    print("-" * 70)
    print(f"{'NumPy (SIMD)':<25} {numpy_time:<15.6f} {'1.00x (baseline)':<20}")
    print(f"{'Bucles Tradicionales':<25} {traditional_time:<15.6f} {f'{traditional_time/numpy_time:.2f}x más lento':<20}")
    print("-" * 70)
    print(f"{'Speedup de SIMD:':<25} {traditional_time/numpy_time:.2f}x")
    print(f"{'Mejora porcentual:':<25} {((traditional_time-numpy_time)/traditional_time)*100:.2f}%")
    print("=" * 70)


def create_comparison_chart(numpy_time: float, traditional_time: float) -> None:
    """
    Crea un gráfico de barras comparando los tiempos

    Args:
        numpy_time: Tiempo de ejecución de NumPy
        traditional_time: Tiempo de ejecución tradicional
    """
    print("\nGenerando gráfico de comparación...")

    methods = ['NumPy\n(SIMD)', 'Bucles\nTradicionales']
    times = [numpy_time, traditional_time]
    colors = ['#2ecc71', '#e74c3c']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Añadir valores encima de las barras
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12, fontweight='bold')
    plt.title('Comparación: NumPy (SIMD) vs Bucles Tradicionales\nMultiplicación de Matrices 1000x1000',
              fontsize=14, fontweight='bold', pad=20)
    plt.yscale('log')  # Escala logarítmica por la gran diferencia
    plt.ylabel('Tiempo de Ejecución (segundos) - Escala Logarítmica', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Añadir texto con el speedup
    speedup = traditional_time / numpy_time
    plt.text(0.5, 0.95, f'Speedup: {speedup:.2f}x',
            transform=plt.gca().transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            ha='center', va='top')

    plt.tight_layout()
    plt.savefig('/home/user/taller_4/ejercicio2_comparacion.png', dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: ejercicio2_comparacion.png")


def main():
    """Función principal para ejecutar el ejercicio"""
    print("=" * 70)
    print("EJERCICIO 2: ARQUITECTURA SIMD (Single Instruction, Multiple Data)")
    print("=" * 70)
    print()

    # Crear instancia
    simd_test = MatrixMultiplicationSIMD(size=1000)

    # Multiplicación con NumPy (SIMD)
    print("1. MULTIPLICACIÓN CON NUMPY (SIMD):")
    print("-" * 70)
    numpy_result, numpy_time = simd_test.multiply_numpy()
    print()

    # Multiplicación tradicional
    print("2. MULTIPLICACIÓN CON BUCLES TRADICIONALES:")
    print("-" * 70)
    traditional_result, traditional_time = simd_test.multiply_traditional()
    print()

    # Verificar que los resultados son correctos
    print("3. VERIFICACIÓN DE RESULTADOS:")
    print("-" * 70)
    results_match = simd_test.verify_results(numpy_result, traditional_result)
    print(f"¿Los resultados coinciden? {'✓ SÍ' if results_match else '✗ NO'}")
    if results_match:
        print("Ambos métodos producen el mismo resultado (dentro de la tolerancia numérica)")
    print()

    # Tabla de comparación
    create_comparison_table(numpy_time, traditional_time)
    print()

    # Gráfico de comparación
    create_comparison_chart(numpy_time, traditional_time)
    print()

    # Explicación detallada
    print("4. EXPLICACIÓN DETALLADA:")
    print("=" * 70)
    print("""
ANÁLISIS DE LA ARQUITECTURA SIMD:

1. ¿Qué es SIMD?
   - Single Instruction, Multiple Data (Una Instrucción, Múltiples Datos)
   - Permite procesar múltiples elementos de datos con una sola instrucción
   - Común en procesadores modernos (SSE, AVX, NEON, etc.)

2. NumPy y SIMD:
   - NumPy usa librerías optimizadas como BLAS (Basic Linear Algebra Subprograms)
   - Estas librerías están compiladas con instrucciones SIMD
   - Operan directamente en memoria a nivel de hardware
   - Aprovechan vectorización automática del compilador

3. Bucles Tradicionales en Python:
   - Python es un lenguaje interpretado
   - Cada operación tiene overhead del intérprete
   - No aprovecha instrucciones SIMD
   - Acceso a elementos uno por uno (escalar)

4. Diferencia de Rendimiento:
   - NumPy es órdenes de magnitud más rápido
   - El speedup típico es de 100x a 1000x o más
   - La diferencia aumenta con el tamaño de las matrices

5. Factores Clave del Speedup:
   a) Vectorización: SIMD procesa múltiples datos simultáneamente
   b) Código compilado: vs código interpretado de Python
   c) Optimizaciones del compilador: loop unrolling, prefetching
   d) Cache efficiency: mejor uso de la jerarquía de memoria
   e) Reduced overhead: sin overhead del intérprete de Python

6. Aplicaciones Prácticas:
   - Machine Learning: operaciones con tensores
   - Procesamiento de imágenes: operaciones pixel a pixel
   - Simulaciones científicas: cálculos numéricos intensivos
   - Procesamiento de señales: FFT, filtros digitales

CONCLUSIÓN:
El uso de bibliotecas optimizadas con SIMD como NumPy es fundamental para
aplicaciones de alto rendimiento. La diferencia de rendimiento demuestra la
importancia de la arquitectura SIMD en el procesamiento moderno de datos.
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
