Genetic Algorithm Library
Esta librería implementa algoritmos genéticos para optimizar cualquier función objetivo.
Instalación
bashpip install genetic_algorithm
O instalar desde la fuente:
cd genetic_algorithm
pip install -e .
Uso básico
pythonimport numpy as np
from genetic_algorithm import algoritmo_genetico

# Definir una función objetivo (por ejemplo, optimizar x^2)
def funcion_objetivo(x):
    return x[0]**2

# Configurar y ejecutar el algoritmo genético
resultado = algoritmo_genetico(
    funcion_objetivo=funcion_objetivo,
    dimensiones=1,
    limites=[(-10, 10)],  # Límites para cada variable
    tamano_poblacion=50,
    max_generaciones=100,
    minimizar=True,  # True para minimizar, False para maximizar
    verbose=True
)

# Mostrar resultados
print(f"Mejor solución encontrada: {resultado['mejor_solucion']}")
print(f"Valor de la función objetivo: {resultado['mejor_valor']}")
print(f"Generaciones utilizadas: {resultado['generaciones']}")
Funciones disponibles

crear_individuo: Crea un individuo aleatorio dentro de los límites especificados
crear_poblacion: Genera una población inicial de individuos
evaluar_poblacion: Evalúa la aptitud de cada individuo en la población
seleccion_torneo: Selecciona individuos mediante torneos
cruce: Realiza el cruce entre dos individuos
mutacion: Aplica mutación a un individuo
algoritmo_genetico: Función principal que ejecuta el algoritmo genético completo

Ejemplos
Optimización de funciones de una variable
python# Optimizar f(x) = x^3
def funcion_x_cubo(x):
    return x[0]**3

resultado = algoritmo_genetico(
    funcion_objetivo=funcion_x_cubo,
    dimensiones=1,
    limites=[(-10, 10)],
    minimizar=True
)
Optimización de funciones multivariable
python# Optimizar f(x,y) = x^2 + y^2
def funcion_suma_cuadrados(variables):
    x, y = variables
    return x**2 + y**2

resultado = algoritmo_genetico(
    funcion_objetivo=funcion_suma_cuadrados,
    dimensiones=2,
    limites=[(-10, 10), (-10, 10)],
    tamano_poblacion=100,
    max_generaciones=200,
    minimizar=True
)
Parámetros avanzados
El algoritmo genético permite ajustar diversos parámetros:

tamano_poblacion: Número de individuos en la población
max_generaciones: Número máximo de generaciones
prob_cruce: Probabilidad de cruce (valor entre 0 y 1)
prob_mutacion: Probabilidad de mutación (valor entre 0 y 1)
minimizar: Si es True, minimiza la función; si es False, maximiza
tolerancia: Tolerancia para convergencia
generaciones_sin_mejora: Número de generaciones sin mejora para detener
verbose: Si es True, muestra información del progreso

Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios importantes antes de enviar un pull request.