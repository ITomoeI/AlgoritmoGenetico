# Librería de Algoritmos Genéticos para Optimización

[![PyPI version](https://img.shields.io/badge/testpypi-v0.1.0-blue)](https://test.pypi.org/project/algoritmo-genetico-itomoei/)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-green.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Una implementación robusta y flexible de algoritmos genéticos para optimización de funciones matemáticas en Python.

## 📋 Tabla de Contenidos

- [Introducción](#introducción)
- [Conceptos Básicos](#conceptos-básicos)
- [Instalación](#instalación)
- [Estructura de la Librería](#estructura-de-la-librería)
- [Guía de Uso](#guía-de-uso)
- [Ejemplos](#ejemplos)
- [API de Referencia](#api-de-referencia)
- [Ajuste de Parámetros](#ajuste-de-parámetros)
- [Casos de Uso Avanzados](#casos-de-uso-avanzados)
- [Visualizaciones](#visualizaciones)
- [Contribuir](#contribuir)
- [Autores](#autores)
- [Licencia](#licencia)

## 🧬 Introducción

Los algoritmos genéticos (AG) son métodos de optimización inspirados en la selección natural y la genética. Esta librería proporciona una implementación completa para encontrar soluciones óptimas a problemas matemáticos complejos.

### ¿Qué problemas puede resolver?

- Encontrar mínimos o máximos de funciones matemáticas
- Optimizar funciones con una o múltiples variables
- Resolver problemas donde las técnicas de cálculo tradicionales fallan
- Explorar espacios de búsqueda complejos con múltiples óptimos locales

## 🔍 Conceptos Básicos

### Componentes clave del algoritmo genético:

1. **Cromosoma/Individuo**: Solución potencial al problema
2. **Población**: Conjunto de individuos
3. **Función Objetivo**: Evalúa la calidad de una solución
4. **Selección**: Proceso de elegir individuos para reproducción
5. **Cruce**: Combinación de material genético entre individuos
6. **Mutación**: Alteración aleatoria de genes
7. **Convergencia**: Proceso por el cual la población evoluciona hacia una solución óptima

### Flujo del algoritmo:

1. Inicializar población aleatoria
2. Evaluar cada individuo mediante la función objetivo
3. Seleccionar individuos para reproducción
4. Aplicar operadores genéticos (cruce y mutación)
5. Reemplazar la población antigua con la nueva
6. Repetir hasta alcanzar criterio de parada

## ⚙️ Instalación

### Desde TestPyPI

```bash
# Instalar la librería
pip install --index-url https://test.pypi.org/simple/ --no-deps algoritmo_genetico_itomoei

# Instalar dependencias
pip install numpy
```

### Desde GitHub

```bash
# Clonar el repositorio
git clone https://github.com/ITomoeI/AlgoritmoGenetico.git

# Entrar al directorio
cd AlgoritmoGenetico

# Instalar en modo desarrollo
pip install -e .
```

### Requisitos
- Python 3.6+
- NumPy

## 🏗️ Estructura de la Librería

La librería se organiza de la siguiente manera:

```
genetic_algorithm/
├── __init__.py           # Expone las funciones principales
└── functions.py          # Implementación de las funciones del algoritmo
```

### Componentes principales:

- `crear_individuo`: Genera una solución candidata aleatoria
- `crear_poblacion`: Inicializa una población de individuos
- `evaluar_poblacion`: Calcula la aptitud de cada individuo
- `seleccion_torneo`: Selecciona individuos mediante torneos
- `cruce`: Combina genes de dos padres
- `mutacion`: Altera aleatoriamente los genes
- `algoritmo_genetico`: Función principal que orquesta todo el proceso

## 📘 Guía de Uso

### Uso básico

```python
from genetic_algorithm import algoritmo_genetico

# Definir la función a optimizar
def funcion_objetivo(x):
    return x[0]**2  # Minimizar x^2

# Ejecutar el algoritmo genético
resultado = algoritmo_genetico(
    funcion_objetivo=funcion_objetivo,
    dimensiones=1,  # Una variable
    limites=[(-10, 10)],  # Rango de búsqueda
    tamano_poblacion=50,
    max_generaciones=100,
    verbose=True  # Mostrar progreso
)

# Obtener resultados
print(f"Mejor solución: {resultado['mejor_solucion']}")
print(f"Mejor valor: {resultado['mejor_valor']}")
```

### Modo minimización vs maximización

Por defecto, el algoritmo busca minimizar la función objetivo. Para maximizar:

```python
# Maximizar f(x) = -(x-5)^2 + 25 (máximo en x=5)
def funcion_objetivo(x):
    return -(x[0]-5)**2 + 25

resultado = algoritmo_genetico(
    funcion_objetivo=funcion_objetivo,
    dimensiones=1,
    limites=[(-10, 10)],
    minimizar=False  # Indicamos maximización
)
```

### Optimización multivariable

```python
# Minimizar función de Rosenbrock
def rosenbrock(variables):
    x, y = variables
    return (1 - x)**2 + 100 * (y - x**2)**2

resultado = algoritmo_genetico(
    funcion_objetivo=rosenbrock,
    dimensiones=2,  # Dos variables
    limites=[(-5, 5), (-5, 5)],  # Límites para x e y
    tamano_poblacion=100,
    max_generaciones=200,
    prob_mutacion=0.1
)
```

### En Google Colab

```python
# Instalar desde TestPyPI
!pip install --index-url https://test.pypi.org/simple/ --no-deps algoritmo_genetico_itomoei
!pip install numpy

# Importar y usar
from genetic_algorithm import algoritmo_genetico
```

## 🧪 Ejemplos

### Ejemplo 1: Función de una variable

```python
def funcion_cuadratica(x):
    return x[0]**2

resultado = algoritmo_genetico(
    funcion_objetivo=funcion_cuadratica,
    dimensiones=1,
    limites=[(-10, 10)],
    tamano_poblacion=50,
    max_generaciones=50
)
```

### Ejemplo 2: Maximización

```python
def funcion_para_maximizar(x):
    return -(x[0] - 5)**2 + 25  # Máximo en x=5 con valor 25

resultado = algoritmo_genetico(
    funcion_objetivo=funcion_para_maximizar,
    dimensiones=1,
    limites=[(-10, 20)],
    minimizar=False
)
```

### Ejemplo 3: Función compleja

```python
def funcion_compleja(x):
    return x[0]**4 - 16*x[0]**2 + 5*x[0]  # Función con múltiples óptimos locales

resultado = algoritmo_genetico(
    funcion_objetivo=funcion_compleja,
    dimensiones=1,
    limites=[(-10, 10)],
    tamano_poblacion=100,
    max_generaciones=200
)
```

### Ejemplo 4: Multivariable

```python
def suma_cuadrados(variables):
    return sum(x**2 for x in variables)

resultado = algoritmo_genetico(
    funcion_objetivo=suma_cuadrados,
    dimensiones=5,  # 5 variables
    limites=[(-10, 10)]*5,  # Mismo límite para cada variable
    tamano_poblacion=200,
    max_generaciones=200
)
```

## 📖 API de Referencia

### algoritmo_genetico

```python
def algoritmo_genetico(
    funcion_objetivo, 
    dimensiones, 
    limites, 
    tamano_poblacion=100, 
    max_generaciones=100, 
    prob_cruce=0.8, 
    prob_mutacion=0.1, 
    minimizar=True, 
    tolerancia=1e-6, 
    generaciones_sin_mejora=20, 
    verbose=False
)
```

#### Parámetros:

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `funcion_objetivo` | Callable | Función a optimizar. Debe aceptar un array numpy de tamaño `dimensiones` |
| `dimensiones` | int | Número de variables de la función |
| `limites` | List[Tuple[float, float]] | Lista de tuplas (min, max) para cada dimensión |
| `tamano_poblacion` | int | Número de individuos en la población |
| `max_generaciones` | int | Número máximo de generaciones |
| `prob_cruce` | float | Probabilidad de cruce (entre 0 y 1) |
| `prob_mutacion` | float | Probabilidad de mutación (entre 0 y 1) |
| `minimizar` | bool | Si es True, minimiza la función; si es False, maximiza |
| `tolerancia` | float | Tolerancia para convergencia |
| `generaciones_sin_mejora` | int | Número de generaciones sin mejora para detener |
| `verbose` | bool | Si es True, muestra información del progreso |

#### Retorno:

Un diccionario con:
- `mejor_solucion`: Array con la mejor solución encontrada
- `mejor_valor`: Valor de la función objetivo para la mejor solución
- `generaciones`: Número de generaciones ejecutadas
- `historial_aptitudes`: Lista con los mejores valores en cada generación
- `convergencia`: Boolean indicando si el algoritmo convergió

### Funciones auxiliares

```python
crear_individuo(dimensiones, limites)
crear_poblacion(tamano_poblacion, dimensiones, limites)
evaluar_poblacion(poblacion, funcion_objetivo, minimizar)
seleccion_torneo(poblacion, aptitudes, tamano_torneo=3)
cruce(padre1, padre2, prob_cruce=0.8)
mutacion(individuo, limites, prob_mutacion=0.1)
```

## ⚓ Ajuste de Parámetros

### Tamaño de población

- **Poblaciones pequeñas** (20-50):
  - ✅ Rápidas de evaluar
  - ✅ Convergen rápido
  - ❌ Menor diversidad genética
  - ❌ Mayor riesgo de óptimos locales

- **Poblaciones grandes** (100-500):
  - ✅ Mayor diversidad genética
  - ✅ Mejor exploración del espacio
  - ❌ Más costosas computacionalmente
  - ❌ Convergencia más lenta

### Probabilidad de cruce

- **Probabilidad alta** (0.8-1.0):
  - Favorece la exploración
  - Útil para problemas complejos

- **Probabilidad baja** (0.2-0.5):
  - Favorece la explotación
  - Útil cuando se está cerca del óptimo

### Probabilidad de mutación

- **Probabilidad alta** (0.1-0.3):
  - Mayor exploración
  - Evita convergencia prematura
  - Útil para escapar de óptimos locales

- **Probabilidad baja** (0.01-0.05):
  - Refinamiento fino de soluciones
  - Mayor estabilidad
  - Útil en las etapas finales

### Criterios de parada

- `max_generaciones`: Límite duro de iteraciones
- `generaciones_sin_mejora`: Detiene cuando no hay progreso
- `tolerancia`: Define qué se considera una mejora significativa

## 🔧 Casos de Uso Avanzados

### Optimización con restricciones

Para problemas con restricciones, modifica la función objetivo para penalizar soluciones inválidas:

```python
def funcion_con_restricciones(x):
    # Función original
    resultado = x[0]**2 + x[1]**2
    
    # Restricción: x[0] + x[1] <= 1
    if x[0] + x[1] > 1:
        # Penalización
        resultado += 1000 * (x[0] + x[1] - 1)**2
    
    return resultado
```

### Optimización de múltiples funciones objetivo

Para problemas multi-objetivo, puedes combinar las funciones con pesos:

```python
def funcion_multi_objetivo(x):
    # Minimizar x^2 y maximizar cos(x) simultáneamente
    f1 = x[0]**2  # Minimizar
    f2 = -np.cos(x[0])  # Minimizar (equivale a maximizar cos(x))
    
    # Ponderación de objetivos (puedes ajustar los pesos)
    return 0.7 * f1 + 0.3 * f2
```

### Optimización de hiperparámetros

Puedes usar algoritmos genéticos para optimizar hiperparámetros de modelos ML:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

def optimizar_hiperparametros(parametros):
    # Convertir a valores enteros/discretos apropiados
    n_estimators = int(parametros[0])
    max_depth = int(parametros[1])
    min_samples_split = int(parametros[2])
    
    # Crear modelo con estos hiperparámetros
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Evaluar con validación cruzada
    scores = cross_val_score(modelo, X, y, cv=5)
    
    # Retornar el negativo (porque estamos minimizando)
    return -scores.mean()

# Límites para los hiperparámetros
limites = [
    (10, 200),    # n_estimators
    (1, 20),      # max_depth
    (2, 10)       # min_samples_split
]

resultado = algoritmo_genetico(
    funcion_objetivo=optimizar_hiperparametros,
    dimensiones=3,
    limites=limites,
    tamano_poblacion=30,
    max_generaciones=20
)
```

## 📊 Visualizaciones

### Visualización 2D simple

```python
import matplotlib.pyplot as plt
import numpy as np

# Luego de ejecutar el algoritmo:
def graficar_funcion_1d(funcion, limites, resultado):
    x = np.linspace(limites[0][0], limites[0][1], 1000)
    y = [funcion(np.array([xi])) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Función objetivo')
    plt.scatter(resultado['mejor_solucion'][0], resultado['mejor_valor'], 
                color='red', s=100, marker='*', label='Mejor solución')
    
    plt.title('Optimización mediante Algoritmo Genético')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()
```

### Visualización de convergencia

```python
def graficar_convergencia(historial_aptitudes):
    plt.figure(figsize=(10, 6))
    plt.plot(historial_aptitudes)
    plt.title('Convergencia del Algoritmo Genético')
    plt.xlabel('Generación')
    plt.ylabel('Mejor aptitud')
    plt.grid(True)
    plt.show()
```

### Visualización 3D para funciones multivariable

```python
from mpl_toolkits.mplot3d import Axes3D

def graficar_funcion_2d(funcion, limites, resultado):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear malla
    x = np.linspace(limites[0][0], limites[0][1], 100)
    y = np.linspace(limites[1][0], limites[1][1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Calcular valores de Z
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = funcion(np.array([X[j, i], Y[j, i]]))
    
    # Graficar superficie
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    # Marcar la mejor solución
    mejor_x, mejor_y = resultado['mejor_solucion']
    mejor_z = resultado['mejor_valor']
    ax.scatter(mejor_x, mejor_y, mejor_z, color='red', s=100, marker='*', 
               label='Mejor solución')
    
    # Configuración del gráfico
    ax.set_title('Superficie de la función objetivo')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.legend()
    
    # Agregar barra de color
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Pasos para contribuir:

1. Fork del repositorio
2. Crear una rama para tu funcionalidad (`git checkout -b nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin nueva-funcionalidad`)
5. Abrir un Pull Request

### Ideas para contribuciones:

- Implementar más operadores de selección (ruleta, elitista)
- Añadir más operadores de cruce (varios puntos, uniforme)
- Optimización para problemas con variables discretas
- Paralelización para poblaciones grandes
- Métodos adaptativos para probabilidades de cruce y mutación

## 👤 Autores

- **Bryan Rojas** - [GitHub](https://github.com/ITomoeI) - brrojas.h14@gmail.com
- **Juan Ayala**

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.

---

## 📚 Referencias

- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.
- Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.
