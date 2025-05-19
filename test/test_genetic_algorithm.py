import pytest
import numpy as np
from genetic_algorithm import (
    crear_individuo,
    crear_poblacion,
    evaluar_poblacion,
    seleccion_torneo,
    cruce,
    mutacion,
    algoritmo_genetico
)

# Prueba para crear_individuo
def test_crear_individuo():
    dimensiones = 3
    limites = [(-5, 5), (-10, 10), (0, 1)]
    individuo = crear_individuo(dimensiones, limites)
    
    assert len(individuo) == dimensiones
    for i, (min_val, max_val) in enumerate(limites):
        assert min_val <= individuo[i] <= max_val

# Prueba para crear_poblacion
def test_crear_poblacion():
    tamano_poblacion = 10
    dimensiones = 2
    limites = [(-5, 5), (-10, 10)]
    
    poblacion = crear_poblacion(tamano_poblacion, dimensiones, limites)
    
    assert len(poblacion) == tamano_poblacion
    for individuo in poblacion:
        assert len(individuo) == dimensiones
        for i, (min_val, max_val) in enumerate(limites):
            assert min_val <= individuo[i] <= max_val

# Prueba para evaluar_poblacion
def test_evaluar_poblacion():
    # Función objetivo de prueba: suma de cuadrados
    def funcion_objetivo(x):
        return sum(xi**2 for xi in x)
    
    poblacion = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0])
    ]
    
    # Minimización
    aptitudes = evaluar_poblacion(poblacion, funcion_objetivo, minimizar=True)
    assert len(aptitudes) == 2
    assert aptitudes[0] == 5.0  # 1² + 2² = 5
    assert aptitudes[1] == 25.0  # 3² + 4² = 25
    
    # Maximización
    aptitudes = evaluar_poblacion(poblacion, funcion_objetivo, minimizar=False)
    assert aptitudes[0] == -5.0
    assert aptitudes[1] == -25.0

# Prueba para algoritmo_genetico
def test_algoritmo_genetico():
    # Función objetivo simple: f(x) = x²
    def funcion_cuadrado(x):
        return x[0]**2
    
    resultado = algoritmo_genetico(
        funcion_objetivo=funcion_cuadrado,
        dimensiones=1,
        limites=[(-10, 10)],
        tamano_poblacion=20,
        max_generaciones=50,
        minimizar=True
    )
    
    # Verificar que el resultado está cerca de cero (mínimo de x²)
    assert abs(resultado['mejor_solucion'][0]) < 0.5
    assert resultado['mejor_valor'] < 0.5
    assert 'generaciones' in resultado
    assert 'historial_aptitudes' in resultado
    
    # Prueba para maximización
    def funcion_para_maximizar(x):
        return -(x[0] - 3)**2 + 9  # Máximo en x=3 con valor 9
    
    resultado = algoritmo_genetico(
        funcion_objetivo=funcion_para_maximizar,
        dimensiones=1,
        limites=[(-10, 10)],
        tamano_poblacion=20,
        max_generaciones=50,
        minimizar=False
    )
    
    # Verificar que el resultado está cerca de 3
    assert abs(resultado['mejor_solucion'][0] - 3) < 0.5
    assert abs(resultado['mejor_valor'] - 9) < 0.5