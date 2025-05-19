import numpy as np
import random
from typing import Callable, List, Tuple, Dict, Any, Union, Optional

def crear_individuo(dimensiones: int, limites: List[Tuple[float, float]]) -> np.ndarray:
    """
    Crea un individuo aleatorio dentro de los límites especificados.
    
    Args:
        dimensiones: Número de variables (genes) del individuo.
        limites: Lista de tuplas (min, max) para cada dimensión.
        
    Returns:
        np.ndarray: Un array con los valores del individuo.
    """
    individuo = np.zeros(dimensiones)
    for i in range(dimensiones):
        min_val, max_val = limites[i]
        individuo[i] = random.uniform(min_val, max_val)
    return individuo

def crear_poblacion(tamano_poblacion: int, dimensiones: int, limites: List[Tuple[float, float]]) -> List[np.ndarray]:
    """
    Crea una población inicial de individuos.
    
    Args:
        tamano_poblacion: Número de individuos en la población.
        dimensiones: Número de variables por individuo.
        limites: Lista de tuplas (min, max) para cada dimensión.
        
    Returns:
        List[np.ndarray]: Lista de individuos.
    """
    return [crear_individuo(dimensiones, limites) for _ in range(tamano_poblacion)]

def evaluar_poblacion(poblacion: List[np.ndarray], funcion_objetivo: Callable, minimizar: bool = True) -> List[float]:
    """
    Evalúa la aptitud de cada individuo en la población.
    
    Args:
        poblacion: Lista de individuos.
        funcion_objetivo: Función a optimizar.
        minimizar: Si es True, busca minimizar la función; si es False, busca maximizar.
        
    Returns:
        List[float]: Lista de valores de aptitud.
    """
    aptitudes = [funcion_objetivo(individuo) for individuo in poblacion]
    return aptitudes if minimizar else [-1 * aptitud for aptitud in aptitudes]

def seleccion_torneo(poblacion: List[np.ndarray], aptitudes: List[float], tamano_torneo: int = 3) -> np.ndarray:
    """
    Selecciona un individuo mediante torneo.
    
    Args:
        poblacion: Lista de individuos.
        aptitudes: Lista de aptitudes correspondientes.
        tamano_torneo: Número de participantes en cada torneo.
        
    Returns:
        np.ndarray: Individuo seleccionado.
    """
    indices_torneo = random.sample(range(len(poblacion)), tamano_torneo)
    indice_ganador = min(indices_torneo, key=lambda i: aptitudes[i])
    return poblacion[indice_ganador].copy()

def cruce(padre1: np.ndarray, padre2: np.ndarray, prob_cruce: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza el cruce entre dos padres para generar dos hijos.
    """
    if random.random() > prob_cruce or len(padre1) < 2:  # <-- Evita cruce si hay menos de 2 genes
        return padre1.copy(), padre2.copy()
    
    punto_cruce = random.randint(1, len(padre1) - 1)  # Ahora seguro para dimensiones >= 2
    hijo1 = np.concatenate([padre1[:punto_cruce], padre2[punto_cruce:]])
    hijo2 = np.concatenate([padre2[:punto_cruce], padre1[punto_cruce:]])
    
    return hijo1, hijo2

def mutacion(individuo: np.ndarray, limites: List[Tuple[float, float]], prob_mutacion: float = 0.1) -> np.ndarray:
    """
    Aplica mutación a un individuo.
    
    Args:
        individuo: Individuo a mutar.
        limites: Lista de tuplas (min, max) para cada gen.
        prob_mutacion: Probabilidad de mutación por gen.
        
    Returns:
        np.ndarray: Individuo mutado.
    """
    for i in range(len(individuo)):
        if random.random() < prob_mutacion:
            min_val, max_val = limites[i]
            individuo[i] = random.uniform(min_val, max_val)
    
    return individuo

def algoritmo_genetico(
    funcion_objetivo: Callable,
    dimensiones: int,
    limites: List[Tuple[float, float]],
    tamano_poblacion: int = 100,
    max_generaciones: int = 100,
    prob_cruce: float = 0.8,
    prob_mutacion: float = 0.1,
    minimizar: bool = True,
    tolerancia: float = 1e-6,
    generaciones_sin_mejora: int = 20,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Algoritmo genético principal para optimizar una función objetivo.
    
    Args:
        funcion_objetivo: Función a optimizar.
        dimensiones: Número de variables (genes) por individuo.
        limites: Lista de tuplas (min, max) para cada dimensión.
        tamano_poblacion: Tamaño de la población.
        max_generaciones: Número máximo de generaciones.
        prob_cruce: Probabilidad de cruce.
        prob_mutacion: Probabilidad de mutación.
        minimizar: Si es True, minimiza la función; si es False, maximiza.
        tolerancia: Tolerancia para convergencia.
        generaciones_sin_mejora: Número de generaciones sin mejora para detener.
        verbose: Si es True, muestra información del progreso.
        
    Returns:
        Dict[str, Any]: Diccionario con resultados de la optimización.
    """
    # Validar que los límites sean correctos
    if len(limites) != dimensiones:
        raise ValueError(f"El número de límites ({len(limites)}) debe coincidir con el número de dimensiones ({dimensiones}).")
    
    # Inicializar población
    poblacion = crear_poblacion(tamano_poblacion, dimensiones, limites)
    
    # Evaluar población inicial
    aptitudes = evaluar_poblacion(poblacion, funcion_objetivo, minimizar)
    
    # Encontrar el mejor individuo
    mejor_indice = np.argmin(aptitudes)
    mejor_individuo = poblacion[mejor_indice].copy()
    mejor_aptitud = aptitudes[mejor_indice]
    
    # Historial para seguimiento
    historial_aptitudes = [mejor_aptitud]
    
    # Contador de generaciones sin mejora
    contador_sin_mejora = 0
    
    # Evolución
    for generacion in range(max_generaciones):
        # Nueva población
        nueva_poblacion = []
        
        # Elitismo: conservar al mejor individuo
        nueva_poblacion.append(mejor_individuo.copy())
        
        # Generar el resto de la población
        while len(nueva_poblacion) < tamano_poblacion:
            # Selección
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            
            # Cruce
            hijo1, hijo2 = cruce(padre1, padre2, prob_cruce)
            
            # Mutación
            hijo1 = mutacion(hijo1, limites, prob_mutacion)
            hijo2 = mutacion(hijo2, limites, prob_mutacion)
            
            # Añadir a la nueva población
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tamano_poblacion:
                nueva_poblacion.append(hijo2)
        
        # Actualizar población
        poblacion = nueva_poblacion
        
        # Evaluar nueva población
        aptitudes = evaluar_poblacion(poblacion, funcion_objetivo, minimizar)
        
        # Encontrar el mejor individuo actual
        mejor_indice_actual = np.argmin(aptitudes)
        mejor_individuo_actual = poblacion[mejor_indice_actual].copy()
        mejor_aptitud_actual = aptitudes[mejor_indice_actual]
        
        # Actualizar el mejor si hay mejora
        if mejor_aptitud_actual < mejor_aptitud:
            mejora = abs(mejor_aptitud - mejor_aptitud_actual)
            mejor_individuo = mejor_individuo_actual.copy()
            mejor_aptitud = mejor_aptitud_actual
            contador_sin_mejora = 0
            
            if verbose:
                print(f"Generación {generacion+1}: Mejor aptitud = {mejor_aptitud if minimizar else -mejor_aptitud}")
        else:
            contador_sin_mejora += 1
        
        # Guardar para historial
        historial_aptitudes.append(mejor_aptitud)
        
        # Criterio de parada por convergencia
        if contador_sin_mejora >= generaciones_sin_mejora:
            if verbose:
                print(f"Convergencia alcanzada en la generación {generacion+1}")
            break
    
    # Resultados
    resultado = {
        'mejor_solucion': mejor_individuo,
        'mejor_valor': mejor_aptitud if minimizar else -mejor_aptitud,
        'generaciones': generacion + 1,
        'historial_aptitudes': historial_aptitudes,
        'convergencia': contador_sin_mejora >= generaciones_sin_mejora
    }
    
    return resultado