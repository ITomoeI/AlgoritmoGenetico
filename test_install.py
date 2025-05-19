from genetic_algorithm import algoritmo_genetico

# Definir una función objetivo simple (minimizar x²)
def funcion_objetivo(x):
    return x[0]**2

# Ejecutar el algoritmo genético
print("Ejecutando algoritmo genético para minimizar x²...")
resultado = algoritmo_genetico(
    funcion_objetivo=funcion_objetivo,
    dimensiones=1,
    limites=[(-10, 10)],
    tamano_poblacion=30,
    max_generaciones=50,
    verbose=True
)

# Mostrar resultados
print("\nResultados finales:")
print(f"Mejor solución encontrada: {resultado['mejor_solucion']}")
print(f"Mejor valor de la función: {resultado['mejor_valor']}")
print(f"Generaciones utilizadas: {resultado['generaciones']}")