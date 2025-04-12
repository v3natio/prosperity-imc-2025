import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Datos de contenedores
multipliers = [10, 80, 37, 31, 17, 90, 50, 20, 73, 89]
inhabitants = [1, 6, 3, 2, 1, 10, 4, 2, 4, 8]
base_treasure = 10000

# Crear DataFrame para los contenedores
container_data = pd.DataFrame({
    'Container': list(range(1, 11)),
    'Multiplier': multipliers,
    'Inhabitants': inhabitants,
})

# Calcular valor esperado simple
container_data['Expected Value'] = [base_treasure * m / i for m, i in zip(multipliers, inhabitants)]

# Parámetros de nuestro perfil
our_position = 90000  # Nuestra posición actual
risk_threshold = 80000  # Umbral para considerar averso al riesgo
average_shells = 60000  # Promedio general

print(f"Nuestra posición: {our_position} shells (Perfil: Averso al riesgo)")

# Definir comportamiento de selección según perfil de riesgo con más énfasis en aversión al riesgo
def calculate_selection_probabilities(container_data, risk_aversion_level=0.7):
    """
    Calcula probabilidades de selección con un nivel personalizado de aversión al riesgo
    
    Parameters:
    - container_data: DataFrame con datos de contenedores
    - risk_aversion_level: Nivel de aversión al riesgo (0-1), donde 1 es máxima aversión
    
    Returns:
    - DataFrame con probabilidades y valores ajustados
    """
    # Copiar datos
    data = container_data.copy()
    
    # Modelo de selección para aversos al riesgo (prefieren EV/varianza alta)
    # Prefieren contenedores con buen balance entre retorno y riesgo
    data['Risk Metric'] = data['Multiplier'] / data['Inhabitants']
    
    # Modelo más sofisticado para aversión al riesgo
    # Mayor aversión penaliza más los contenedores con muchos habitantes
    data['Risk Averse Score'] = data['Expected Value'] * (1 / (data['Inhabitants'] ** (1 + risk_aversion_level/2)))
    
    # Normalizar para obtener probabilidades
    total_risk_averse = data['Risk Averse Score'].sum()
    data['Risk Averse Prob'] = data['Risk Averse Score'] / total_risk_averse
    
    # Modelo para amantes del riesgo (prefieren multiplicadores altos)
    # Les importa más el multiplicador que los habitantes
    data['Risk Loving Score'] = data['Multiplier'] ** 2 / (data['Inhabitants'] ** 0.5)
    
    # Normalizar para obtener probabilidades
    total_risk_loving = data['Risk Loving Score'].sum()
    data['Risk Loving Prob'] = data['Risk Loving Score'] / total_risk_loving
    
    # Estimamos la proporción de jugadores arriba y abajo del umbral
    # Considerando el promedio y asumiendo distribución normal
    from scipy.stats import norm
    std_estimated = (risk_threshold - average_shells) / norm.ppf(0.75)  # Estimación aproximada
    z_score = (risk_threshold - average_shells) / std_estimated
    risk_averse_pct = 1 - norm.cdf(z_score)
    risk_loving_pct = 1 - risk_averse_pct
    
    print(f"Porcentaje estimado de jugadores aversos al riesgo: {risk_averse_pct:.2%}")
    print(f"Porcentaje estimado de jugadores amantes del riesgo: {risk_loving_pct:.2%}")
    
    # Calcular probabilidad combinada basada en la distribución de jugadores
    data['Combined Prob'] = (data['Risk Averse Prob'] * risk_averse_pct + 
                            data['Risk Loving Prob'] * risk_loving_pct)
    
    # Recalcular EV considerando las nuevas probabilidades
    # El factor 10 es un parámetro que controla cuánto afecta la probabilidad al divisor
    data['Adjusted EV'] = [base_treasure * m / (i + p * 10) 
                           for m, i, p in zip(data['Multiplier'], 
                                             data['Inhabitants'], 
                                             data['Combined Prob'])]
    
    # Calcular el valor esperado desde nuestra perspectiva (más aversa al riesgo)
    # Damos más peso a evitar contenedores con muchos habitantes
    data['Our EV Perspective'] = [base_treasure * m / (i + p * (10 + 5 * risk_aversion_level)) 
                                 for m, i, p in zip(data['Multiplier'], 
                                                   data['Inhabitants'], 
                                                   data['Combined Prob'])]
    
    # Riesgo percibido (para nosotros como aversos al riesgo)
    # Combinación de habitantes existentes y probabilidad de selección adicional
    data['Perceived Risk'] = data['Inhabitants'] + data['Combined Prob'] * 10
    
    # Índice de aversión al riesgo: cuanto mayor, mejor para un averso al riesgo
    data['Risk Aversion Index'] = data['Multiplier'] / data['Perceived Risk']
    
    return data, risk_averse_pct, risk_loving_pct

# Definimos un nivel de aversión al riesgo alto dado que tenemos 90,000 shells
# El nivel va de 0 (neutral) a 1 (extremadamente averso)
our_risk_aversion = 0.8  # Alto nivel de aversión al riesgo

# Calcular probabilidades y valores ajustados
result_data, risk_averse_pct, risk_loving_pct = calculate_selection_probabilities(
    container_data, risk_aversion_level=our_risk_aversion)

# Ordenar por valor esperado desde nuestra perspectiva
best_containers_for_us = result_data.sort_values('Our EV Perspective', ascending=False)

print("\nMejores contenedores según nuestra perspectiva (aversa al riesgo):")
print(best_containers_for_us[['Container', 'Multiplier', 'Inhabitants', 
                              'Expected Value', 'Combined Prob', 
                              'Adjusted EV', 'Our EV Perspective']].head().to_string(index=False))

# Visualización
plt.figure(figsize=(15, 10))

# Graficar probabilidades de selección
plt.subplot(2, 2, 1)
x = result_data['Container']
width = 0.3
plt.bar(x - width/2, result_data['Risk Averse Prob'], width=width, label='Aversos al Riesgo')
plt.bar(x + width/2, result_data['Risk Loving Prob'], width=width, label='Amantes del Riesgo')
plt.xlabel('Número de Contenedor')
plt.ylabel('Probabilidad de Selección')
plt.title('Probabilidades de Selección por Perfil de Riesgo')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar valor esperado según diferentes perspectivas
plt.subplot(2, 2, 2)
width = 0.25
plt.bar(x - width, result_data['Expected Value'], width=width, label='EV Original')
plt.bar(x, result_data['Adjusted EV'], width=width, label='EV Mercado')
plt.bar(x + width, result_data['Our EV Perspective'], width=width, label='Nuestra Perspectiva')
plt.xlabel('Número de Contenedor')
plt.ylabel('Valor Esperado')
plt.title('Valor Esperado según Diferentes Perspectivas')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar índice de aversión al riesgo
plt.subplot(2, 2, 3)
plt.bar(result_data['Container'], result_data['Risk Aversion Index'], color='green')
plt.xlabel('Número de Contenedor')
plt.ylabel('Índice de Aversión al Riesgo')
plt.title('Índice de Aversión al Riesgo por Contenedor')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar ranking de contenedores según nuestra perspectiva
plt.subplot(2, 2, 4)
sorted_containers = best_containers_for_us['Container'].values
sorted_evs = best_containers_for_us['Our EV Perspective'].values
plt.bar(range(len(sorted_containers)), sorted_evs, color='purple')
plt.xticks(range(len(sorted_containers)), sorted_containers)
plt.xlabel('Contenedor (ordenado para perfil averso)')
plt.ylabel('Valor Esperado (Perspectiva Aversa)')
plt.title('Ranking de Contenedores para Perfil Averso al Riesgo')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('risk_averse_analysis.png')

# Evaluación adicional considerando nuestra posición específica
# Análisis de sensibilidad: cómo cambia el ranking con diferentes niveles de aversión
aversion_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
top_containers = {}

for level in aversion_levels:
    level_data, _, _ = calculate_selection_probabilities(container_data, risk_aversion_level=level)
    top = level_data.sort_values('Our EV Perspective', ascending=False).iloc[0]['Container']
    top_containers[level] = int(top)

print("\nMejor contenedor según nivel de aversión al riesgo:")
for level, container in top_containers.items():
    print(f"Nivel de aversión {level:.1f}: Contenedor #{container}")

# Análisis de robustez: cómo cambia el ranking si cambia el % de jugadores con cada perfil
risk_averse_percentages = [0.2, 0.3, 0.4, 0.5, 0.6]
robustness_results = {}

for pct in risk_averse_percentages:
    # Recalcular con porcentaje fijo
    data = result_data.copy()
    data['Adjusted Prob'] = data['Risk Averse Prob'] * pct + data['Risk Loving Prob'] * (1-pct)
    
    # Recalcular EV desde nuestra perspectiva
    data['Test EV'] = [base_treasure * m / (i + p * (10 + 5 * our_risk_aversion)) 
                      for m, i, p in zip(data['Multiplier'], 
                                        data['Inhabitants'], 
                                        data['Adjusted Prob'])]
    
    top = data.sort_values('Test EV', ascending=False).iloc[0]['Container']
    robustness_results[pct] = int(top)

print("\nRobustez del ranking según % de jugadores aversos al riesgo:")
for pct, container in robustness_results.items():
    print(f"{pct:.1%} jugadores aversos al riesgo: Contenedor #{container}")

# Conclusión
best_container = best_containers_for_us.iloc[0]['Container']
best_ev = best_containers_for_us.iloc[0]['Our EV Perspective']
second_best = best_containers_for_us.iloc[1]['Container']
second_ev = best_containers_for_us.iloc[1]['Our EV Perspective']

# Verificar consistencia entre diferentes metodologías
methods = ['Expected Value', 'Adjusted EV', 'Our EV Perspective', 'Risk Aversion Index']
method_results = {}

for method in methods:
    top = result_data.sort_values(method, ascending=False).iloc[0]['Container']
    method_results[method] = int(top)

print("\nConsistencia entre diferentes metodologías:")
for method, container in method_results.items():
    print(f"Según {method}: Contenedor #{container}")

print("\nCONCLUSIÓN FINAL:")
print(f"Para nuestro perfil de 90,000 shells (averso al riesgo), el mejor contenedor es el #{int(best_container)}")
print(f"Con un valor esperado de {best_ev:.2f} desde nuestra perspectiva")
print(f"Seguido por el contenedor #{int(second_best)} con un valor esperado de {second_ev:.2f}")
print(f"La diferencia porcentual entre ambos es: {(best_ev/second_ev - 1):.2%}")

# Detalles del mejor contenedor
best_row = best_containers_for_us.iloc[0]
print("\nDetalles del mejor contenedor para nosotros:")
print(f"Contenedor #{int(best_container)}:")
print(f"- Multiplicador: {best_row['Multiplier']}")
print(f"- Habitantes: {best_row['Inhabitants']}")
print(f"- Valor esperado original: {best_row['Expected Value']:.2f}")
print(f"- Probabilidad estimada de selección: {best_row['Combined Prob']:.4f}")
print(f"- Índice de aversión al riesgo: {best_row['Risk Aversion Index']:.2f}")
print(f"- Valor esperado para nuestro perfil: {best_row['Our EV Perspective']:.2f}")

# Evaluación de estrategia de segundo contenedor (si fuera relevante)
print("\nEvaluación para un posible segundo contenedor:")
first_choice = int(best_container)
second_options = best_containers_for_us[best_containers_for_us['Container'] != first_choice].reset_index(drop=True)
print(f"Después de seleccionar el contenedor #{first_choice}, las mejores opciones serían:")
print(second_options[['Container', 'Multiplier', 'Inhabitants', 'Our EV Perspective']].head(3).to_string(index=False))