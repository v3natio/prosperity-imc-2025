import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Datos de contenedores
multipliers = [10, 80, 37, 31, 17, 90, 50, 20, 73, 89]
inhabitants = [1, 6, 3, 2, 1, 10, 4, 2, 4, 8]
base_treasure = 10000
total_participants = 15000  # Número total de participantes
second_container_cost = 50000  # Costo de elegir un segundo contenedor

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
print(f"Total de participantes: {total_participants}")
print(f"Costo de elegir un segundo contenedor: {second_container_cost:,} shells")

# Definir comportamiento de selección según perfil de riesgo con más énfasis en aversión al riesgo
def calculate_selection_probabilities(container_data, risk_aversion_level=0.7, total_participants=15000):
    """
    Calcula probabilidades de selección con un nivel personalizado de aversión al riesgo
    
    Parameters:
    - container_data: DataFrame con datos de contenedores
    - risk_aversion_level: Nivel de aversión al riesgo (0-1), donde 1 es máxima aversión
    - total_participants: Número total de participantes en el juego
    
    Returns:
    - DataFrame con probabilidades y valores ajustados
    """
    # Copiar datos
    data = container_data.copy()
    
    # Modelo de selección para aversos al riesgo basado en Expected Value
    data['Risk Metric'] = data['Expected Value'] / data['Inhabitants']
    
    # Modelo más sofisticado para aversión al riesgo
    # Mayor aversión penaliza más los contenedores con muchos habitantes
    data['Risk Averse Score'] = data['Expected Value'] * (1 / (data['Inhabitants'] ** (1 + risk_aversion_level/2)))
    
    # Normalizar para obtener probabilidades
    total_risk_averse = data['Risk Averse Score'].sum()
    data['Risk Averse Prob'] = data['Risk Averse Score'] / total_risk_averse
    
    # Modelo para amantes del riesgo ahora basado en Expected Value
    data['Risk Loving Score'] = data['Expected Value'] ** 2 / (data['Inhabitants'] ** 0.5)
    
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
    
    # NUEVO: Estimamos qué % de jugadores elegirá un segundo contenedor
    # Asumimos que jugadores con más shells que el costo del segundo contenedor tienen mayor probabilidad
    # Esto es simplificado y podría mejorarse con datos más precisos
    z_score_second = (second_container_cost - average_shells) / std_estimated
    pct_can_afford_second = 1 - norm.cdf(z_score_second)
    
    # Los jugadores amantes del riesgo son más propensos a elegir un segundo contenedor
    # Los jugadores aversos al riesgo son menos propensos
    second_container_pct = pct_can_afford_second * (0.8 * risk_loving_pct + 0.2 * risk_averse_pct)
    
    print(f"Porcentaje estimado de jugadores que elegirán un segundo contenedor: {second_container_pct:.2%}")
    
    # Calcular probabilidad combinada basada en la distribución de jugadores para el primer contenedor
    data['Combined Prob First'] = (data['Risk Averse Prob'] * risk_averse_pct + 
                                  data['Risk Loving Prob'] * risk_loving_pct)
    
    # NUEVO: Calcular probabilidad para el segundo contenedor
    # Asumimos que los jugadores evitan elegir el mismo contenedor dos veces
    # La probabilidad se redistribuye entre los demás contenedores
    data['Second Container Prob'] = data['Combined Prob First'] * second_container_pct / (1 - data['Combined Prob First'])
    data['Second Container Prob'] = data['Second Container Prob'] / data['Second Container Prob'].sum()
    
    # NUEVO: Probabilidad total considerando primeras y segundas elecciones
    data['Total Selection Prob'] = data['Combined Prob First'] + data['Second Container Prob']
    
    # Calcular el número esperado de jugadores por contenedor (primeras elecciones)
    data['Expected Players First'] = data['Combined Prob First'] * total_participants
    
    # Calcular el número esperado de jugadores por contenedor (segundas elecciones)
    data['Expected Players Second'] = data['Second Container Prob'] * (total_participants * second_container_pct)
    
    # Total de jugadores esperados por contenedor
    data['Expected Players Total'] = data['Expected Players First'] + data['Expected Players Second']
    
    # Densidad de competidores (jugadores esperados por habitante existente)
    data['Competitor Density'] = data['Expected Players Total'] / data['Inhabitants']
    
    # Factor de congestión que afecta el valor esperado
    # Cuantos más jugadores compitan por el mismo contenedor, menor será el valor esperado real
    data['Congestion Factor'] = 1 + (data['Expected Players Total'] / total_participants) * 10
    
    # Recalcular EV considerando las nuevas probabilidades y el factor de congestión
    data['Adjusted EV'] = data['Expected Value'] / data['Congestion Factor']
    
    # Calcular el valor esperado desde nuestra perspectiva (más aversa al riesgo)
    # Ajustamos el valor esperado con un factor que incluye nuestra aversión al riesgo
    risk_penalty = 1 + (data['Expected Players Total'] / total_participants) * (10 + 5 * risk_aversion_level)
    data['Our EV Perspective'] = data['Expected Value'] / risk_penalty
    
    # Riesgo percibido (para nosotros como aversos al riesgo)
    # Ahora incluye habitantes existentes, jugadores esperados y nuestra aversión al riesgo
    data['Perceived Risk'] = data['Inhabitants'] + data['Expected Players Total'] * (1 + risk_aversion_level) / 100
    
    # Índice de aversión al riesgo
    data['Risk Aversion Index'] = data['Expected Value'] / data['Perceived Risk']
    
    return data, risk_averse_pct, risk_loving_pct, second_container_pct

# Definimos un nivel de aversión al riesgo alto dado que tenemos 90,000 shells
# El nivel va de 0 (neutral) a 1 (extremadamente averso)
our_risk_aversion = 0.8  # Alto nivel de aversión al riesgo

# Calcular probabilidades y valores ajustados
result_data, risk_averse_pct, risk_loving_pct, second_container_pct = calculate_selection_probabilities(
    container_data, risk_aversion_level=our_risk_aversion, total_participants=total_participants)

# Ordenar por valor esperado desde nuestra perspectiva
best_containers_for_us = result_data.sort_values('Our EV Perspective', ascending=False)

print("\nMejores contenedores según nuestra perspectiva (aversa al riesgo):")
print(best_containers_for_us[['Container', 'Multiplier', 'Inhabitants', 
                              'Expected Value', 'Expected Players Total', 
                              'Adjusted EV', 'Our EV Perspective']].head().to_string(index=False))

# Visualización
plt.figure(figsize=(15, 12))

# Graficar probabilidades de selección
plt.subplot(3, 2, 1)
x = result_data['Container']
width = 0.3
plt.bar(x - width/2, result_data['Combined Prob First'], width=width, label='Primera Elección')
plt.bar(x + width/2, result_data['Second Container Prob'], width=width, label='Segunda Elección')
plt.xlabel('Número de Contenedor')
plt.ylabel('Probabilidad de Selección')
plt.title('Probabilidades de Selección por Orden de Elección')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar valor esperado según diferentes perspectivas
plt.subplot(3, 2, 2)
width = 0.25
plt.bar(x - width, result_data['Expected Value'], width=width, label='EV Original')
plt.bar(x, result_data['Adjusted EV'], width=width, label='EV Mercado')
plt.bar(x + width, result_data['Our EV Perspective'], width=width, label='Nuestra Perspectiva')
plt.xlabel('Número de Contenedor')
plt.ylabel('Valor Esperado')
plt.title('Valor Esperado según Diferentes Perspectivas')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar jugadores esperados por contenedor
plt.subplot(3, 2, 3)
width = 0.3
plt.bar(x - width/2, result_data['Expected Players First'], width=width, label='Primera Elección')
plt.bar(x + width/2, result_data['Expected Players Second'], width=width, label='Segunda Elección')
plt.xlabel('Número de Contenedor')
plt.ylabel('Jugadores Esperados')
plt.title(f'Distribución Esperada de {total_participants} Participantes')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar densidad de competidores
plt.subplot(3, 2, 4)
plt.bar(result_data['Container'], result_data['Competitor Density'], color='red')
plt.xlabel('Número de Contenedor')
plt.ylabel('Densidad de Competidores')
plt.title('Jugadores Esperados por Habitante Existente')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar índice de aversión al riesgo
plt.subplot(3, 2, 5)
plt.bar(result_data['Container'], result_data['Risk Aversion Index'], color='green')
plt.xlabel('Número de Contenedor')
plt.ylabel('Índice de Aversión al Riesgo')
plt.title('Índice de Aversión al Riesgo por Contenedor')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar ranking de contenedores según nuestra perspectiva
plt.subplot(3, 2, 6)
sorted_containers = best_containers_for_us['Container'].values
sorted_evs = best_containers_for_us['Our EV Perspective'].values
plt.bar(range(len(sorted_containers)), sorted_evs, color='purple')
plt.xticks(range(len(sorted_containers)), sorted_containers)
plt.xlabel('Contenedor (ordenado para perfil averso)')
plt.ylabel('Valor Esperado (Perspectiva Aversa)')
plt.title('Ranking de Contenedores para Perfil Averso al Riesgo')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('risk_averse_analysis_with_second_container.png')

# Conclusión para primer contenedor
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

print("\nConsistencia entre diferentes metodologías para primer contenedor:")
for method, container in method_results.items():
    print(f"Según {method}: Contenedor #{container}")

print("\nCONCLUSIÓN PARA PRIMER CONTENEDOR:")
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
print(f"- Jugadores esperados: {best_row['Expected Players Total']:.1f} de {total_participants:,} participantes")
print(f"- Densidad de competidores: {best_row['Competitor Density']:.2f} jugadores por habitante")
print(f"- Índice de aversión al riesgo: {best_row['Risk Aversion Index']:.2f}")
print(f"- Valor esperado para nuestro perfil: {best_row['Our EV Perspective']:.2f}")

# NUEVO: Análisis para un segundo contenedor considerando el costo
print("\n" + "="*50)
print("ANÁLISIS DE SEGUNDO CONTENEDOR (con costo de 50,000 shells)")
print("="*50)

# Calcular si vale la pena pagar 50,000 por un segundo contenedor
first_choice = int(best_container)
remaining_shells = our_position - second_container_cost
print(f"Shells remanentes después de pagar el segundo contenedor: {remaining_shells:,}")

if remaining_shells <= 0:
    print("No podemos permitirnos un segundo contenedor (costo mayor a nuestra posición)")
else:
    # Recalcular aversión al riesgo con nuestra nueva posición
    # Asumimos que con menos shells seremos más aversos al riesgo
    our_new_aversion = min(1.0, our_risk_aversion + 0.2)
    print(f"Nuestro nuevo nivel de aversión al riesgo: {our_new_aversion:.1f} (aumentó debido a menor capital)")

    # Segunda elección no debería incluir el mismo contenedor
    second_options = result_data[result_data['Container'] != first_choice].copy()
    
    # Recalcular perspectiva con nueva aversión al riesgo
    # Y considerando que ya habrá más jugadores en los contenedores (los que eligieron primero)
    second_risk_penalty = 1 + ((second_options['Expected Players Total'] + second_options['Expected Players First']) / 
                              total_participants) * (10 + 5 * our_new_aversion)
    second_options['Second Choice EV'] = second_options['Expected Value'] / second_risk_penalty
    
    # Ordenar por perspectiva para segunda elección
    second_best_containers = second_options.sort_values('Second Choice EV', ascending=False)
    
    print("\nMejores opciones para segundo contenedor:")
    print(second_best_containers[['Container', 'Multiplier', 'Inhabitants', 
                                 'Expected Value', 'Expected Players Total', 
                                 'Second Choice EV']].head(3).to_string(index=False))
    
    # Mejor segundo contenedor
    second_container = second_best_containers.iloc[0]['Container']
    second_container_ev = second_best_containers.iloc[0]['Second Choice EV']
    
    # IMPORTANTE: Análisis de si vale la pena elegir un segundo contenedor
    # Calcular valor esperado real considerando el costo del segundo contenedor
    
    # Valor esperado de no elegir segundo contenedor (solo el primero)
    first_only_ev = best_ev
    
    # Valor esperado combinado de elegir dos contenedores (considerando costo)
    # Necesitamos considerar el costo de oportunidad de los 50,000 shells
    # Una forma es calcularlo como retorno relativo vs. nuestro capital inicial
    combined_ev = best_ev + (second_container_ev - second_container_cost / our_position * best_ev)
    
    # Otra forma es como valor esperado neto
    net_combined_ev = best_ev + second_container_ev - second_container_cost / base_treasure
    
    print("\nAnálisis de valor esperado para decisión de segundo contenedor:")
    print(f"EV con solo primer contenedor (#{first_choice}): {first_only_ev:.2f}")
    print(f"EV combinado con segundo contenedor (#{int(second_container)}), ajustado por costo: {combined_ev:.2f}")
    print(f"EV neto combinado: {net_combined_ev:.2f}")
    
    # Decisión final sobre segundo contenedor
    if combined_ev > first_only_ev:
        improvement = (combined_ev / first_only_ev - 1) * 100
        print(f"\nRECOMENDACIÓN: SÍ elegir un segundo contenedor (mejora EV en {improvement:.2f}%)")
        print(f"El segundo contenedor debería ser el #{int(second_container)}")
    else:
        decrease = (1 - combined_ev / first_only_ev) * 100
        print(f"\nRECOMENDACIÓN: NO elegir un segundo contenedor (reduce EV en {decrease:.2f}%)")
        print(f"El costo de 50,000 shells no se justifica con el valor esperado del contenedor #{int(second_container)}")

# Estimación de la porción que obtendríamos del mejor contenedor (simplificada)
print("\n" + "="*50)
print("CONCLUSIÓN FINAL")
print("="*50)

# Análisis con el mejor primer contenedor
best_container_treasure = base_treasure * best_row['Multiplier']
expected_share_first = best_container_treasure / (best_row['Expected Players Total'] + best_row['Inhabitants'])
print(f"Tesoro en el contenedor #{int(best_container)}: {best_container_treasure:,} shells")
print(f"Porción esperada con competencia: {expected_share_first:.2f} shells")
print(f"Rendimiento esperado sobre nuestra posición actual: {expected_share_first/our_position - 1:.2%}")

# Recomendación final simplificada
print("\nESTRATEGIA ÓPTIMA:")
print(f"1. Seleccionar el contenedor #{int(best_container)} como primera elección")

if remaining_shells <= 0:
    print("2. No podemos permitirnos un segundo contenedor")
elif combined_ev > first_only_ev:
    print(f"2. Seleccionar el contenedor #{int(second_container)} como segunda elección")
    # Estimación combinada simplificada
    second_best_treasure = base_treasure * second_best_containers.iloc[0]['Multiplier']
    expected_share_second = second_best_treasure / (second_best_containers.iloc[0]['Expected Players Total'] + 
                                                  second_best_containers.iloc[0]['Inhabitants'])
    combined_gain = expected_share_first + expected_share_second - second_container_cost
    print(f"   Ganancia esperada total: {combined_gain:.2f} shells")
    print(f"   ROI total esperado: {combined_gain/our_position - 1:.2%}")
else:
    print("2. No elegir un segundo contenedor (el costo no justifica el beneficio esperado)")