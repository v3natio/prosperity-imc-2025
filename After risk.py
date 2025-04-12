import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos de shells
shells_data = pd.read_csv('sea_shells_data.csv')

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

# Segmentar jugadores por perfil de riesgo
risk_threshold = 80000  # Valor umbral mencionado

# Promedio actual según el enunciado
average_shells = 60000

# Verificar cuántos jugadores están en cada categoría de riesgo según los datos proporcionados
risk_averse = shells_data[shells_data['Sea Shells'] * 1000 >= risk_threshold]
risk_loving = shells_data[shells_data['Sea Shells'] * 1000 < risk_threshold]

print(f"Total de jugadores: {len(shells_data)}")
print(f"Jugadores aversos al riesgo (≥ 80,000): {len(risk_averse)}")
print(f"Jugadores amantes del riesgo (< 80,000): {len(risk_loving)}")

# Análisis de la distribución real
print("\nEstadísticas de Sea Shells:")
print(f"Mínimo: {shells_data['Sea Shells'].min() * 1000:.2f}")
print(f"Máximo: {shells_data['Sea Shells'].max() * 1000:.2f}")
print(f"Promedio: {shells_data['Sea Shells'].mean() * 1000:.2f}")
print(f"Mediana: {shells_data['Sea Shells'].median() * 1000:.2f}")

# Definir comportamiento de selección según perfil de riesgo
def calculate_selection_probabilities(container_data, risk_averse_pct, risk_loving_pct):
    """
    Calcula probabilidades de selección según perfil de riesgo
    
    Parameters:
    - container_data: DataFrame con datos de contenedores
    - risk_averse_pct: Porcentaje de jugadores aversos al riesgo
    - risk_loving_pct: Porcentaje de jugadores amantes del riesgo
    
    Returns:
    - DataFrame con probabilidades
    """
    # Copiar datos
    data = container_data.copy()
    
    # Modelo de selección para aversos al riesgo (prefieren EV/varianza alta)
    # Prefieren contenedores con buen balance entre retorno y riesgo
    data['Risk Metric'] = data['Multiplier'] / data['Inhabitants']
    data['Risk Averse Score'] = data['Expected Value'] * (1 / (data['Inhabitants'] + 1))
    
    # Normalizar para obtener probabilidades
    total_risk_averse = data['Risk Averse Score'].sum()
    data['Risk Averse Prob'] = data['Risk Averse Score'] / total_risk_averse
    
    # Modelo para amantes del riesgo (prefieren multiplicadores altos)
    # Les importa más el multiplicador que los habitantes
    data['Risk Loving Score'] = data['Multiplier'] ** 2 / (data['Inhabitants'] ** 0.5)
    
    # Normalizar para obtener probabilidades
    total_risk_loving = data['Risk Loving Score'].sum()
    data['Risk Loving Prob'] = data['Risk Loving Score'] / total_risk_loving
    
    # Calcular probabilidad combinada
    data['Combined Prob'] = (data['Risk Averse Prob'] * risk_averse_pct + 
                            data['Risk Loving Prob'] * risk_loving_pct)
    
    # Recalcular EV considerando las nuevas probabilidades
    data['Adjusted EV'] = [base_treasure * m / (i + p * 10) 
                           for m, i, p in zip(data['Multiplier'], 
                                             data['Inhabitants'], 
                                             data['Combined Prob'])]
    
    return data

# En base a los datos proporcionados, calculamos porcentajes aproximados
# Todos los jugadores del dataset están por encima de 80,000
# Por lo que estimaremos la proporción en base a los datos del enunciado
# El promedio actual es 60,000 y el umbral es 80,000

# Asumimos que la distribución sigue aproximadamente una distribución normal
# y estimamos la proporción de jugadores arriba de 80,000
from scipy.stats import norm

# Calculamos la desviación estándar observada en los datos
std_observed = shells_data['Sea Shells'].std() * 1000

# Estimamos la proporción de jugadores con más de 80,000
z_score = (risk_threshold - average_shells) / std_observed
risk_averse_pct = 1 - norm.cdf(z_score)
risk_loving_pct = 1 - risk_averse_pct

print(f"\nPorcentaje estimado de jugadores aversos al riesgo: {risk_averse_pct:.2%}")
print(f"Porcentaje estimado de jugadores amantes del riesgo: {risk_loving_pct:.2%}")

# Calcular probabilidades y valores esperados ajustados
result_data = calculate_selection_probabilities(container_data, risk_averse_pct, risk_loving_pct)

# Ordenar por valor esperado ajustado para ver los mejores contenedores
best_containers = result_data.sort_values('Adjusted EV', ascending=False)

print("\nMejores contenedores según valor esperado ajustado:")
print(best_containers[['Container', 'Multiplier', 'Inhabitants', 
                       'Expected Value', 'Combined Prob', 'Adjusted EV']].head().to_string(index=False))

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

# Graficar valor esperado original vs ajustado
plt.subplot(2, 2, 2)
plt.bar(x - width/2, result_data['Expected Value'], width=width, label='EV Original')
plt.bar(x + width/2, result_data['Adjusted EV'], width=width, label='EV Ajustado')
plt.xlabel('Número de Contenedor')
plt.ylabel('Valor Esperado')
plt.title('Valor Esperado Original vs Ajustado')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Graficar relación entre multiplicador, habitantes y EV ajustado
plt.subplot(2, 2, 3)
sizes = result_data['Adjusted EV'] / 10000  # Escalar para visualización
plt.scatter(result_data['Multiplier'], result_data['Inhabitants'], s=sizes*100, alpha=0.7)
for i, row in result_data.iterrows():
    plt.annotate(f"{row['Container']}", 
                (row['Multiplier'], row['Inhabitants']),
                textcoords="offset points",
                xytext=(0,10),
                ha='center')
plt.xlabel('Multiplicador')
plt.ylabel('Habitantes')
plt.title('Relación entre Multiplicador, Habitantes y EV Ajustado')
plt.grid(True, linestyle='--', alpha=0.7)

# Graficar ranking de contenedores por EV ajustado
plt.subplot(2, 2, 4)
sorted_containers = best_containers['Container'].values
sorted_evs = best_containers['Adjusted EV'].values
plt.bar(range(len(sorted_containers)), sorted_evs)
plt.xticks(range(len(sorted_containers)), sorted_containers)
plt.xlabel('Contenedor (ordenado por valor)')
plt.ylabel('Valor Esperado Ajustado')
plt.title('Ranking de Contenedores por EV Ajustado')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('risk_profile_analysis.png')

# Conclusión
best_container = best_containers.iloc[0]['Container']
best_ev = best_containers.iloc[0]['Adjusted EV']
second_best = best_containers.iloc[1]['Container']
second_ev = best_containers.iloc[1]['Adjusted EV']

print("\nCONCLUSIÓN:")
print(f"El mejor contenedor es el #{int(best_container)} con un valor esperado ajustado de {best_ev:.2f}")
print(f"Seguido por el contenedor #{int(second_best)} con un valor esperado de {second_ev:.2f}")
print(f"La diferencia porcentual entre ambos es: {(best_ev/second_ev - 1):.2%}")

# Detalles del mejor contenedor
best_row = best_containers.iloc[0]
print("\nDetalles del mejor contenedor:")
print(f"Contenedor #{int(best_container)}:")
print(f"- Multiplicador: {best_row['Multiplier']}")
print(f"- Habitantes: {best_row['Inhabitants']}")
print(f"- Valor esperado original: {best_row['Expected Value']:.2f}")
print(f"- Probabilidad estimada de selección: {best_row['Combined Prob']:.4f}")
print(f"- Valor esperado ajustado: {best_row['Adjusted EV']:.2f}")