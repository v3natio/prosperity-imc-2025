import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data from the image
multipliers = [10, 80, 37, 31, 17, 90, 50, 20, 73, 89]
inhabitants = [1, 6, 3, 2, 1, 10, 4, 2, 4, 8]
base_treasure = 10000

# Create a DataFrame for better organization
container_data = pd.DataFrame({
    'Container': list(range(1, 11)),
    'Multiplier': multipliers,
    'Inhabitants': inhabitants,
})

# Calculate simple expected value (without selection percentage)
def calculate_simple_ev(multiplier, inhabitant, base_treasure):
    """Calculate simple expected value without considering selection percentage"""
    return (base_treasure * multiplier) / inhabitant

# Calculate expected values
container_data['Expected Value'] = [calculate_simple_ev(m, i, base_treasure) 
                                   for m, i in zip(multipliers, inhabitants)]

# Sort by expected value to see best options
sorted_by_ev = container_data.sort_values('Expected Value', ascending=False).reset_index(drop=True)

# Create more comprehensive model with selection percentage
def calculate_comprehensive_ev(multiplier, inhabitant, base_treasure, selection_percentage):
    """Calculate expected value considering both inhabitants and selection percentage"""
    divisor = inhabitant + selection_percentage * 10  # Scale percentage to match inhabitants magnitude
    return (base_treasure * multiplier) / divisor

# Define different scenarios for selection percentages
percentage_scenarios = {
    'Equal Popularity': [0.1] * 10,  # All containers equally likely to be chosen
    'High Multiplier Bias': [0.05, 0.15, 0.1, 0.1, 0.05, 0.2, 0.1, 0.05, 0.1, 0.15],  # People prefer high multipliers
    'Low Inhabitant Bias': [0.15, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.1, 0.05]   # People prefer low inhabitants
}

# Calculate expected values with different scenarios
for scenario, percentages in percentage_scenarios.items():
    evs = []
    for i in range(len(container_data)):
        m = container_data.loc[i, 'Multiplier']
        inh = container_data.loc[i, 'Inhabitants']
        p = percentages[i] if i < len(percentages) else 0.1
        evs.append(calculate_comprehensive_ev(m, inh, base_treasure, p))
    
    container_data[f'EV ({scenario})'] = evs

# Calculate optimal strategy for selecting two containers (first free, second with cost)
def optimal_two_container_strategy(container_data, second_container_cost, scenario='Expected Value'):
    """
    Determine optimal strategy for selecting two containers
    
    Parameters:
    - container_data: DataFrame with container information
    - second_container_cost: Cost to open second container
    - scenario: Which expected value column to use
    
    Returns:
    - Dictionary with optimal strategy information
    """
    # Deep copy to avoid modifying original data
    data = container_data.copy()
    
    # For first container (free), simply take the highest expected value
    best_first_idx = data[scenario].idxmax()
    best_first = data.loc[best_first_idx, 'Container']
    best_first_value = data.loc[best_first_idx, scenario]
    
    # Remove the first container from consideration for the second
    data = data[data['Container'] != best_first].reset_index(drop=True)
    
    # For second container, we need to account for the cost
    data['Net Value'] = data[scenario] - second_container_cost
    
    # Find the best second container
    best_second_idx = data['Net Value'].idxmax()
    best_second = data.loc[best_second_idx, 'Container']
    best_second_value = data.loc[best_second_idx, 'Net Value']
    
    # Check if second container is worth it
    worth_second = best_second_value > 0
    
    return {
        'first_container': int(best_first),
        'first_container_value': best_first_value,
        'second_container': int(best_second) if worth_second else None,
        'second_container_value': best_second_value if worth_second else 0,
        'worth_second_container': worth_second,
        'total_expected_value': best_first_value + (best_second_value if worth_second else 0)
    }

# Example costs for second container
costs = [10000, 25000, 50000, 75000, 100000]

# Print basic analysis results
print("Shipping Container Treasure Analysis")
print("=" * 60)
print("\nBasic Container Data:")
print("-" * 60)
print(container_data[['Container', 'Multiplier', 'Inhabitants', 'Expected Value']].to_string(index=False))

print("\nTop Containers by Expected Value:")
print("-" * 60)
print(sorted_by_ev[['Container', 'Multiplier', 'Inhabitants', 'Expected Value']].head(3).to_string(index=False))

print("\nComprehensive Analysis with Different Selection Percentage Scenarios:")
print("=" * 60)
for scenario in percentage_scenarios:
    top_containers = container_data.sort_values(f'EV ({scenario})', ascending=False)[
        ['Container', 'Multiplier', 'Inhabitants', f'EV ({scenario})']].head(3)
    print(f"\nTop Containers for {scenario} Scenario:")
    print("-" * 60)
    print(top_containers.to_string(index=False))

print("\nOptimal Two-Container Strategies with Different Costs:")
print("=" * 60)
for cost in costs:
    strategy = optimal_two_container_strategy(container_data, cost)
    print(f"\nWith Second Container Cost: {cost}")
    print("-" * 40)
    print(f"First container (free): Container {strategy['first_container']}")
    print(f"Expected value: {strategy['first_container_value']:.2f}")
    
    if strategy['worth_second_container']:
        print(f"Second container: Container {strategy['second_container']}")
        print(f"Expected value after cost: {strategy['second_container_value']:.2f}")
    else:
        print("Second container: Not worth opening due to cost")
    
    print(f"Total expected value: {strategy['total_expected_value']:.2f}")

# Create visualizations
plt.figure(figsize=(15, 10))

# Bar chart for simple expected values
plt.subplot(2, 2, 1)
plt.bar(container_data['Container'], container_data['Expected Value'], color='skyblue')
plt.xlabel('Container Number')
plt.ylabel('Expected Value')
plt.title('Simple Expected Value by Container')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Bar chart for multiplier-to-inhabitants ratio
plt.subplot(2, 2, 2)
plt.bar(container_data['Container'], container_data['Multiplier'] / container_data['Inhabitants'], color='lightgreen')
plt.xlabel('Container Number')
plt.ylabel('Multiplier/Inhabitants Ratio')
plt.title('Multiplier to Inhabitant Ratio by Container')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Bar chart comparing top containers across scenarios
plt.subplot(2, 2, 3)
scenario_columns = [f'EV ({scenario})' for scenario in percentage_scenarios]
top_3_containers = sorted_by_ev['Container'].head(3).tolist()
scenario_data = container_data[container_data['Container'].isin(top_3_containers)][['Container'] + scenario_columns]

# Reshape for plotting
scenario_data_melted = pd.melt(scenario_data, id_vars='Container', value_vars=scenario_columns, 
                              var_name='Scenario', value_name='Expected Value')
scenario_data_melted['Scenario'] = scenario_data_melted['Scenario'].apply(lambda x: x[4:-1])  # Remove 'EV (' and ')'

# Plot grouped bar chart
container_groups = scenario_data_melted.groupby('Container')
x = np.arange(len(top_3_containers))
width = 0.25
i = 0

for scenario in [s[4:-1] for s in scenario_columns]:  # Remove 'EV (' and ')'
    scenario_data = scenario_data_melted[scenario_data_melted['Scenario'] == scenario]
    plt.bar(x + (i-1)*width, scenario_data['Expected Value'], width=width, 
            label=scenario)
    i += 1

plt.xlabel('Container Number')
plt.ylabel('Expected Value')
plt.title('Top Containers Across Different Scenarios')
plt.xticks(x, top_3_containers)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Line chart showing impact of second container cost
plt.subplot(2, 2, 4)
cost_impact = []
for cost in np.linspace(0, 200000, 50):
    strategy = optimal_two_container_strategy(container_data, cost)
    cost_impact.append({
        'Cost': cost,
        'Total Expected Value': strategy['total_expected_value'],
        'Worth Second Container': strategy['worth_second_container'],
        'Second Container': strategy['second_container'] if strategy['worth_second_container'] else None
    })

cost_impact_df = pd.DataFrame(cost_impact)

# Plot total expected value line
plt.plot(cost_impact_df['Cost'], cost_impact_df['Total Expected Value'], 'b-', label='Total Expected Value')

# Mark the threshold where second container is no longer worth it
threshold_idx = cost_impact_df['Worth Second Container'].idxmin()
if threshold_idx < len(cost_impact_df) - 1:
    threshold_cost = cost_impact_df.loc[threshold_idx, 'Cost']
    plt.axvline(x=threshold_cost, color='r', linestyle='--', 
               label=f'Cost Threshold: {threshold_cost:.2f}')

plt.xlabel('Second Container Cost')
plt.ylabel('Total Expected Value')
plt.title('Impact of Second Container Cost on Total Expected Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('container_analysis.png')  # Save the figure
plt.show()

# Print final recommendations
print("\nFinal Recommendations:")
print("=" * 60)
print("1. For your free first container choice:")
top_container = sorted_by_ev.iloc[0]['Container']
top_ev = sorted_by_ev.iloc[0]['Expected Value']
print(f"   Choose Container {int(top_container)} with expected value of {top_ev:.2f}")

print("\n2. For your second container choice (if applicable):")
print("   It depends on the cost in SeaShells:")
threshold_cost = float('inf')
for i, row in cost_impact_df.iterrows():
    if not row['Worth Second Container'] and i > 0:
        threshold_cost = cost_impact_df.iloc[i-1]['Cost']
        break

print(f"   - If cost is less than {threshold_cost:.2f}, open a second container")
print(f"   - If cost is more than {threshold_cost:.2f}, don't open a second container")

for cost in [10000, 25000, 50000, 100000]:
    strategy = optimal_two_container_strategy(container_data, cost)
    if strategy['worth_second_container']:
        print(f"   - If cost is {cost}, choose Container {strategy['second_container']} " +
              f"(net value: {strategy['second_container_value']:.2f})")
    else:
        print(f"   - If cost is {cost}, don't open a second container")