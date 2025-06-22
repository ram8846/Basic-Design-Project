import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define project tasks
tasks = {
    'Section': [
        '1: Introduction to Design',
        '2: Problem Definition',
        '3: System Requirements',
        '4: Conceptual Design',
        '5: Functional Decomposition',
        '6: Behavioral Models',
        '7: Modeling & Simulation',
        '8: Prototyping & Testing',
        '9: Design Alternatives',
        '10: Iterative Optimization',
        '11: System Reliability',
        '12: Project Management'
    ],
    'Start Date': [datetime(2025, 3, 3) + timedelta(weeks=i) for i in range(12)],
    'Duration (days)': [6]*12  # Assume each task takes 6 days
}

# Create DataFrame
df = pd.DataFrame(tasks)
df['End Date'] = df['Start Date'] + pd.to_timedelta(df['Duration (days)'], unit='d')

# Plot Gantt Chart
plt.figure(figsize=(10, 6))
for i, task in df.iterrows():
    plt.barh(task['Section'], (task['End Date'] - task['Start Date']).days, left=task['Start Date'], color='skyblue')

plt.xlabel('Timeline')
plt.title('📅 Gantt Chart: Jet Engine Health Monitoring Project')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
