import json
import numpy as np

# Cargar el archivo JSON
with open('/home/mapacheroja/apr-lab-2/20260107_063620/dqn_results.json', 'r') as file:
    data = json.load(file)

# Extraer los pasos hasta el éxito
steps_to_success = data["final_eval"]["steps_to_success"]

# Calcular el promedio y desviación estándar de los pasos hasta el éxito
mean_steps = np.mean(steps_to_success)
std_steps = np.std(steps_to_success)

# Mostrar los resultados
print(f"Promedio de pasos hasta el éxito: {mean_steps:.2f}")
print(f"Desviación estándar de los pasos hasta el éxito: {std_steps:.2f}")
