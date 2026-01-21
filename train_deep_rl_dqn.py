# Importación de librerías
from datetime import datetime
import os
os.environ['MPLBACKEND'] = 'Agg'  # Establecer el backend de Matplotlib antes de importar matplotlib
import gymnasium as gym
import numpy as np
import time
import optuna
import matplotlib
matplotlib.use('Agg', force=True)  # Usar backend no interactivo para guardar imágenes
import matplotlib.pyplot as plt
plt.ioff()  # Desactivar el modo interactivo de Matplotlib
from stable_baselines3 import DQN 
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
import joblib
import json
from typing import Tuple, Dict, Any

from cubesat_detumbling_rl import CubeSatDetumblingEnv  
from SatellitePersonality import SatellitePersonality 

best_reward = -float('inf')  # Inicializa la mejor recompensa con un valor muy bajo

# Adapter wrapper to make a Gymnasium env present the legacy Gym API
# (reset()->obs, step()->(obs, reward, done, info)). Stable-Baselines3's
# DummyVecEnv expects the old-style reset/step signatures, otherwise it may
# try to store a tuple into a numeric buffer and raise the ValueError shown.
class GymnasiumToGymWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Gymnasium reset returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs
        return result

    def step(self, action):
        result = self.env.step(action)
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return result

class TrialEvalCallback(EvalCallback):
    """
    Callback que evalúa el modelo periódicamente y le avisa a Optuna.
    Si el modelo va mal, Optuna manda la orden de cortar (Pruning).
    """
    def __init__(self, eval_env, trial, n_eval_episodes=50, eval_freq=1000, deterministic=True, verbose=0):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return True


def make_env(env_id: str, seed: int, monitor_path: str) -> gym.Env:
    """
    Crea el entorno de simulación y lo adapta para que sea compatible con Stable Baselines3.
    Args:
        env_id: ID del entorno (no usado aquí, pero podría usarse para seleccionar
            diferentes entornos si se implementan varios).
        seed: Semilla para el entorno.
        monitor_path: Ruta del archivo para Monitor (puede ser None).
    Returns:
        gym.Env: Entorno vectorizado que contiene una instancia del entorno base envuelta con:
    1) GymnasiumToGymWrapper para adaptar la API de Gymnasium a Gym.
    2) Monitor para registrar estadísticas.
    """
    def _init():
        env = CubeSatDetumblingEnv(render_mode=None)
        env = Monitor(env, filename=monitor_path)  # si monitor_path es None, igual funciona
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return DummyVecEnv([_init])

def train_dqn_return_model(env, hyperparams, total_timesteps=200000, tensorboard_log=None, trial=None):
    """
    Configura y entrena un modelo DQN, integrando soporte opcional para Optuna.

    Args:
        env (gym.Env): Entorno de entrenamiento (vectorizado).
        hyperparams (dict): Diccionario con hiperparámetros (lr, batch_size, gamma, etc.).
        total_timesteps (int): Pasos totales de entrenamiento.
        tensorboard_log (str, optional): Ruta para logs de TensorBoard.
        trial (optuna.trial.Trial, optional): Objeto Trial de Optuna para habilitar Pruning.

    Returns:
        tuple: (modelo_entrenado, historial_recompensas, historial_exitos)
    """

    # Configuración del agente con los hiperparámetros dados
    model = DQN(
        "MlpPolicy",
        env,
        exploration_fraction=0.6,
        exploration_final_eps=0.05,
        learning_rate=hyperparams.get('learning_rate', 1e-3),
        batch_size=hyperparams.get('batch_size', 64),
        gamma=hyperparams.get('gamma', 0.995),
        train_freq=hyperparams.get('train_freq', 4),
        buffer_size=hyperparams.get('buffer_size', 1000000),
        learning_starts=hyperparams.get('learning_starts', 2000),
        target_update_interval=hyperparams.get('target_update_interval', 10000),
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    callbacks = []
    eval_env = None

    # Configuración crítica para Optuna: Evaluación periódica y Poda (Pruning)
    if trial is not None:
        # Se crea un entorno separado para evaluación para no interferir con el entrenamiento
        eval_env = make_env("CubeSatDetumblingEnv", seed=42, monitor_path=None)
        
        eval_callback = TrialEvalCallback(
            eval_env=eval_env, 
            trial=trial,
            n_eval_episodes=10,
            eval_freq=5000,
        )
        callbacks.append(eval_callback)

    try:
        # Inicio del bucle de entrenamiento
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except optuna.exceptions.TrialPruned:
        # Relanzar la excepción para que Optuna maneje el pruning
        raise optuna.exceptions.TrialPruned()
    finally:
        # Asegurar siempre el cierre del entorno de evaluación para liberar memoria
        if eval_env is not None:
            eval_env.close()

    # Evaluación final post-entrenamiento para métricas de retorno
    try:
        ep_rewards, ep_lengths = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
        reward_history = list(ep_rewards)
        success_history = [1 if r >= 100 else 0 for r in reward_history]
    except Exception:
        reward_history = []
        success_history = []

    return model, reward_history, success_history

def save_plot_rewards_per_episode(reward_history, hyperparameters, save_dir):
    """
    Grafica la recompensa cruda obtenida en cada episodio sin suavizado.

    Args:
        reward_history (list): Historial de recompensas por episodio.
        hyperparameters (dict): Configuración usada (para nombrar el archivo).
        save_dir (str): Carpeta de destino.
    """
    if not reward_history:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Recompensa por Episodio", color='tab:blue')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa por Episodio durante el Entrenamiento')
    plt.grid(True)
    plt.legend()

    filename = f"recompensa_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"  
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def save_plot_success_rate(success_history, hyperparameters, save_dir):
    """
    Grafica la tasa de éxito acumulada durante el entrenamiento y guarda la imagen.
    
    Args:
        success_history (list): Lista de 1s y 0s indicando si un episodio fue exitoso (1) o no (0).
        hyperparameters (dict): Diccionario con los mejores hiperparámetros.
        save_dir (str): Directorio donde guardar la imagen.
    """    
    if not success_history:
        return
    
    # Cálculo de la media acumulativa: suma progresiva dividida por el número de intentos hasta ese punto
    success_rate = np.cumsum(success_history) / np.arange(1, len(success_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(success_rate, label="Tasa de Éxito Acumulada", color='tab:green')
    plt.xlabel('Episodio')
    plt.ylabel('Tasa de Éxito')
    plt.title('Tasa de Éxito Acumulada durante el Entrenamiento')
    plt.grid(True)
    plt.legend()

    filename = f"tasa_exito_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def save_plot_average_reward_per_episode(reward_history, hyperparameters, save_dir, window_size=50):
    """
    Grafica la recompensa media por episodio con una ventana deslizante para suavizar el gráfico y guarda la imagen.
    
    Args:
        reward_history (list): Lista de recompensas acumuladas por episodio.
        hyperparameters (dict): Diccionario con los mejores hiperparámetros.
        save_dir (str): Directorio donde guardar la imagen.
        window_size (int): Tamaño de la ventana para la media móvil.
    """
    if not reward_history:
        return
    
    # Convolución para la media móvil: ('valid' recorta los bordes donde no hay datos suficientes)
    smoothed_rewards = np.convolve(reward_history, np.ones(window_size) / window_size, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f'Recompensa Media ({window_size} episodios)', color='tab:orange')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Media')
    plt.title(f'Recompensa Media por Episodio (Ventana {window_size})')
    plt.grid(True)
    plt.legend()

    filename = f"recompensa_media_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def save_plot_combined_rewards(reward_history, hyperparameters, save_dir, window_size=50):
    """
    Superpone la recompensa cruda (ruido) con la media móvil (tendencia) en un solo gráfico.
    Es la visualización más completa para evaluar estabilidad y aprendizaje.
    """
    if not reward_history:
        return
    
    if len(reward_history) < window_size:
        window_size = max(1, len(reward_history))
    
    smoothed_rewards = np.convolve(reward_history, np.ones(window_size) / window_size, mode='valid')
    
    plt.figure(figsize=(10, 6))
    
    # 1. Graficar datos crudos (alpha bajo para transparencia)
    plt.plot(reward_history, label="Recompensa cruda", color='lightblue', alpha=0.6)
    
    # 2. Graficar media móvil (linea sólida y oscura)
    # IMPORTANTE: Desplazamos el eje X para alinear la media con el final de la ventana
    plt.plot(range(window_size-1, len(reward_history)), smoothed_rewards, 
             label=f'Media Móvil ({window_size} ep)', color='tab:blue', linewidth=2)
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Evolución del Entrenamiento: Cruda vs Tendencia')
    plt.grid(True, alpha=0.3)
    plt.legend()

    filename = f"recompensa_combinada_lr{hyperparameters['learning_rate']}_bs{hyperparameters['batch_size']}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def train_dqn(cfg, best_params):
    """
    Orquesta el entrenamiento final del agente DQN utilizando los mejores hiperparámetros encontrados.

    Se encarga de crear directorios, configurar semillas, iniciar el entorno,
    ejecutar el entrenamiento y guardar tanto el modelo final como las gráficas de rendimiento.

    Args:
        cfg (dict): Configuración general (rutas, semillas, timesteps, IDs).
        best_params (dict): Mejores hiperparámetros obtenidos por Optuna.

    Returns:
        str: Ruta del archivo donde se guardó el modelo entrenado (.zip).
    """
    # 1. Preparación del sistema de archivos
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # Generar ID único para esta ejecución (incluye timestamp para evitar sobrescrituras)
    run_id = f"{cfg['model_name']}_{cfg['env_id']}_seed{cfg['seed']}_{int(time.time())}"
    run_path = os.path.join(cfg['log_dir'], run_id)
    os.makedirs(run_path, exist_ok=True)

    monitor_path = os.path.join(run_path, "monitor.csv")

    # 2. Reproducibilidad: Semilla global (SB3 + numpy + random)
    set_random_seed(cfg['seed'])

    # 3. Creación del entorno (vectorizado para SB3)
    env = make_env(cfg['env_id'], cfg['seed'], monitor_path)

    # 4. Ejecución del entrenamiento
    # Se llama a la función helper que contiene la lógica del bucle principal
    model, reward_history, success_history = train_dqn_return_model(
        env, best_params, total_timesteps=cfg['total_timesteps'], tensorboard_log=None
    )
    
    # 5. Persistencia
    model_path = os.path.join(cfg['save_dir'], f"{run_id}.zip")
    model.save(model_path)
    env.close() # Importante cerrar para liberar el puerto de comunicación o recursos gráficos

    # 6. Generación de reportes visuales
    save_plot_rewards_per_episode(reward_history, best_params, cfg['save_dir'])
    save_plot_success_rate(success_history, best_params, cfg['save_dir'])
    save_plot_average_reward_per_episode(reward_history, best_params, cfg['save_dir'], window_size=50)
    save_plot_combined_rewards(reward_history, best_params, cfg['save_dir'], window_size=50)

    return model_path

def evaluate_model(
    model_path: str,
    env_id: str,
    n_eval_episodes: int = 10,
    deterministic: bool = True
) -> Tuple[float, float, float]:
    """
    Carga un modelo DQN entrenado y evalúa su rendimiento en un entorno nuevo.

    Esta función crea una instancia fresca del entorno, carga los pesos del modelo
    desde el disco y ejecuta una política (usualmente determinista) para calcular
    la recompensa promedio y su desviación estándar.

    Args:
        model_path (str): Ruta al archivo .zip del modelo guardado.
        env_id (str): Identificador del entorno (ej. 'CartPole-v1' o tu ID personalizado).
        n_eval_episodes (int, opcional): Número de episodios de prueba para promediar. 
                                         Por defecto es 10.
        deterministic (bool, opcional): Si True, usa acciones deterministas (la mejor predicción).
                                        Si False, usa acciones estocásticas. Por defecto True.

    Returns:
        Tuple[float, float]: Una tupla conteniendo:
            - mean_reward (float): La recompensa media acumulada por episodio.
            - std_reward (float): La desviación estándar de la recompensa.
    
    Example:
        >>> mean, std = evaluate_model("dqn_lunar", "LunarLander-v2", n_eval_episodes=20)
        >>> print(f"Resultado: {mean}")
    """
    
    # 1. Cargar el modelo
    # Nota: Si usas otro algoritmo (PPO, A2C), cambia DQN por la clase correspondiente
    # o pasa la clase como argumento.
    try:
        model = DQN.load(model_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {model_path}")
        return 0.0, 0.0

    # 2. Crear entorno de evaluación
    # Es crucial que este entorno esté 'limpio' y tenga la misma config que el de entrenamiento
    try:
        eval_env = make_env(env_id, seed=0, monitor_path=None)
    except NameError:
        raise NameError("La función 'make_env' no está definida. Asegúrate de importarla.")

    # 3. Evaluar
    # evaluate_policy se encarga de resetear el env y acumular rewards
    ep_rewards, ep_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        return_episode_rewards=True
    )

    # Calcular métricas
    mean_reward = np.mean(ep_rewards) if ep_rewards else 0.0
    std_reward = np.std(ep_rewards) if ep_rewards else 0.0
    success_rate = sum(1 for r in ep_rewards if r >= 100) / len(ep_rewards) if ep_rewards else 0.0

    # 4. Limpieza
    try:
        eval_env.close()
    except Exception as e:
        print(f"Advertencia al cerrar el entorno: {e}")

    print(f"Evaluación del modelo '{model_path}':")
    print(f"  -> Recompensa Media: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  -> Tasa de Éxito: {success_rate:.2%}")

    return mean_reward, std_reward, success_rate

def objective(trial: optuna.trial.Trial, cfg: Dict[str, Any]) -> float:
    """
    Función objetivo para optimización de hiperparámetros con Optuna y Pruning.

    Esta función define una sola "prueba" (trial) en el proceso de optimización.
    Sugiere hiperparámetros, entrena un modelo DQN, y reporta el progreso a Optuna
    para permitir la terminación temprana (pruning) si el modelo no promete.

    Args:
        trial (optuna.trial.Trial): Objeto de prueba de Optuna. Se usa para sugerir
                                    hiperparámetros y reportar métricas intermedias.
        cfg (Dict[str, Any]): Diccionario de configuración general. Debe contener:
                              - 'save_dir': Directorio base para guardar resultados.
                              - 'env_id': ID del entorno de Gym.
                              - 'trial_timesteps': Pasos totales de entrenamiento por trial.

    Returns:
        float: La métrica objetiva a maximizar (en este caso, mean_reward).
               Retorna -inf si ocurre un error no controlado.

    Raises:
        optuna.exceptions.TrialPruned: Si Optuna decide cancelar el trial prematuramente.
    """
    global best_reward  # Variable global para rastrear el mejor modelo entre todos los trials

    # 1. Configurar directorios únicos para este trial
    trial_dir = os.path.join(cfg['save_dir'], f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # 2. Sugerir Hiperparámetros (Search Space)
    # Optuna elige valores dentro de estos rangos inteligentemente
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_float('gamma', 0.90, 0.9999)

    target_update_interval = trial.suggest_categorical('target_update_interval', [500, 1000, 2000])
    learning_starts = trial.suggest_categorical('learning_starts', [100, 1000])

    hyperparams = {
        'learning_rate': learning_rate, 
        'batch_size': batch_size, 
        'gamma': gamma,
        'target_update_interval': target_update_interval,
        'learning_starts': learning_starts,
        'buffer_size': 50000,
    }

    # 3. Crear entorno temporal
    # Es vital usar seed=trial.number para reproducibilidad dentro del mismo estudio
    trial_env = make_env(cfg['env_id'], seed=trial.number, monitor_path=None)

    try:
        # 4. Entrenar con soporte para Pruning
        # trial_timesteps define qué tan largo es el entrenamiento de prueba
        t_steps = int(cfg.get('trial_timesteps', 5000))
        
        # IMPORTANTE: 'train_dqn_return_model' debe implementar el callback de Optuna
        # internamente usando el objeto 'trial' que le pasamos.
        model, reward_history, success_history = train_dqn_return_model(
            trial_env, 
            hyperparams, 
            total_timesteps=t_steps, 
            tensorboard_log=None,
            trial=trial  # <--- CRÍTICO: Permite reportar rewards paso a paso
        )

        # 5. Calcular métrica final del trial
        # Usamos el promedio de recompensas históricas como métrica de éxito
        mean_reward = np.mean(reward_history) if reward_history else -float('inf')
        
        print(f"Trial {trial.number} finalizado. Mean Reward: {mean_reward:.2f}")

        # 6. Guardar el modelo SOLO si es el mejor globalmente
        if mean_reward > best_reward:
            best_reward = mean_reward
            mp = os.path.join(trial_dir, f"best_model_trial_{trial.number}.zip")
            model.save(mp)
            print(f"¡Nuevo mejor modelo global detectado en Trial {trial.number}!")

        # 7. Generar y guardar reportes gráficos
        # Asumimos que estas funciones guardan .png en trial_dir
        save_plot_rewards_per_episode(reward_history, hyperparams, trial_dir)
        save_plot_combined_rewards(reward_history, hyperparams, trial_dir, window_size=50)
        save_plot_average_reward_per_episode(reward_history, hyperparams, trial_dir)

    except optuna.exceptions.TrialPruned:
        # El Pruning lanza esta excepción para detener el flujo.
        # Debemos atraparla para cerrar el entorno correctamente antes de salir.
        print(f"Trial {trial.number} podado (Pruned) por bajo rendimiento.")
        trial_env.close()
        # Re-lanzamos la excepción para que Optuna sepa que fue podado y no un error.
        raise optuna.exceptions.TrialPruned()

    except Exception as e:
        # Captura cualquier otro error (memoria, sintaxis, env crash)
        print(f"Error inesperado en trial {trial.number}: {e}")
        trial_env.close()
        return -float('inf') # Retornamos un valor muy bajo para que Optuna evite estos params

    # Limpieza final exitosa
    trial_env.close()
    
    return mean_reward

def optimize_hyperparameters(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configura y ejecuta la optimización de hiperparámetros con Optuna usando MedianPruner.

    Esta función inicializa un estudio de Optuna persistente (guardado en SQLite),
    define la estrategia de poda (Pruning) para descartar trials ineficientes,
    y ejecuta el bucle de optimización llamando a la función `objective`.

    Args:
        cfg (Dict[str, Any]): Diccionario de configuración. Claves requeridas:
            - 'save_dir': Ruta donde se guardará la base de datos .db.
            - 'n_trials': (Opcional) Número total de pruebas a ejecutar. Por defecto 15.

    Returns:
        Dict[str, Any]: Un diccionario con los mejores hiperparámetros encontrados.
                        Ej: {'learning_rate': 0.001, 'gamma': 0.99, ...}
    
    Notes:
        - Utiliza 'MedianPruner': Detiene un trial si su rendimiento es peor que la 
          mediana de los trials anteriores en el mismo paso.
        - Persistencia: Si se interrumpe el script, se puede volver a ejecutar y 
          Optuna retomará el estudio donde lo dejó gracias a `load_if_exists=True`.
    """
    
    # 1. Asegurar que el directorio de guardado existe
    os.makedirs(cfg['save_dir'], exist_ok=True)
    storage_path = os.path.join(cfg['save_dir'], "optuna_dqn.db")
    
    # Cadena de conexión para SQLite (base de datos local en archivo)
    storage_url = f"sqlite:///{storage_path}"

    # 2. Definir el Pruner (Estrategia de poda)
    # MedianPruner es excelente para RL porque no requiere configuración compleja.
    # - n_startup_trials=5: Deja correr 5 pruebas completas antes de empezar a podar nada.
    # - n_warmup_steps=1000: En cada trial, deja correr los primeros 1000 pasos sin podar 
    #   (el agente es tonto al inicio, hay que darle tiempo).
    pruner = MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=1000, 
        interval_steps=1000  # Chequea si podar cada 1000 pasos
    )

    # 3. Crear o cargar el estudio
    print(f"-> Conectando a base de datos en: {storage_url}")
    study = optuna.create_study(
        direction='maximize',       # Buscamos maximizar la recompensa
        storage=storage_url,        # Persistencia en disco
        study_name="dqn_study",     # Nombre único del estudio
        load_if_exists=True,        # Retomar si ya existe
        pruner=pruner
    )

    # 4. Ejecutar optimización
    n_trials = cfg.get('n_trials', 15)
    print(f"-> Iniciando optimización por {n_trials} trials...")
    
    try:
        # Usamos lambda para pasar 'cfg' a la función objective
        study.optimize(lambda trial: objective(trial, cfg), n_trials=n_trials)
    except KeyboardInterrupt:
        print("\nOptimización interrumpida por el usuario. Guardando progreso...")
    except Exception as e:
        print(f"Error crítico durante la optimización: {e}")

    # 5. Reportar resultados
    print("\n" + "="*40)
    print(f"RESULTADOS DEL ESTUDIO ({len(study.trials)} trials completados)")
    print("="*40)
    
    if len(study.trials) > 0:
        print(f"Mejor Trial ID: {study.best_trial.number}")
        print(f"Mejor Valor (Reward): {study.best_value:.4f}")
        print("Mejores Hiperparámetros:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        
        return study.best_params
    else:
        print("No se completó ningún trial exitosamente.")
        return {}

def main():
    """
    Función principal que orquesta el flujo de trabajo completo de RL.

    El flujo de ejecución es:
    1. Configuración del entorno y parámetros generales.
    2. Optimización de hiperparámetros (Optuna) para encontrar la mejor configuración.
    3. Entrenamiento del modelo final usando los mejores hiperparámetros encontrados.
    4. Evaluación del modelo final en un entorno limpio.
    """
    
    # 0. Imprimir información de contexto (Opcional, para registro)
    # Asegúrate de que SatellitePersonality esté importado o definido
    try:
        print(f"--- Iniciando misión ---")
        print(f"Nombre del satélite: {SatellitePersonality.SATELLITE_NAME}")
        print(f"Ubicación: Lat={SatellitePersonality.OBSERVER_LATITUDE}, Lon={SatellitePersonality.OBSERVER_LONGITUDE}")
    except NameError:
        print("Aviso: 'SatellitePersonality' no está definido, saltando info de cabecera.")

    # # 1. Configuración Centralizada (Diccionario maestro)
    # # Es buena práctica tener todo aquí para no tocar código interno después.
    cfg = {
        'env_id': 'CubeSatDetumblingEnv', 
        'total_timesteps': 500_000,    # Entrenamiento final largo (ej. 200k pasos)
        'trial_timesteps': 30_000,     # Entrenamiento corto para pruebas de Optuna (ej. 10k pasos)
        'n_trials': 20,                # Cuántas pruebas hará Optuna
        'seed': 123,                   # Semilla para reproducibilidad
        'log_dir': 'logs',             # Carpeta para TensorBoard
        'save_dir': 'models',          # Carpeta para guardar .zip y .db
        'model_name': 'dqn_satellite_final2',
    }

    # Asegurar que los directorios existen antes de empezar
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    print("\n" + "="*50)
    print("FASE 1: Optimización de hiperparámetros (Optuna)")
    print("="*50)
    
    # 2. Ejecutar Optimización
    # Esto busca la mejor combinación de learning_rate, gamma, etc.
    best_params = optimize_hyperparameters(cfg)

    # Guardar mejores hiperparámetros en JSON
    hyperparams_path = os.path.join(cfg['save_dir'], 'best_hyperparams2.json')
    with open(hyperparams_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Mejores hiperparámetros guardados en: {hyperparams_path}")

    print("\n" + "="*50)
    print("FASE 2: Entrenamiento del modelo final")
    print("="*50)
    print(f"Entrenando con los mejores parámetros: {best_params}")

    # 3. Entrenamiento Final
    # Entrenamos el modelo definitivo usando 'best_params' y 'total_timesteps'
    # Nota: train_dqn debe estar adaptada para recibir (cfg, hyperparameters)
    model_path = train_dqn(cfg, best_params)

    print("\n" + "="*50)
    print("FASE 3: Evaluación y validación")
    print("="*50)

    # 4. Evaluar el modelo entrenado
    # Cargamos el modelo guardado y probamos 10 episodios para ver la métrica final
    mean_reward, std_reward, success_rate = evaluate_model(
        model_path=model_path,
        env_id=cfg['env_id'],
        n_eval_episodes=50,
        deterministic=True # Importante para evaluación final
    )

    print(f"\nResultados finales del proyecto:")
    print(f"-> Modelo guardado en: {model_path}")
    print(f"-> Performance Final: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"-> Tasa de Éxito: {success_rate:.2%}")

if __name__ == "__main__":
    main()