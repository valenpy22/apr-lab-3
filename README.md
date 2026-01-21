# Instrucciones para ejecución del código para Laboratorio 2
## 1. Clonar el repositorio
```bash
git clone https://github.com/valenpy22/apr-lab-2.git
cd apr-lab-2
```

## 2. Crear un entorno conda
```bash
conda create --name nuevo_entorno python=3.10
conda activate nuevo_entorno
```

## 3. Instalar las dependencias
Se recomienda instalar las librerías base con Conda y las específicas con Pip para evitar conflictos.
```bash
conda install -c conda-forge stable-baselines3 gymnasium optuna matplotlib pyzmq python-dotenv pyIGRF pymongo pillow tensorboard
pip install pybamm skyfield numpy scipy gym gymnasium joblib
```

## 4. Usar las variables de entorno
```bash
chmod +x set_env.sh
source set_env.sh
```

## 5. Ejecutar el código
El script run_loop.sh está diseñado para reiniciar el entrenamiento automáticamente en caso de fallo. Para detener la ejecución, presiona Ctrl + C en la terminal.
```bash
chmod +x run_loop.sh
./run_loop.sh

```

## 6. Monitoreo del entrenamiento
```bash
Using cpu device
-----------------------------------------
| time/                   |             |
|    fps                  | 1335        |
|    iterations           | 25          |
|    time_elapsed         | 38          |
|    total_timesteps      | 51200       |
| train/                  |             |
|    approx_kl            | 0.004337039 |
|    clip_fraction        | 0.0359      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.49       |
|    explained_variance   | 0.0798254   |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0162      |
|    n_updates            | 240         |
|    policy_gradient_loss | -0.00465    |
|    value_loss           | 0.0179      |
-----------------------------------------

```

## 7. Guardado de modelos
El modelo entrenado se guardará automáticamente en la carpeta models en formato .zip. Puedes cargar el modelo más tarde usando Stable Baselines3 para hacer más evaluaciones o continuar el entrenamiento.
