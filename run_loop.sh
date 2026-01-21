#!/bin/bash

while true
do
  echo "🚀 Ejecutando train_deep_rl_dqn.py..."
  python train_deep_rl_dqn.py

  EXIT_CODE=$?
  echo "💥 Script terminó con código $EXIT_CODE"

  echo "⏳ Reiniciando en 3 segundos..."
  sleep 3
done

