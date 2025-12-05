#!/usr/bin/env python3
"""Test epsilon decay - ile interakcji do eksploatacji."""

from ml_model import MLModel

m = MLModel()
print('Parametry modelu:')
print(f'Epsilon początkowe: {m.epsilon:.1%}')
print(f'Epsilon minimalne: {m.epsilon_min:.1%}')
print(f'Epsilon decay: {m.epsilon_decay}')
print()

# Symulacja: ile interakcji do epsilon < 50%
epsilon = m.epsilon
interactions = 0
while epsilon >= 0.5:
    epsilon *= m.epsilon_decay
    interactions += 1

print(f'Po {interactions} interakcjach epsilon spadnie poniżej 50%')
print(f'  → ~50% szans na eksplorację, 50% na model')
print()

# Ile do epsilon < 20%
epsilon = m.epsilon
interactions = 0
while epsilon >= 0.2:
    epsilon *= m.epsilon_decay
    interactions += 1

print(f'Po {interactions} interakcjach epsilon spadnie poniżej 20%')
print(f'  → ~20% szans na eksplorację, 80% na model')
print()

# Ile do epsilon < 10%
epsilon = m.epsilon
interactions = 0
while epsilon >= 0.1:
    epsilon *= m.epsilon_decay
    interactions += 1

print(f'Po {interactions} interakcjach epsilon spadnie poniżej 10%')
print(f'  → ~10% szans na eksplorację, 90% na model')
print()

# Ile do minimum
epsilon = m.epsilon
interactions = 0
while epsilon > m.epsilon_min:
    epsilon *= m.epsilon_decay
    interactions += 1

print(f'Po {interactions} interakcjach epsilon osiągnie minimum ({m.epsilon_min:.1%})')
print(f'  → {m.epsilon_min:.1%} szans na eksplorację, {100-m.epsilon_min*100:.1f}% na model')
print()

# Tabela pokazująca epsilon po kolejnych interakcjach
print('Szczegółowa tabela epsilon:')
print('Interakcje | Epsilon | Szansa na model')
print('-' * 40)
epsilon = m.epsilon
for i in [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50]:
    epsilon = m.epsilon * (m.epsilon_decay ** i)
    epsilon = max(m.epsilon_min, epsilon)
    model_chance = 100 - epsilon * 100
    print(f'{i:10d} | {epsilon:6.1%} | {model_chance:6.1f}%')
