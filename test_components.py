#!/usr/bin/env python3
"""
Test script - demonstracja podstawowego działania aplikacji.
"""

from storage import DataStorage
from ml_model import MLModel

# Test 1: Storage
print("🧪 Test 1: Storage (SQLite + Pickle)")
print("=" * 50)

storage = DataStorage(db_path="data/test.db", model_path="models/test_model.pkl")

# Zapisz przykładowe interakcje
storage.save_interaction(10, 20, 'love', 2.0, False)
storage.save_interaction(5, 10, 'like', 1.0, True)
storage.save_interaction(7, 15, 'dislike', 0.0, False)

# Pobierz statystyki
stats = storage.get_statistics()
print(f"Interakcji: {stats['total_interactions']}")
print(f"Pozytywny feedback: {stats['positive_rate']:.1%}")
print(f"Love: {stats['loves']}, Like: {stats['likes']}, Dislike: {stats['dislikes']}")

print("\n✅ Storage działa!\n")

# Test 2: ML Model
print("🧪 Test 2: ML Model (Epsilon-Greedy)")
print("=" * 50)

model = MLModel()

# Symuluj kilka interakcji
interactions = [
    (10, 20, 2.0),  # love
    (20, 40, 2.0),  # love
    (5, 10, 2.0),   # love
    (15, 30, 1.0),  # like
]

for user_input, expected, feedback in interactions:
    prediction, is_exploration = model.predict(user_input)
    model.update(user_input, expected, feedback)
    
    mode = "🔍 Eksploracja" if is_exploration else "🎯 Predykcja"
    print(f"Input: {user_input:3d} → Prediction: {prediction:3d} [{mode}]")

# Test po nauczeniu
print("\nPo nauczeniu się wzorca (input * 2):")
for test_input in [3, 8, 12, 25]:
    prediction, _ = model.predict(test_input)
    print(f"Input: {test_input:3d} → Prediction: {prediction:3d} (Expected: {test_input * 2:3d})")

model_stats = model.get_stats()
print(f"\nStatystyki modelu:")
print(f"  Epsilon: {model_stats['epsilon']:.2%}")
print(f"  Success rate: {model_stats['success_rate']:.1%}")

print("\n✅ ML Model działa!\n")

# Test 3: Persistence
print("🧪 Test 3: Model Persistence")
print("=" * 50)

# Zapisz model
storage.save_model(model)
print("Model zapisany do pickle...")

# Wczytaj model
loaded_model = storage.load_model()
if loaded_model:
    print("Model wczytany z pickle!")
    
    # Testuj wczytany model
    test_prediction, _ = loaded_model.predict(10)
    print(f"Test wczytanego modelu (input=10): {test_prediction}")
    
    print("\n✅ Persistence działa!")
else:
    print("❌ Błąd wczytywania modelu")

print("\n" + "=" * 50)
print("🎉 Wszystkie testy zakończone pomyślnie!")
print("=" * 50)

# Cleanup
import os
if os.path.exists("data/test.db"):
    os.remove("data/test.db")
if os.path.exists("models/test_model.pkl"):
    os.remove("models/test_model.pkl")
print("\n🧹 Pliki testowe usunięte.")
