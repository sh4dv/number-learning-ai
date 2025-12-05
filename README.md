# 🤖 Number Learning AI

# Warning: this whole app is currently just in polish language.

Interaktywna aplikacja konsolowa wykorzystująca Machine Learning, która uczy się mapowania wejście → wyjście na podstawie Twojego feedbacku. Podajesz liczbę, AI odpowiada, oceniasz wynik — a model uczy się i z czasem trafia coraz lepiej.

## 📖 Opis

Jak działa program:
1. Podaj dodatnią liczbę całkowitą.
2. AI generuje odpowiedź (liczbę wyjściową).
3. Oceń ją: `dislike` (źle) lub `like` (idealnie!).
4. AI zapisuje interakcję i uczy się z Twojego feedbacku.

## 🎓 Jak model się uczy

1. Eksploracja
Program losuje wyniki, a uzytkownik je poprawia
2. Model dopasuje wzór
LinearRegression korzysta z:
y = a0 + a1 * x + a2 * x^2 + a3 * x^2
Współczynniki a0..a3 są uczone.

Wbudowane tryby pracy:
- Tryb standardowy (interaktywny) — oceniasz odpowiedzi AI.
- `train` — podajesz pary INPUT→OUTPUT i uczysz wzorzec wprost.
- `auto_train` — AI samo generuje dane według zadanego wzorca (np. `*2`, `+100`, `x*2+1`).
- `testing_model` — automatyczne testowanie accuracy dla wzorca.

Przykładowe wzorce, których model może się nauczyć perfekcyjnie (przy spójnych danych):
- Mnożenie: `x*2`, `x*3`, `x*10`
- Dodawanie: `x+100`, `x+1`
- Potęgi (do 3. stopnia): `x^2`, `x^3`
- Kombinacje liniowo-wielomianowe: `x*2+1`, `x*3+10`, `x^2 + x + 1`

## 🧠 Algorytm i technologia

- Epsilon-Greedy: z prawdopodobieństwem ε model eksploruje (losowy, ale sensowny output), a z 1−ε korzysta z predykcji modelu.
- Model ML: Pipeline `PolynomialFeatures(degree=3)` + `LinearRegression` (scikit-learn). Pozwala modelować funkcje do 3. stopnia: y = a₀ + a₁x + a₂x² + a₃x³.
- Trening: batch retrain na podstawie pozytywnych przykładów (feedback > 0), wywoływany w trybach treningowych oraz okresowo.
- Persistencja: dane interakcji w SQLite, model ML w pliku pickle.

Domyślne parametry eksploracji (`ml_model.py`):
- `epsilon_start = 0.4`
- `epsilon_min = 0.03`
- `epsilon_decay = 0.92`

Strategia eksploracji (`ml_model.py`):
- Wielokrotności inputu (np. ×2, ×3, ×5, ×10)
- Wartości „blisko” inputu (50%–300%)
- Przesunięcia stałe (±10 … ±1000)
- Losowo z inteligentnego zakresu [x/2, x×50]

## 🧩 Architektura

- `main.py` — orchestrator: pętla interakcji, komendy, przepływ danych.
- `ml_model.py` — MLModel: epsilon-greedy, predykcja, batch retrain (Polynomial Regression).
- `storage.py` — DataStorage: SQLite (`data/interactions.db`), pickle (`models/ml_model.pkl`), eksport do CSV, zarządzanie modelami.
- `statistics.py` — metryki sesji, krzywa uczenia, trend, mini-wykres ASCII.
- `ui.py` — bogaty interfejs konsolowy (Rich): panele, tabele, prompty, progress.

## 🗂️ Struktura projektu

```
test2/
├── main.py           # Główna aplikacja (entry point)
├── ml_model.py       # Model ML (epsilon-greedy + Polynomial Regression)
├── storage.py        # SQLite + pickle, eksport CSV, zarządzanie modelami
├── statistics.py     # Statystyki, trend, krzywa uczenia
├── ui.py             # Interfejs (Rich)
├── requirements.txt  # Zależności
├── README.md         # Dokumentacja
├── data/
│   ├── interactions.db  # Baza SQLite (tworzona automatycznie)
│   └── export.csv       # Eksport (na żądanie)
└── models/
  └── ml_model.pkl     # Zapisany model (tworzony automatycznie)
```

## 🚀 Instalacja

Wymagania: Python 3.10+, macOS/Linux/Windows.

```bash
# w katalogu projektu
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux (zsh)
pip install -r requirements.txt
```

## 🎮 Uruchomienie

```bash
python3 main.py
```

Po uruchomieniu naciśnij Enter i wpisuj liczby lub komendy.

## 🧭 Komendy (skrót)

- `train`: tryb ręcznego treningu (podajesz INPUT i idealny OUTPUT).
- `auto_train`: automatyczny trening dla wzorca (np. `*2`, `+100`, `^2`, `%10`, `x*2+10`).
- `testing_model`: automatyczne testowanie accuracy na zadanym wzorcu.
- `history`: ostatnie 10 interakcji z bazy.
- `stats`: szczegółowe statystyki, trend i krzywa uczenia.
- `explain`: wyjaśnienie ostatniej predykcji (tryb, epsilon, pewność).
- `reset`: usuwa dane (model zostaje).
- `retrain`: batch retrain na wszystkich danych pozytywnych.
- `new_model` / `save_model` / `load_model` / `list_models` / `delete_model`: zarządzanie modelami.
- `export`: eksport interakcji do `data/export.csv`.
- `help`, `quit`.

## 🔬 Tryby treningu i testów

### `train`
Podajesz pary INPUT→OUTPUT, które traktowane są jako idealne (`like`). Model szybko uczy się wzorca.

### `auto_train`
Podajesz liczbę przykładów i wzorzec operacji. Obsługiwane formaty:
- `*k`, `/k`, `+k`, `-k`, `^n` lub `**n`, `%m`
- pełne wyrażenia z `x`, np. `x*2+10`, `x*x+1` (symbol `^` zamieniany jest na `**` dla potęgowania)

### `testing_model`
Generuje losowe inputy i porównuje predykcje z wynikiem wzorca. Zapisuje feedback i może dalej douczać model.

## 💾 Dane i trwałość

- Baza: `data/interactions.db` (SQLite) — wszystkie interakcje z feedbackiem.
- Model: `models/ml_model.pkl` (pickle) — automatycznie wczytywany przy starcie i zapisywany okresowo/przy wyjściu.
- Eksport: `export` tworzy `data/export.csv` z pełną historią.

## 🛠️ Konfiguracja (kluczowe fragmenty)

`MLModel.__init__` (domyślnie):

```python
MLModel(
  epsilon_start=0.4,
  epsilon_min=0.03,
  epsilon_decay=0.92,
  output_range=(1, 100000)
)
```

Model: `Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])`.

Retrain: `batch_retrain()` trenuje wyłącznie na przykładach z feedbackiem > 0 (like).

## 📦 Zależności

- `scikit-learn` — PolynomialFeatures + LinearRegression
- `numpy` — obliczenia numeryczne
- `rich` — kolorowy UI w terminalu

Instalacja: `pip install -r requirements.txt`.

## 🐛 Troubleshooting

- Brak `sklearn` lub `rich`: `pip install -r requirements.txt`
- Problem z bazą: usuń `data/interactions.db` i uruchom ponownie.
- Uprawnienia do zapisu modeli: `chmod -R 755 models/` (macOS/Linux).

## 📌 Uwagi

- Model jest najlepszy, gdy uczy się jednego spójnego wzorca. Jeśli chcesz zmienić cel — rozważ `reset` i ponowny trening.
- Potęgi większe niż 3 mogą nie być dokładnie odwzorowane przez regresję 3. stopnia.

## 📄 Licencja

Dowolne użycie/modyfikacja/rozpowszechnianie — projekt edukacyjny.

---

Miłego trenowania! 🚀
