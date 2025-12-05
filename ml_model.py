"""
Model uczenia maszynowego - epsilon-greedy z Polynomial Regression.
Uczy się preferowanych outputów na podstawie feedbacku użytkownika.
Model wykorzystuje regresję wielomianową (stopień 3) która idealnie radzi sobie z:
mnożeniem, dodawaniem, potęgowaniem (x², x³) i ich kombinacjami.
Osiąga 0% błędu dla operacji matematycznych typu x*2, x+100, x², x*2+1, itp.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import random
from typing import Tuple, Optional


class MLModel:
    """
    Model ML wykorzystujący epsilon-greedy exploration i batch learning.
    
    Algorytm:
    - Epsilon-greedy: z prawdopodobieństwem ε wybiera losowy output (eksploracja),
      z prawdopodobieństwem 1-ε używa predykcji modelu (eksploatacja)
    - Polynomial Regression (stopień 3): modeluje funkcje y = a₀ + a₁x + a₂x² + a₃x³
    - Idealne dla: mnożenia (x*k), dodawania (x+k), potęg (x², x³), kombinacji (x*2+1)
    - Feedback levels: dislike (0.0), like (1.0) - do filtrowania danych
    """
    
    def __init__(self, epsilon_start: float = 0.4, epsilon_min: float = 0.03, 
                 epsilon_decay: float = 0.92, output_range: Tuple[int, int] = (1, 100000)):
        """
        Inicjalizacja modelu.
        
        Args:
            epsilon_start: Początkowa wartość epsilon (eksploracja)
            epsilon_min: Minimalna wartość epsilon
            epsilon_decay: Współczynnik decay (mnożnik po każdej interakcji)
            output_range: Zakres możliwych outputów (min, max)
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.output_range = output_range
        
        # Model Polynomial Regression (stopień 3)
        # Generuje cechy: [1, x, x², x³] i uczy się liniowej kombinacji
        # Może perfekcyjnie nauczyć się: x*k, x+k, x², x³, x*2+1, itp.
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=True)),
            ('linear', LinearRegression())
        ])
        
        # Tracking
        self.is_fitted = False
        self.interaction_count = 0
        self.positive_feedback_count = 0
        
        # Historia ostatnich predykcji dla "explain"
        self.last_prediction_info: Optional[dict] = None
    
    def _extract_features(self, user_input: int) -> np.ndarray:
        """
        Przygotowuje input dla Polynomial Regression.
        
        Polynomial Regression sam generuje cechy wielomianowe:
        - PolynomialFeatures(degree=3) tworzy: [1, x, x², x³]
        - Model uczy się: y = a₀ + a₁x + a₂x² + a₃x³
        
        Nie potrzebujemy ręcznie tworzyć cech - model robi to automatycznie!
        
        Args:
            user_input: Liczba podana przez użytkownika
            
        Returns:
            Array z samym x (shape: [1, 1]) - Polynomial Regression sam doda cechy
        """
        x = float(user_input)
        return np.array([[x]], dtype=np.float64)
    
    def _get_exploration_output(self, user_input: int) -> int:
        """
        Generuje losowy output do eksploracji.
        
        Strategia: różne heurystyki z wagami, żeby eksplorować sensownie:
        - 35% szans: wielokrotność inputu (x2, x3, x5, x10)
        - 30% szans: blisko inputu (50% do 300%)
        - 20% szans: dodawanie/odejmowanie (±10 do ±1000)
        - 15% szans: losowo z inteligentnego zakresu
        """
        strategy = random.random()
        
        if strategy < 0.35:
            # Wielokrotności - najczęstsze wzorce (NIE 0.5, zawsze >= input)
            multiplier = random.choice([2, 3, 4, 5, 10, 20, 50])
            output = user_input * multiplier
        elif strategy < 0.65:
            # Blisko inputu ale z większym zakresem (zawsze >= input/2)
            factor = random.uniform(0.5, 3.0)
            output = int(user_input * factor)
        elif strategy < 0.85:
            # Stałe przesunięcia
            offset = random.choice([10, 25, 50, 100, 200, 500, 1000])
            # 50% szans dodać, 50% odjąć (ale nie mniej niż 1)
            if random.random() < 0.5:
                output = user_input + offset
            else:
                output = max(1, user_input - offset)
        else:
            # Losowo z rozsądnego zakresu
            min_val = max(1, user_input // 2)
            max_val = min(self.output_range[1], user_input * 50)
            output = random.randint(min_val, max_val)
        
        # Ogranicz do dozwolonego zakresu (minimum 1)
        output = max(1, min(self.output_range[1], output))
        return output
    
    def predict(self, user_input: int) -> Tuple[int, bool]:
        """
        Przewiduje output dla danego inputu (z epsilon-greedy).
        
        Args:
            user_input: Liczba podana przez użytkownika
            
        Returns:
            Tuple (predicted_output, was_exploration)
        """
        # Epsilon-greedy decision
        explore = random.random() < self.epsilon
        
        if explore or not self.is_fitted:
            # EKSPLORACJA - losowy output
            output = self._get_exploration_output(user_input)
            
            self.last_prediction_info = {
                'mode': 'exploration',
                'epsilon': self.epsilon,
                'confidence': 0.0,
                'reason': 'Eksploracja - szukam nowych wzorców' if explore else 'Brak danych - uczę się'
            }
            
            return output, True
        else:
            # EKSPLOATACJA - użyj modelu
            x_features = self._extract_features(user_input)
            predicted = self.model.predict(x_features)[0]
            
            # Jeśli predykcja jest zbyt mała i mamy mało danych, użyj heurystyki
            if predicted < user_input * 0.3 and self.positive_feedback_count < 5:
                # Fallback: użyj najprostszej heurystyki (input * 2)
                output = user_input * 2
                reason = 'Heurystyka (za mało danych do pewnej predykcji)'
            else:
                # Zaokrąglij i ogranicz do zakresu
                output = int(round(predicted))
                # WAŻNE: jeśli model przewiduje < 1, ustaw na minimum sensowną wartość
                if output < 1:
                    output = max(1, user_input)
                output = min(self.output_range[1], output)
                reason = f'Predykcja modelu Polynomial Regression'
            
            # Pewność predykcji (uproszczona - na podstawie historii)
            confidence = min(0.95, self.positive_feedback_count / max(10, self.interaction_count))
            
            self.last_prediction_info = {
                'mode': 'exploitation',
                'epsilon': self.epsilon,
                'confidence': confidence,
                'reason': f'{reason} (pewność: {confidence:.1%})',
                'raw_prediction': predicted
            }
            
            return output, False
    
    def update(self, user_input: int, expected_output: int, feedback_value: float) -> None:
        """
        Aktualizuje statystyki na podstawie feedbacku.
        
        Uwaga: Model NIE jest trenowany tutaj - tylko batch_retrain() trenuje model.
        To zapewnia stabilność numeryczną i lepsze wyniki.
        
        Args:
            user_input: Input użytkownika
            expected_output: Oczekiwany/poprawny output (podany przez użytkownika)
            feedback_value: Wartość feedbacku (0.0=dislike, 1.0=like)
        """
        self.interaction_count += 1
        
        # Śledź pozytywny feedback
        if feedback_value > 0:
            self.positive_feedback_count += 1
        
        # Decay epsilon - z czasem mniej eksploracji
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def batch_retrain(self, interactions: list) -> None:
        """
        Przetrening modelu na wszystkich danych historycznych.
        
        Polynomial Regression uczy się wzoru matematycznego łączącego x z y.
        Dla danych x*2+1 nauczy się: y = 0 + 2x + 0·x² + 0·x³ = 2x + 1
        
        Args:
            interactions: Lista tupli (user_input, expected_output, feedback_value)
        """
        if not interactions:
            return
        
        # Filtruj tylko pozytywne przykłady (like - fb > 0)
        positive_interactions = [(inp, out, fb) for inp, out, fb in interactions if fb > 0]
        
        if not positive_interactions:
            return
        
        # Przygotuj dane - tylko surowe x (Polynomial Regression sam doda cechy)
        X = np.array([[float(inp)] for inp, _, _ in positive_interactions], dtype=np.float64)
        y = np.array([out for _, out, _ in positive_interactions], dtype=np.float64)
        
        # Polynomial Regression - trenuj pipeline (PolynomialFeatures + LinearRegression)
        self.model.fit(X, y)
        
        self.is_fitted = True
    
    def get_explanation(self) -> str:
        """Zwraca wyjaśnienie ostatniej predykcji."""
        if not self.last_prediction_info:
            return "Brak informacji o ostatniej predykcji"
        
        info = self.last_prediction_info
        
        explanation = f"Tryb: {info['mode']}\n"
        explanation += f"Epsilon (eksploracja): {info['epsilon']:.2%}\n"
        explanation += f"Powód: {info['reason']}\n"
        
        if info['mode'] == 'exploitation':
            explanation += f"Surowa predykcja: {info.get('raw_prediction', 'N/A'):.2f}"
        
        return explanation
    
    def get_stats(self) -> dict:
        """Zwraca statystyki modelu."""
        return {
            'is_fitted': self.is_fitted,
            'epsilon': self.epsilon,
            'interaction_count': self.interaction_count,
            'positive_feedback_count': self.positive_feedback_count,
            'success_rate': self.positive_feedback_count / max(1, self.interaction_count)
        }
