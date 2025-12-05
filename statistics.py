"""
Moduł statystyk i analityki - monitoring postępów uczenia.
"""

from datetime import datetime, timedelta
from typing import List, Dict
import math


class Statistics:
    """Zarządza statystykami i metrykami uczenia."""
    
    def __init__(self):
        """Inicjalizacja trackera statystyk."""
        self.session_start = datetime.now()
        self.session_interactions = 0
        self.session_positives = 0
    
    def update_session(self, feedback_value: float) -> None:
        """
        Aktualizuje statystyki sesji.
        
        Args:
            feedback_value: Wartość feedbacku (0.0, 1.0, 2.0)
        """
        self.session_interactions += 1
        if feedback_value > 0:
            self.session_positives += 1
    
    def get_session_stats(self) -> Dict:
        """Zwraca statystyki bieżącej sesji."""
        duration = datetime.now() - self.session_start
        
        return {
            'duration': duration,
            'interactions': self.session_interactions,
            'positive_rate': self.session_positives / max(1, self.session_interactions)
        }
    
    def calculate_learning_curve(self, interactions: List[Dict], window_size: int = 10) -> List[float]:
        """
        Oblicza krzywą uczenia (moving average pozytywnego feedbacku).
        
        Args:
            interactions: Lista interakcji z bazy danych
            window_size: Rozmiar okna dla średniej ruchomej
            
        Returns:
            Lista wartości accuracy w czasie
        """
        if not interactions:
            return []
        
        # Odwróć kolejność (od najstarszych)
        interactions = list(reversed(interactions))
        
        curve = []
        for i in range(len(interactions)):
            # Weź ostatnie window_size interakcji
            start_idx = max(0, i - window_size + 1)
            window = interactions[start_idx:i+1]
            
            # Policz pozytywne
            positives = sum(1 for interaction in window 
                          if interaction['feedback_value'] > 0)
            
            accuracy = positives / len(window)
            curve.append(accuracy)
        
        return curve
    
    def generate_progress_bar(self, value: float, width: int = 20) -> str:
        """
        Generuje tekstowy progress bar.
        
        Args:
            value: Wartość 0.0-1.0
            width: Szerokość w znakach
            
        Returns:
            String z progress barem
        """
        filled = int(value * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {value:.1%}"
    
    def generate_mini_chart(self, values: List[float], height: int = 5, width: int = 40) -> str:
        """
        Generuje mini wykres ASCII krzywej uczenia.
        
        Args:
            values: Lista wartości 0.0-1.0
            height: Wysokość wykresu w liniach
            width: Szerokość wykresu
            
        Returns:
            Multi-line string z wykresem
        """
        if not values:
            return "Brak danych do wykresu"
        
        # Próbkuj wartości jeśli za dużo
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values
        
        # Normalizuj do wysokości
        chart_lines = []
        for row in range(height, 0, -1):
            line = ""
            threshold = row / height
            
            for val in sampled:
                if val >= threshold:
                    line += "▓"
                elif val >= threshold - 0.2:
                    line += "░"
                else:
                    line += " "
            
            # Dodaj oś Y
            y_label = f"{threshold:.1f}"
            chart_lines.append(f"{y_label:>4} │{line}")
        
        # Dodaj oś X
        chart_lines.append("     └" + "─" * len(sampled))
        
        return "\n".join(chart_lines)
    
    def format_duration(self, duration: timedelta) -> str:
        """Formatuje czas trwania czytelnie."""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def get_feedback_distribution(self, stats: Dict) -> str:
        """
        Tworzy wizualizację rozkładu feedbacku.
        
        Args:
            stats: Słownik ze statystykami z DataStorage
            
        Returns:
            Sformatowany string z rozkładem
        """
        total = stats['total_interactions']
        if total == 0:
            return "Brak danych"
        
        loves = stats['loves']
        likes = stats['likes']
        dislikes = stats['dislikes']
        
        # Normalizuj do 100%
        love_pct = loves / total
        like_pct = likes / total
        dislike_pct = dislikes / total
        
        # Bar dla każdego typu
        bar_width = 30
        
        love_bar = '💖' * int(love_pct * bar_width)
        like_bar = '👍' * int(like_pct * bar_width)
        dislike_bar = '👎' * int(dislike_pct * bar_width)
        
        result = f"Love:    {love_bar} {love_pct:.1%} ({loves})\n"
        result += f"Like:    {like_bar} {like_pct:.1%} ({likes})\n"
        result += f"Dislike: {dislike_bar} {dislike_pct:.1%} ({dislikes})"
        
        return result
    
    def calculate_trend(self, interactions: List[Dict], recent_n: int = 20) -> str:
        """
        Oblicza trend (improvement/decline) w ostatnich N interakcjach.
        
        Args:
            interactions: Lista interakcji
            recent_n: Liczba ostatnich interakcji do analizy
            
        Returns:
            String opisujący trend
        """
        if len(interactions) < recent_n:
            return "Za mało danych"
        
        # Podziel na dwie połowy
        mid = recent_n // 2
        first_half = interactions[:mid]
        second_half = interactions[mid:recent_n]
        
        # Policz pozytywne w każdej połowie
        first_positive = sum(1 for i in first_half if i['feedback_value'] > 0)
        second_positive = sum(1 for i in second_half if i['feedback_value'] > 0)
        
        first_rate = first_positive / len(first_half)
        second_rate = second_positive / len(second_half)
        
        diff = second_rate - first_rate
        
        if diff > 0.1:
            return f"📈 Wyraźna poprawa (+{diff:.1%})"
        elif diff > 0:
            return f"↗️  Niewielka poprawa (+{diff:.1%})"
        elif diff < -0.1:
            return f"📉 Spadek (-{abs(diff):.1%})"
        elif diff < 0:
            return f"↘️  Niewielki spadek ({diff:.1%})"
        else:
            return "➡️  Stabilny"
