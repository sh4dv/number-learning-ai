"""
Moduł zarządzania danymi - SQLite dla interakcji oraz pickle dla modelu ML.
Obsługuje persistence między sesjami i możliwość resetu.
"""

import sqlite3
import pickle
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class DataStorage:
    """Zarządzanie bazą danych SQLite i persistencją modelu."""
    
    def __init__(self, db_path: str = "data/interactions.db", model_path: str = "models/ml_model.pkl"):
        """
        Inicjalizacja storage.
        
        Args:
            db_path: Ścieżka do bazy danych SQLite
            model_path: Ścieżka do zapisanego modelu pickle
        """
        self.db_path = db_path
        self.model_path = model_path
        self.models_dir = "models"
        
        # Upewnij się, że katalogi istnieją
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Tworzy tabelę interakcji jeśli nie istnieje."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input INTEGER NOT NULL,
                    model_output INTEGER NOT NULL,
                    expected_output INTEGER,
                    feedback TEXT NOT NULL,
                    feedback_value REAL NOT NULL,
                    exploration BOOLEAN NOT NULL
                )
            """)
            
            # Dodaj kolumnę expected_output do istniejących tabel (jeśli nie ma)
            try:
                conn.execute("ALTER TABLE interactions ADD COLUMN expected_output INTEGER")
                conn.commit()
            except sqlite3.OperationalError:
                # Kolumna już istnieje
                pass
            
            # Indeks dla szybszych zapytań
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON interactions(timestamp DESC)
            """)
            
            conn.commit()
    
    def save_interaction(self, user_input: int, model_output: int, 
                        expected_output: Optional[int],
                        feedback: str, feedback_value: float, 
                        exploration: bool) -> None:
        """
        Zapisuje pojedynczą interakcję do bazy danych.
        
        Args:
            user_input: Liczba podana przez użytkownika
            model_output: Liczba zwrócona przez model
            expected_output: Oczekiwana/poprawna odpowiedź (None jeśli brak)
            feedback: Typ feedbacku ('like', 'dislike', 'love')
            feedback_value: Wartość numeryczna (0.0, 1.0, 2.0)
            exploration: Czy był to ruch eksploracyjny (losowy)
        """
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO interactions 
                (timestamp, user_input, model_output, expected_output, feedback, feedback_value, exploration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, user_input, model_output, expected_output, feedback, feedback_value, exploration))
            conn.commit()
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """
        Pobiera ostatnie N interakcji.
        
        Args:
            limit: Maksymalna liczba wyników
            
        Returns:
            Lista słowników z danymi interakcji
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM interactions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_interactions(self) -> List[Tuple[int, int, float]]:
        """
        Pobiera wszystkie interakcje dla treningu modelu.
        
        Returns:
            Lista tupli (user_input, expected_output, feedback_value)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_input, 
                       COALESCE(expected_output, model_output) as output,
                       feedback_value 
                FROM interactions 
                ORDER BY timestamp ASC
            """)
            
            return cursor.fetchall()
    
    def get_statistics(self) -> Dict:
        """
        Oblicza podstawowe statystyki z bazy danych.
        
        Returns:
            Słownik ze statystykami
        """
        with sqlite3.connect(self.db_path) as conn:
            # Całkowita liczba interakcji
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            
            if total == 0:
                return {
                    'total_interactions': 0,
                    'likes': 0,
                    'dislikes': 0,
                    'loves': 0,
                    'positive_rate': 0.0,
                    'exploration_rate': 0.0
                }
            
            # Rozkład feedbacku
            feedback_counts = conn.execute("""
                SELECT feedback, COUNT(*) as count 
                FROM interactions 
                GROUP BY feedback
            """).fetchall()
            
            feedback_dict = {row[0]: row[1] for row in feedback_counts}
            
            # Procent eksploracji
            exploration_count = conn.execute("""
                SELECT COUNT(*) FROM interactions WHERE exploration = 1
            """).fetchone()[0]
            
            likes = feedback_dict.get('like', 0)
            loves = feedback_dict.get('love', 0)
            dislikes = feedback_dict.get('dislike', 0)
            
            positive_rate = (likes + loves) / total if total > 0 else 0.0
            exploration_rate = exploration_count / total if total > 0 else 0.0
            
            return {
                'total_interactions': total,
                'likes': likes,
                'dislikes': dislikes,
                'loves': loves,
                'positive_rate': positive_rate,
                'exploration_rate': exploration_rate
            }
    
    def delete_last_interaction(self) -> Optional[Dict]:
        """
        Usuwa ostatnią interakcję z bazy danych.
        
        Returns:
            Słownik z danymi usuniętej interakcji lub None jeśli baza pusta
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Pobierz ostatnią interakcję
            cursor = conn.execute("""
                SELECT * FROM interactions 
                ORDER BY id DESC 
                LIMIT 1
            """)
            last_interaction = cursor.fetchone()
            
            if not last_interaction:
                return None
            
            # Usuń ostatnią interakcję
            conn.execute("""
                DELETE FROM interactions 
                WHERE id = ?
            """, (last_interaction['id'],))
            conn.commit()
            
            return dict(last_interaction)
    
    def reset_database(self) -> None:
        """Usuwa wszystkie dane z bazy (hard reset)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM interactions")
            conn.commit()
    
    def save_model(self, model) -> None:
        """
        Zapisuje model ML do pliku pickle.
        
        Args:
            model: Obiekt modelu do zapisania
        """
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
    
    def load_model(self) -> Optional[object]:
        """
        Wczytuje model ML z pliku pickle.
        
        Returns:
            Wczytany model lub None jeśli plik nie istnieje
        """
        if not os.path.exists(self.model_path):
            return None
        
        try:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Błąd wczytywania modelu: {e}")
            return None
    
    def model_exists(self) -> bool:
        """Sprawdza czy zapisany model istnieje."""
        return os.path.exists(self.model_path)
    
    def delete_model(self) -> None:
        """Usuwa zapisany model."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def get_file_sizes(self) -> Dict[str, int]:
        """Zwraca rozmiary plików w bajtach."""
        sizes = {
            'model_bytes': 0,
            'database_bytes': 0
        }
        
        if os.path.exists(self.model_path):
            sizes['model_bytes'] = os.path.getsize(self.model_path)
        
        if os.path.exists(self.db_path):
            sizes['database_bytes'] = os.path.getsize(self.db_path)
        
        return sizes
    
    def save_model_as(self, model, model_name: str) -> str:
        """
        Zapisuje model pod określoną nazwą.
        
        Args:
            model: Obiekt modelu do zapisania
            model_name: Nazwa modelu (bez rozszerzenia)
            
        Returns:
            Pełna ścieżka do zapisanego pliku
        """
        # Usuń rozszerzenie jeśli użytkownik je podał
        if model_name.endswith('.pkl'):
            model_name = model_name[:-4]
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model_path
    
    def load_model_by_name(self, model_name: str) -> Optional[object]:
        """
        Wczytuje model po nazwie.
        
        Args:
            model_name: Nazwa modelu (bez rozszerzenia)
            
        Returns:
            Wczytany model lub None jeśli nie istnieje
        """
        # Usuń rozszerzenie jeśli użytkownik je podał
        if model_name.endswith('.pkl'):
            model_name = model_name[:-4]
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Błąd wczytywania modelu '{model_name}': {e}")
            return None
    
    def list_available_models(self) -> List[Dict[str, any]]:
        """
        Zwraca listę dostępnych modeli.
        
        Returns:
            Lista słowników z informacjami o modelach
        """
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                model_path = os.path.join(self.models_dir, filename)
                model_name = filename[:-4]  # Usuń .pkl
                
                # Pobierz informacje o pliku
                stat = os.stat(model_path)
                size_kb = stat.st_size / 1024
                modified = datetime.fromtimestamp(stat.st_mtime)
                
                models.append({
                    'name': model_name,
                    'filename': filename,
                    'path': model_path,
                    'size_kb': size_kb,
                    'modified': modified.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sortuj po dacie modyfikacji (najnowsze pierwsze)
        models.sort(key=lambda x: x['modified'], reverse=True)
        
        return models
    
    def delete_model_by_name(self, model_name: str) -> bool:
        """
        Usuwa model po nazwie.
        
        Args:
            model_name: Nazwa modelu (bez rozszerzenia)
            
        Returns:
            True jeśli usunięto, False jeśli nie istniał
        """
        # Usuń rozszerzenie jeśli użytkownik je podał
        if model_name.endswith('.pkl'):
            model_name = model_name[:-4]
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        if os.path.exists(model_path):
            os.remove(model_path)
            return True
        
        return False
    
    def export_to_csv(self, output_path: str = "data/export.csv") -> str:
        """
        Eksportuje wszystkie interakcje do CSV.
        
        Args:
            output_path: Ścieżka do pliku CSV
            
        Returns:
            Ścieżka do utworzonego pliku
        """
        import csv
        
        interactions = self.get_recent_interactions(limit=100000)  # wszystkie
        
        with open(output_path, 'w', newline='') as csvfile:
            if interactions:
                fieldnames = interactions[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(interactions)
        
        return output_path
