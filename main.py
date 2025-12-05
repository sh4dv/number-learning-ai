#!/usr/bin/env python3
"""
Number Learning AI - Interaktywna aplikacja konsolowa z ML.

Użytkownik podaje liczby, AI odpowiada, użytkownik ocenia.
AI uczy się preferencji użytkownika i z czasem daje lepsze odpowiedzi.

Algorytm: Epsilon-Greedy + LinearRegression
"""

import sys
from typing import Optional

from storage import DataStorage
from ml_model import MLModel
from statistics import Statistics
from ui import UI


class NumberLearningApp:
    """Główna aplikacja - orchestrator wszystkich komponentów."""
    
    def __init__(self):
        """Inicjalizacja aplikacji i wszystkich modułów."""
        self.ui = UI()
        self.storage = DataStorage()
        self.model = MLModel()
        self.stats = Statistics()
        
        # Wczytaj istniejący model jeśli istnieje
        self._load_existing_model()
    
    def _load_existing_model(self) -> None:
        """Próbuje wczytać zapisany model z poprzedniej sesji."""
        if self.storage.model_exists():
            loaded_model = self.storage.load_model()
            if loaded_model:
                self.model = loaded_model
                self.ui.show_info("Wczytano zapisany model z poprzedniej sesji.")
            else:
                self.ui.show_info("Nie można wczytać modelu. Zaczynam od nowa.")
        else:
            self.ui.show_info("Brak zapisanego modelu. Zaczynam świeży trening!")
    
    def _validate_number_input(self, user_input: str) -> Optional[int]:
        """
        Waliduje input użytkownika jako dodatnią liczbę całkowitą.
        
        Args:
            user_input: String z inputem użytkownika
            
        Returns:
            Int jeśli valid, None jeśli invalid
        """
        try:
            number = int(user_input)
            if number <= 0:
                self.ui.show_error("Liczba musi być dodatnia!")
                return None
            return number
        except ValueError:
            return None  # Nie jest liczbą - prawdopodobnie komenda
    
    def _handle_command(self, command: str) -> bool:
        """
        Obsługuje specjalne komendy użytkownika.
        
        Args:
            command: String z komendą
            
        Returns:
            True jeśli program ma się zakończyć, False w przeciwnym razie
        """
        command = command.lower()
        
        if command in ['quit', 'exit', 'q']:
            return True
        
        elif command == 'help':
            self.ui.show_help()
        
        elif command == 'history':
            interactions = self.storage.get_recent_interactions(limit=10)
            self.ui.show_history(interactions)
        
        elif command == 'stats':
            self._show_detailed_statistics()
        
        elif command == 'explain':
            explanation = self.model.get_explanation()
            self.ui.show_explanation(explanation)
        
        elif command == 'reset':
            if self.ui.confirm_reset():
                self._reset_data_only()
                self.ui.show_reset_confirmation()
        
        elif command == 'retrain':
            self._retrain_model()
        
        elif command == 'undo':
            self._undo_last_interaction()
        
        elif command == 'new_model':
            self._create_new_model()
        
        elif command == 'save_model':
            self._save_model_as()
        
        elif command == 'load_model':
            self._load_model_by_name()
        
        elif command == 'list_models':
            self._list_available_models()
        
        elif command == 'delete_model':
            self._delete_model_by_name()
        
        elif command == 'export':
            filepath = self.storage.export_to_csv()
            self.ui.show_export_confirmation(filepath)
        
        elif command == 'train':
            self._training_mode()
        
        elif command == 'auto_train':
            self._auto_training_mode()
        
        elif command == 'testing_model':
            self._testing_model_mode()
        
        else:
            self.ui.show_error(f"Nieznana komenda: '{command}'. Wpisz 'help' aby zobaczyć dostępne komendy.")
        
        return False
    
    def _show_detailed_statistics(self) -> None:
        """Wyświetla szczegółowe statystyki."""
        # Pobierz dane
        db_stats = self.storage.get_statistics()
        model_stats = self.model.get_stats()
        session_stats = self.stats.get_session_stats()
        
        # Dodaj formatowanie czasu
        session_stats['duration_str'] = self.stats.format_duration(session_stats['duration'])
        
        # Trend
        recent_interactions = self.storage.get_recent_interactions(limit=20)
        trend = self.stats.calculate_trend(recent_interactions, recent_n=min(20, len(recent_interactions)))
        
        # Krzywa uczenia
        all_interactions = self.storage.get_recent_interactions(limit=1000)
        learning_curve = self.stats.calculate_learning_curve(all_interactions, window_size=10)
        chart = self.stats.generate_mini_chart(learning_curve, height=5, width=50)
        
        self.ui.show_detailed_stats(db_stats, model_stats, session_stats, trend, chart)
    
    def _reset_data_only(self) -> None:
        """Resetuje tylko dane - model jest zachowany."""
        self.storage.reset_database()
        self.stats = Statistics()
        # Model NIE jest resetowany - użytkownik może go zachować!
    
    def _retrain_model(self) -> None:
        """Przetrenuję model na wszystkich danych historycznych."""
        self.ui.show_retrain_progress()
        
        all_interactions = self.storage.get_all_interactions()
        self.model.batch_retrain(all_interactions)
        
        # Zapisz przetrenowany model
        self.storage.save_model(self.model)
    
    def _create_new_model(self) -> None:
        """Tworzy nowy, czysty model od zera."""
        if self.ui.confirm_new_model():
            self.model = MLModel()
            self.ui.show_new_model_created()
    
    def _save_model_as(self) -> None:
        """Zapisuje aktualny model pod określoną nazwą."""
        model_name = self.ui.get_model_name("Podaj nazwę modelu do zapisania")
        
        if not model_name:
            self.ui.show_error("Anulowano zapisywanie modelu.")
            return
        
        try:
            path = self.storage.save_model_as(self.model, model_name)
            self.ui.show_model_saved(model_name, path)
        except Exception as e:
            self.ui.show_error(f"Błąd zapisywania modelu: {e}")
    
    def _load_model_by_name(self) -> None:
        """Wczytuje model po nazwie."""
        # Najpierw pokaż dostępne modele
        models = self.storage.list_available_models()
        if models:
            self.ui.show_models_list(models)
        
        model_name = self.ui.get_model_name("Podaj nazwę modelu do wczytania")
        
        if not model_name:
            self.ui.show_error("Anulowano wczytywanie modelu.")
            return
        
        loaded_model = self.storage.load_model_by_name(model_name)
        
        if loaded_model:
            self.model = loaded_model
            self.ui.show_model_loaded(model_name)
        else:
            self.ui.show_error(f"Model '{model_name}' nie istnieje!")
    
    def _undo_last_interaction(self) -> None:
        """Cofa ostatnią interakcję - usuwa z bazy i przetrenuję model."""
        deleted = self.storage.delete_last_interaction()
        
        if not deleted:
            self.ui.show_error("Brak interakcji do cofnięcia!")
            return
        
        # Wyświetl informacje o usuniętej interakcji
        self.ui.show_info(f"🔙 Cofnięto interakcję:")
        self.ui.show_info(f"   Input: {deleted['user_input']} → Output: {deleted['model_output']}")
        self.ui.show_info(f"   Feedback: {deleted['feedback']} ({deleted['feedback_value']})")
        
        # Przetrenuję model na pozostałych danych
        self._retrain_model()
        
        # Aktualizuj liczniki
        self.model.interaction_count = max(0, self.model.interaction_count - 1)
        if deleted['feedback_value'] > 0:
            self.model.positive_feedback_count = max(0, self.model.positive_feedback_count - 1)
        
        self.ui.show_success("✅ Interakcja została cofnięta i model przetrenowany!")
    
    def _list_available_models(self) -> None:
        """Wyświetla listę dostępnych modeli."""
        models = self.storage.list_available_models()
        self.ui.show_models_list(models)
    
    def _delete_model_by_name(self) -> None:
        """Usuwa model po nazwie."""
        # Najpierw pokaż dostępne modele
        models = self.storage.list_available_models()
        if models:
            self.ui.show_models_list(models)
        else:
            self.ui.show_error("Brak modeli do usunięcia.")
            return
        
        model_name = self.ui.get_model_name("Podaj nazwę modelu do usunięcia")
        
        if not model_name:
            self.ui.show_error("Anulowano usuwanie modelu.")
            return
        
        if self.ui.confirm_model_delete(model_name):
            if self.storage.delete_model_by_name(model_name):
                self.ui.show_model_deleted(model_name)
            else:
                self.ui.show_error(f"Model '{model_name}' nie istnieje!")
    
    def _auto_training_mode(self) -> None:
        """Tryb auto-treningu - automatyczne generowanie przykładów z operacjami matematycznymi."""
        import random
        import time
        
        self.ui.show_auto_training_mode_start()
        
        # Zapytaj czy zresetować dane (zalecane dla czystego wzorca)
        if self.ui.confirm_auto_train_reset():
            self.ui.show_info("🗑️  Resetuję dane i model dla czystego treningu...")
            self.storage.reset_database()
            self.storage.delete_model()
            self.model = MLModel()
            self.stats = Statistics()
            self.ui.show_info("✅ Reset zakończony - czysty start!")
        
        # Pobierz parametry od użytkownika
        num_examples = self.ui.get_auto_train_examples_count()
        if num_examples is None:
            return
        
        operation = self.ui.get_auto_train_operation()
        if operation is None:
            return
        
        # START POMIARU CZASU
        start_time = time.time()
        
        # Generuj i trenuj
        training_count = 0
        
        self.ui.show_info(f"🤖 Rozpoczynam auto-trening: {num_examples} przykładów z operacją '{operation}'")
        
        for i in range(num_examples):
            # Generuj losową liczbę z logarytmicznym rozkładem (więcej małych liczb)
            # 30% szans na liczbę 1-20, 30% na 21-100, 40% na 101-1000
            rand = random.random()
            if rand < 0.3:
                random_input = random.randint(1, 20)
            elif rand < 0.6:
                random_input = random.randint(21, 100)
            else:
                random_input = random.randint(101, 1000)
            
            # Oblicz oczekiwany output według wybranej operacji
            try:
                expected_output = self._calculate_operation(random_input, operation)
            except Exception as e:
                self.ui.show_error(f"Błąd w operacji '{operation}': {e}")
                break
            
            if expected_output is None or expected_output <= 0:
                self.ui.show_error(f"Nieprawidłowy wynik operacji dla liczby {random_input}")
                continue
            
            # Zapisz interakcję
            feedback_value = 1.0  # like - idealne przykłady
            
            self.storage.save_interaction(
                user_input=random_input,
                model_output=expected_output,
                expected_output=expected_output,
                feedback='like',
                feedback_value=feedback_value,
                exploration=False
            )
            
            # Aktualizuj model
            self.model.update(random_input, expected_output, feedback_value)
            
            # Aktualizuj statystyki sesji
            self.stats.update_session(feedback_value)
            
            training_count += 1
            
            # Pokaż progress co 10 przykładów lub ostatni
            if (i + 1) % 10 == 0 or (i + 1) == num_examples:
                self.ui.show_auto_training_progress(i + 1, num_examples, random_input, expected_output)
            
            # Auto-save co 20 przykładów
            if training_count % 20 == 0:
                self.storage.save_model(self.model)
        
        # KONIEC POMIARU CZASU - ONLINE TRAINING
        online_training_time = time.time() - start_time
        
        # Podsumowanie
        self.ui.show_auto_training_mode_end(training_count, operation, online_training_time)
        
        # Przetrenuję model na wszystkich danych
        if training_count > 0:
            self.ui.show_info("🔄 Optymalizuję model na podstawie wszystkich danych...")
            
            # POMIAR CZASU BATCH RETRAIN
            batch_start_time = time.time()
            all_interactions = self.storage.get_all_interactions()
            self.model.batch_retrain(all_interactions)
            batch_retrain_time = time.time() - batch_start_time
            
            self.storage.save_model(self.model)
            
            # Całkowity czas
            total_time = time.time() - start_time
            
            # Pobierz rozmiary plików
            file_sizes = self.storage.get_file_sizes()
            
            self.ui.show_training_time_stats(
                online_time=online_training_time,
                batch_time=batch_retrain_time,
                total_time=total_time,
                num_examples=training_count,
                file_sizes=file_sizes
            )
    
    def _calculate_operation(self, number: int, operation: str) -> Optional[int]:
        """Oblicza wynik operacji matematycznej na liczbie."""
        try:
            # Obsługa różnych formatów operacji
            operation = operation.strip()
            
            # Jeśli zawiera 'x', użyj eval() dla pełnych wyrażeń (np. x*2+1, x*x+1)
            if 'x' in operation:
                # WAŻNE: Zamień ^ na ** (^ to XOR, nie potęgowanie!)
                operation = operation.replace('^', '**')
                
                # Bezpieczny eval z ograniczonym kontekstem
                safe_dict = {'x': number, '__builtins__': {}}
                result = eval(operation, safe_dict)
                return int(result) if isinstance(result, (int, float)) and result > 0 else None
            
            # Proste operatory bez 'x':
            
            # Mnożenie: *2, *3 itp.
            if operation.startswith('*'):
                multiplier = float(operation[1:])
                return int(number * multiplier)
            
            # Dzielenie: /2, /3 itp.
            elif operation.startswith('/'):
                divisor = float(operation[1:])
                if divisor == 0:
                    return None
                return int(number / divisor)
            
            # Dodawanie: +10, +100 itp.
            elif operation.startswith('+'):
                addend = int(operation[1:])
                return number + addend
            
            # Odejmowanie: -10, -50 itp.
            elif operation.startswith('-'):
                subtrahend = int(operation[1:])
                result = number - subtrahend
                return result if result > 0 else None
            
            # Potęga: ^2, **2 itp.
            elif operation.startswith('^') or operation.startswith('**'):
                if operation.startswith('**'):
                    exponent = int(operation[2:])
                else:
                    exponent = int(operation[1:])
                result = number ** exponent
                # Ogranicz do rozsądnych wartości
                return int(result) if result < 10**10 else None
            
            # Modulo: %10, %100 itp.
            elif operation.startswith('%'):
                modulo = int(operation[1:])
                if modulo == 0:
                    return None
                result = number % modulo
                return result if result > 0 else number  # Jeśli 0, zwróć oryginalną liczbę
            
            else:
                return None
        
        except Exception:
            return None
    
    def _training_mode(self) -> None:
        """Tryb treningu - użytkownik kontroluje input i output."""
        self.ui.show_training_mode_start()
        
        training_count = 0
        
        while True:
            # Pobierz input
            input_str = self.ui.get_training_input()
            
            if input_str is None:
                break
            
            if input_str.lower() == 'stop':
                break
            
            # Waliduj input
            try:
                user_input = int(input_str)
                if user_input <= 0:
                    self.ui.show_error("Input musi być dodatni!")
                    continue
            except ValueError:
                self.ui.show_error("Podaj poprawną liczbę!")
                continue
            
            # Pobierz output
            output_str = self.ui.get_training_output()
            
            if output_str is None:
                break
            
            if output_str.lower() == 'stop':
                break
            
            # Waliduj output
            try:
                user_output = int(output_str)
                if user_output <= 0:
                    self.ui.show_error("Output musi być dodatni!")
                    continue
            except ValueError:
                self.ui.show_error("Podaj poprawną liczbę!")
                continue
            
            # Zapisz jako like (idealny przykład)
            feedback_value = 1.0  # like
            
            self.storage.save_interaction(
                user_input=user_input,
                model_output=user_output,
                expected_output=user_output,  # W treningu: oczekiwany = rzeczywisty output
                feedback='like',
                feedback_value=feedback_value,
                exploration=False  # training = nie eksploracja
            )
            
            # Aktualizuj model
            self.model.update(user_input, user_output, feedback_value)
            
            # Aktualizuj statystyki sesji
            self.stats.update_session(feedback_value)
            
            # Potwierdzenie
            self.ui.show_training_saved(user_input, user_output)
            
            training_count += 1
            
            # Auto-save co 5 przykładów
            if training_count % 5 == 0:
                self.storage.save_model(self.model)
        
        # Podsumowanie
        self.ui.show_training_mode_end(training_count)
        
        # PRZETRENUJĘ model na wszystkich danych (offline learning)
        if training_count > 0:
            self.ui.show_info("🔄 Optymalizuję model na podstawie wszystkich danych...")
            all_interactions = self.storage.get_all_interactions()
            self.model.batch_retrain(all_interactions)
            self.storage.save_model(self.model)
            self.ui.show_info("✅ Model zoptymalizowany i zapisany!")
    
    def _testing_model_mode(self) -> None:
        """Tryb testowania modelu - automatyczne testowanie na losowych przykładach."""
        import random
        import time
        
        self.ui.show_testing_mode_start()
        
        # Pobierz parametry od użytkownika
        num_tests = self.ui.get_testing_examples_count()
        if num_tests is None:
            return
        
        operation = self.ui.get_testing_operation()
        if operation is None:
            return
        
        # START POMIARU CZASU
        start_time = time.time()
        
        # Statystyki testowania
        correct_predictions = 0
        total_tests = 0
        test_results = []
        
        self.ui.show_info(f"🧪 Rozpoczynam testowanie: {num_tests} przykładów z wzorcem '{operation}'")
        
        for i in range(num_tests):
            # Generuj losową liczbę (od 1 do 1000)
            random_input = random.randint(1, 1000)
            
            # Oblicz prawidłową odpowiedź według wzorca
            try:
                correct_answer = self._calculate_operation(random_input, operation)
            except Exception as e:
                self.ui.show_error(f"Błąd w operacji '{operation}': {e}")
                break
            
            if correct_answer is None or correct_answer <= 0:
                continue
            
            # AI przewiduje odpowiedź
            ai_prediction, is_exploration = self.model.predict(random_input)
            
            # Sprawdź czy AI zgadło
            is_correct = (ai_prediction == correct_answer)
            
            # Określ feedback
            if is_correct:
                feedback = 'like'
                feedback_value = 1.0
                correct_predictions += 1
            else:
                feedback = 'dislike'
                feedback_value = 0.0
            
            # Zapisz interakcję
            self.storage.save_interaction(
                user_input=random_input,
                model_output=ai_prediction,
                expected_output=correct_answer,
                feedback=feedback,
                feedback_value=feedback_value,
                exploration=is_exploration
            )
            
            # Aktualizuj model (uczy się z poprawnej odpowiedzi)
            self.model.update(random_input, correct_answer, feedback_value)
            
            # Aktualizuj statystyki sesji
            self.stats.update_session(feedback_value)
            
            total_tests += 1
            
            # Zachowaj wynik do statystyk
            test_results.append({
                'input': random_input,
                'ai_prediction': ai_prediction,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'is_exploration': is_exploration
            })
            
            # Pokaż progress co 10 przykładów lub ostatni
            if (i + 1) % 10 == 0 or (i + 1) == num_tests:
                accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
                self.ui.show_testing_progress(
                    i + 1, 
                    num_tests, 
                    random_input, 
                    ai_prediction, 
                    correct_answer, 
                    is_correct,
                    accuracy
                )
            
            # Auto-save co 20 przykładów
            if total_tests % 20 == 0:
                self.storage.save_model(self.model)
        
        # KONIEC TESTOWANIA
        testing_time = time.time() - start_time
        
        # Oblicz finalną dokładność
        final_accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
        
        # Podsumowanie
        self.ui.show_testing_mode_end(
            total_tests, 
            correct_predictions, 
            final_accuracy, 
            operation, 
            testing_time
        )
        
        # Zapisz model po testowaniu
        if total_tests > 0:
            self.storage.save_model(self.model)
            self.ui.show_info("✅ Model zaktualizowany i zapisany!")
    
    def _main_interaction_loop(self) -> None:
        """Główna pętla interakcji z użytkownikiem."""
        while True:
            # 1. Pobierz input użytkownika
            user_input_str = self.ui.get_user_input()
            
            if user_input_str is None:
                # Ctrl+C lub EOF
                break
            
            # 2. Sprawdź czy to komenda
            number = self._validate_number_input(user_input_str)
            
            if number is None:
                # To nie liczba - sprawdź czy komenda
                should_quit = self._handle_command(user_input_str)
                if should_quit:
                    break
                continue
            
            # 3. Model generuje predykcję
            prediction, is_exploration = self.model.predict(number)
            
            # 4. Wyświetl predykcję
            self.ui.show_prediction(number, prediction, is_exploration)
            
            # 5. Pobierz feedback
            feedback = self.ui.get_feedback()
            
            if feedback is None:
                # Użytkownik przerwał
                break
            
            # 6. Mapuj feedback na wartość numeryczną
            feedback_values = {
                'dislike': 0.0,  # Błędna odpowiedź
                'like': 1.0      # Idealna odpowiedź
            }
            feedback_value = feedback_values[feedback]
            
            # 6a. Jeśli dislike, zapytaj o oczekiwaną odpowiedź
            # Like = idealnie, używamy tego co model zwrócił
            expected_output = prediction  # Domyślnie - to co model zwrócił
            if feedback == 'dislike':
                try:
                    expected_output = self.ui.get_expected_output(number)
                    if expected_output is None:
                        # Użytkownik nie podał - użyj predykcji modelu
                        expected_output = prediction
                except Exception:
                    expected_output = prediction
            
            # 7. Zapisz interakcję
            self.storage.save_interaction(
                user_input=number,
                model_output=prediction,
                expected_output=expected_output,
                feedback=feedback,
                feedback_value=feedback_value,
                exploration=is_exploration
            )
            
            # 8. Aktualizuj model z OCZEKIWANYM outputem
            self.model.update(number, expected_output, feedback_value)
            
            # 9. Aktualizuj statystyki sesji
            self.stats.update_session(feedback_value)
            
            # 10. Wyświetl potwierdzenie i szybkie statystyki
            self.ui.show_feedback_confirmation(feedback)
            
            db_stats = self.storage.get_statistics()
            model_stats = self.model.get_stats()
            self.ui.show_quick_stats(db_stats, model_stats)
            
            # 11. Automatyczny retrain co 10 interakcji (uczenie na bieżąco)
            if self.stats.session_interactions % 10 == 0:
                all_interactions = self.storage.get_all_interactions()
                self.model.batch_retrain(all_interactions)
                self.ui.show_info("🔄 Model automatycznie przetrenowany na wszystkich danych!")
            
            # 12. Okresowo zapisuj model (co 5 interakcji)
            if self.stats.session_interactions % 5 == 0:
                self.storage.save_model(self.model)
    
    def run(self) -> None:
        """Główna metoda uruchomieniowa aplikacji."""
        try:
            # Ekran powitalny
            self.ui.show_welcome()
            input("\n[Naciśnij Enter aby rozpocząć...]\n")
            
            # Główna pętla
            self._main_interaction_loop()
            
        except KeyboardInterrupt:
            # Graceful shutdown przy Ctrl+C
            self.ui.console.print("\n")
        
        finally:
            # Zapisz model przed wyjściem
            self.storage.save_model(self.model)
            
            # Pożegnanie
            self.ui.show_goodbye()


def main():
    """Entry point aplikacji."""
    app = NumberLearningApp()
    app.run()


if __name__ == "__main__":
    main()
