"""
Moduł UI - piękny interfejs konsolowy z wykorzystaniem Rich.
Kolorowe outputy, tabele, panele i czytelne komunikaty.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.layout import Layout
from rich.text import Text
from rich import box
from typing import List, Dict, Optional
import time


class UI:
    """Zarządza interfejsem użytkownika w konsoli."""
    
    def __init__(self):
        """Inicjalizacja konsoli Rich."""
        self.console = Console()
    
    def clear(self) -> None:
        """Czyści ekran konsoli."""
        self.console.clear()
    
    def show_welcome(self) -> None:
        """Wyświetla ekran powitalny."""
        self.clear()
        
        welcome_text = """
[bold cyan]🤖 Number Learning AI[/bold cyan]

Witaj w interaktywnej aplikacji ML!

[yellow]Jak to działa:[/yellow]
1. Podajesz liczbę (dodatnią całkowitą)
2. AI generuje swoją odpowiedź
3. Dajesz feedback: [green]like[/green] (idealnie) lub [red]dislike[/red] (źle)
4. AI uczy się Twoich preferencji i z czasem daje lepsze odpowiedzi!

[cyan]📋 Dostępne komendy:[/cyan]
  [white]train[/white]         - Tryb treningu (kontrolujesz output)
  [white]auto_train[/white]    - Auto-trening (losowe liczby + operacje matematyczne)
  [white]testing_model[/white] - Test modelu (automatyczne testowanie na wzorcu)
  [white]history[/white]        - Ostatnie 10 interakcji
  [white]stats[/white]          - Szczegółowe statystyki i wykres
    [white]show_formula[/white]   - Pokaż wyuczony wzór
  [white]explain[/white]        - Wyjaśnij ostatnią predykcję
  [white]reset[/white]          - Resetuj dane (zachowaj model)
  [white]retrain[/white]        - Przetrenuję na wszystkich danych
  [white]undo[/white]           - Cofnij ostatnią interakcję
  [white]new_model[/white]     - Utwórz nowy czysty model
  [white]save_model[/white]    - Zapisz model pod nazwą
  [white]load_model[/white]    - Wczytaj zapisany model
  [white]list_models[/white]   - Lista dostępnych modeli
  [white]delete_model[/white]  - Usuń zapisany model
  [white]export[/white]         - Eksportuj do CSV
  [white]help[/white]           - Pomoc
  [white]quit[/white]           - Zakończ program

[dim]Algorytm: Epsilon-Greedy + Polynomial Regression[/dim]
        """
        
        panel = Panel(
            welcome_text,
            title="💡 Machine Learning Console App",
            border_style="cyan",
            box=box.DOUBLE
        )
        
        self.console.print(panel)
        self.console.print()
    
    def show_help(self) -> None:
        """Wyświetla pomoc z dostępnymi komendami."""
        help_table = Table(title="📚 Dostępne Komendy", box=box.ROUNDED, border_style="blue")
        
        help_table.add_column("Komenda", style="cyan", no_wrap=True)
        help_table.add_column("Opis", style="white")
        
        commands = [
            ("train", "Tryb treningu - kontrolujesz output (wpisz 'stop' aby zakończyć)"),
            ("auto_train", "Automatyczny trening - generowanie przykładów z operacjami matematycznymi"),
            ("testing_model", "Automatyczne testowanie modelu na wzorcu (AI dostaje losowe liczby)"),
            ("history", "Wyświetl ostatnie 10 interakcji"),
            ("stats", "Pokaż szczegółowe statystyki i postępy"),
            ("show_formula", "Wypisz wyuczony wzór i współczynniki"),
            ("explain", "Wyjaśnij ostatnią predykcję AI"),
            ("reset", "Resetuj tylko dane (zachowaj model)"),
            ("retrain", "Przetrenuję model na wszystkich danych"),
            ("undo", "Cofnij ostatnią interakcję i przetrenuję model"),
            ("new_model", "Utwórz nowy, czysty model (od zera)"),
            ("save_model", "Zapisz aktualny model pod nazwą"),
            ("load_model", "Wczytaj zapisany model"),
            ("list_models", "Pokaż dostępne modele"),
            ("delete_model", "Usuń zapisany model"),
            ("export", "Eksportuj dane do CSV"),
            ("help", "Wyświetl tę pomoc"),
            ("quit", "Zakończ program"),
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        self.console.print(help_table)
        self.console.print()
    
    def get_user_input(self) -> Optional[str]:
        """
        Pobiera liczbę od użytkownika (lub komendę).
        
        Returns:
            String z inputem lub None przy błędzie
        """
        try:
            user_input = Prompt.ask("\n[bold yellow]Podaj liczbę (lub komendę)[/bold yellow]")
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def show_prediction(self, user_input: int, prediction: int, is_exploration: bool) -> None:
        """
        Wyświetla predykcję AI.
        
        Args:
            user_input: Liczba użytkownika
            prediction: Predykcja AI
            is_exploration: Czy to była eksploracja
        """
        mode_text = "[yellow]🔍 Eksploracja[/yellow]" if is_exploration else "[green]🎯 Predykcja[/green]"
        
        prediction_panel = Panel(
            f"[bold white]Twoja liczba:[/bold white] [cyan]{user_input}[/cyan]\n"
            f"[bold white]AI odpowiada:[/bold white] [magenta bold]{prediction}[/magenta bold]\n"
            f"[bold white]Tryb:[/bold white] {mode_text}",
            title="💭 Odpowiedź AI",
            border_style="magenta",
            box=box.ROUNDED
        )
        
        self.console.print(prediction_panel)
    
    def get_feedback(self) -> Optional[str]:
        """
        Pobiera feedback od użytkownika.
        
        Returns:
            'dislike', 'like' lub None przy błędzie
        """
        self.console.print("\n[bold]Oceń odpowiedź:[/bold]")
        self.console.print("  [red]d[/red] / [red]dislike[/red] - Źle (podaj poprawną odpowiedź)")
        self.console.print("  [green]l[/green] / [green]like[/green]       - Idealnie! 👍")
        
        try:
            feedback = Prompt.ask("[bold yellow]Twoja ocena[/bold yellow]", 
                                choices=["d", "dislike", "l", "like"],
                                show_choices=False).lower()
            
            # Normalizuj do pełnych nazw
            if feedback in ['d', 'dislike']:
                return 'dislike'
            elif feedback in ['l', 'like']:
                return 'like'
            
        except (KeyboardInterrupt, EOFError):
            return None
    
    def show_feedback_confirmation(self, feedback: str) -> None:
        """Wyświetla potwierdzenie feedbacku z emotikonami."""
        if feedback == 'like':
            self.console.print("[green bold]👍 Like! Idealna odpowiedź![/green bold]")
        elif feedback == 'dislike':
            self.console.print("[red]👎 Dislike. Uczę się z poprawnej odpowiedzi.[/red]")
    
    def get_expected_output(self, user_input: int) -> Optional[int]:
        """
        Pyta użytkownika o oczekiwaną odpowiedź (przy dislike).
        
        Args:
            user_input: Liczba wejściowa użytkownika
            
        Returns:
            Oczekiwana wartość wyjściowa lub None
        """
        try:
            self.console.print(f"\n[yellow]❓ Jaka powinna być poprawna odpowiedź dla {user_input}?[/yellow]")
            expected = IntPrompt.ask(
                "[bold cyan]Poprawna odpowiedź[/bold cyan]"
            )
            return expected if expected > 0 else None
        except (KeyboardInterrupt, EOFError, ValueError):
            return None
    
    def show_quick_stats(self, stats: Dict, model_stats: Dict) -> None:
        """
        Wyświetla szybkie statystyki po każdej interakcji.
        
        Args:
            stats: Statystyki z DataStorage
            model_stats: Statystyki z MLModel
        """
        total = stats['total_interactions']
        positive_rate = stats['positive_rate']
        epsilon = model_stats['epsilon']
        
        # Koloruj positive rate
        if positive_rate > 0.7:
            rate_color = "green"
        elif positive_rate > 0.4:
            rate_color = "yellow"
        else:
            rate_color = "red"
        
        quick_stats = (
            f"[dim]Interakcji: {total} | "
            f"Sukces: [{rate_color}]{positive_rate:.1%}[/{rate_color}] | "
            f"Eksploracja: {epsilon:.1%}[/dim]"
        )
        
        self.console.print(quick_stats)
    
    def show_history(self, interactions: List[Dict]) -> None:
        """
        Wyświetla historię ostatnich interakcji.
        
        Args:
            interactions: Lista interakcji z bazy danych
        """
        if not interactions:
            self.console.print("[yellow]Brak historii interakcji.[/yellow]")
            return
        
        table = Table(title="📜 Historia (ostatnie 10)", box=box.SIMPLE, border_style="cyan")
        
        table.add_column("#", style="dim", width=4)
        table.add_column("Czas", style="cyan", width=19)
        table.add_column("Input", style="white", justify="right")
        table.add_column("Output", style="magenta", justify="right")
        table.add_column("Feedback", justify="center")
        table.add_column("Tryb", style="dim")
        
        for idx, interaction in enumerate(reversed(interactions), 1):
            timestamp = interaction['timestamp'][:19]  # YYYY-MM-DD HH:MM:SS
            user_input = str(interaction['user_input'])
            output = str(interaction['model_output'])
            feedback = interaction['feedback']
            exploration = interaction['exploration']
            
            # Koloruj feedback (mapuj starsze 'love' na 'like')
            if feedback == 'love' or feedback == 'like':
                feedback_display = "[green]👍 Like[/green]"
            else:
                feedback_display = "[red]👎 Dislike[/red]"
            
            mode = "🔍 Explore" if exploration else "🎯 Predict"
            
            table.add_row(str(idx), timestamp, user_input, output, feedback_display, mode)
        
        self.console.print(table)
        self.console.print()
    
    def show_detailed_stats(self, stats: Dict, model_stats: Dict, 
                           session_stats: Dict, trend: str, 
                           learning_curve_chart: str) -> None:
        """
        Wyświetla szczegółowe statystyki.
        
        Args:
            stats: Statystyki ogólne
            model_stats: Statystyki modelu
            session_stats: Statystyki sesji
            trend: String z trendem
            learning_curve_chart: Wykres ASCII
        """
        # Panel główny
        main_stats = f"""
[bold cyan]📊 Statystyki Ogólne[/bold cyan]
  Całkowite interakcje: [white]{stats['total_interactions']}[/white]
  Pozytywny feedback:   [green]{stats['positive_rate']:.1%}[/green]
  
[bold yellow]📈 Rozkład Feedbacku[/bold yellow]
    Like (👍):     {stats['likes'] + stats.get('loves', 0)} ({(stats['likes'] + stats.get('loves', 0))/max(1, stats['total_interactions']):.1%})
    Dislike (👎):  {stats['dislikes']} ({stats['dislikes']/max(1, stats['total_interactions']):.1%})

[bold magenta]🤖 Model ML[/bold magenta]
  Status:          [green]Wytrenowany[/green] if model_stats['is_fitted'] else [red]Uczący się[/red]
  Epsilon (🔍):    {model_stats['epsilon']:.2%}
  Sukces modelu:   {model_stats['success_rate']:.1%}

[bold green]⏱️  Obecna Sesja[/bold green]
  Czas trwania:    {session_stats['duration_str']}
  Interakcje:      {session_stats['interactions']}
  Sukces:          {session_stats['positive_rate']:.1%}

[bold blue]📉 Trend (ostatnie 20)[/bold blue]
  {trend}
        """
        
        panel = Panel(main_stats, title="📈 Szczegółowe Statystyki", 
                     border_style="cyan", box=box.DOUBLE)
        
        self.console.print(panel)
        
        # Wykres krzywej uczenia
        if learning_curve_chart:
            chart_panel = Panel(learning_curve_chart, 
                              title="📊 Krzywa Uczenia (Accuracy over Time)",
                              border_style="green", box=box.ROUNDED)
            self.console.print(chart_panel)
        
        self.console.print()
    
    def show_explanation(self, explanation: str) -> None:
        """Wyświetla wyjaśnienie ostatniej predykcji."""
        panel = Panel(explanation, title="🔍 Wyjaśnienie Predykcji", 
                     border_style="yellow", box=box.ROUNDED)
        self.console.print(panel)
        self.console.print()

    def show_formula(self, formula: str) -> None:
        """Wyświetla nauczony wzór modelu wraz z współczynnikami."""
        panel = Panel(formula, title="🧠 Wyuczony Wzór", border_style="green", box=box.ROUNDED)
        self.console.print(panel)
        self.console.print()
    
    def confirm_reset(self) -> bool:
        """Prosi o potwierdzenie resetu."""
        self.console.print("[bold yellow]⚠️  UWAGA: Reset usunie wszystkie dane ale ZACHOWA model![/bold yellow]")
        return Confirm.ask("[yellow]Czy na pewno chcesz zresetować dane?[/yellow]", default=False)
    
    def show_reset_confirmation(self) -> None:
        """Potwierdza wykonanie resetu."""
        self.console.print("[green]✓ Reset danych zakończony. Model został zachowany![/green]\n")
    
    def show_models_list(self, models: List[Dict]) -> None:
        """
        Wyświetla listę dostępnych modeli.
        
        Args:
            models: Lista słowników z informacjami o modelach
        """
        if not models:
            self.console.print("[yellow]📁 Brak zapisanych modeli.[/yellow]\n")
            return
        
        table = Table(title="🧠 Dostępne Modele", box=box.ROUNDED, border_style="cyan")
        
        table.add_column("Nazwa", style="cyan", no_wrap=True)
        table.add_column("Rozmiar", style="yellow", justify="right")
        table.add_column("Ostatnia modyfikacja", style="white")
        
        for model in models:
            size_str = f"{model['size_kb']:.2f} KB"
            table.add_row(model['name'], size_str, model['modified'])
        
        self.console.print(table)
        self.console.print()
    
    def get_model_name(self, prompt_text: str = "Podaj nazwę modelu") -> Optional[str]:
        """
        Pobiera nazwę modelu od użytkownika.
        
        Args:
            prompt_text: Tekst promptu
            
        Returns:
            Nazwa modelu lub None jeśli anulowano
        """
        try:
            name = Prompt.ask(f"[cyan]{prompt_text}[/cyan]")
            if name and name.strip():
                return name.strip()
            return None
        except (KeyboardInterrupt, EOFError):
            return None
    
    def confirm_model_delete(self, model_name: str) -> bool:
        """
        Prosi o potwierdzenie usunięcia modelu.
        
        Args:
            model_name: Nazwa modelu do usunięcia
        """
        self.console.print(f"[bold red]⚠️  UWAGA: Usuniesz model '{model_name}'![/bold red]")
        return Confirm.ask("[yellow]Czy na pewno?[/yellow]", default=False)
    
    def show_model_saved(self, model_name: str, path: str) -> None:
        """Potwierdza zapisanie modelu."""
        self.console.print(f"[green]✓ Model zapisany jako:[/green] [cyan]{model_name}[/cyan]")
        self.console.print(f"[dim]Ścieżka: {path}[/dim]\n")
    
    def show_model_loaded(self, model_name: str) -> None:
        """Potwierdza wczytanie modelu."""
        self.console.print(f"[green]✓ Model wczytany:[/green] [cyan]{model_name}[/cyan]\n")
    
    def show_model_deleted(self, model_name: str) -> None:
        """Potwierdza usunięcie modelu."""
        self.console.print(f"[green]✅ Model usunięty:[/green] [cyan]{model_name}[/cyan]\n")
    
    def confirm_new_model(self) -> bool:
        """Prosi o potwierdzenie utworzenia nowego modelu."""
        self.console.print("[bold yellow]⚠️  UWAGA: Utworzysz nowy, niewytrenowany model![/bold yellow]")
        self.console.print("[dim]Aktualny model nie zostanie zapisany, chyba że użyjesz 'save_model' wcześniej.[/dim]")
        return Confirm.ask("[yellow]Czy na pewno chcesz utworzyć nowy model?[/yellow]", default=False)
    
    def show_new_model_created(self) -> None:
        """Potwierdza utworzenie nowego modelu."""
        self.console.print("[green]✅ Utworzono nowy, czysty model![/green]")
        self.console.print("[cyan]💡 Możesz teraz trenować go od zera.[/cyan]\n")
    
    def show_retrain_progress(self) -> None:
        """Wyświetla progress bar dla retreningu."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Przetre modeluję model na wszystkich danych...", total=None)
            time.sleep(1)  # Symulacja treningu
            progress.update(task, completed=True)
        
        self.console.print("[green]✅ Retraining zakończony![/green]\n")
    
    def show_export_confirmation(self, filepath: str) -> None:
        """Potwierdza eksport danych."""
        self.console.print(f"[green]✅ Dane wyeksportowane do:[/green] [cyan]{filepath}[/cyan]\n")
    
    def show_error(self, message: str) -> None:
        """Wyświetla komunikat błędu."""
        self.console.print(f"[bold red]❌ Błąd:[/bold red] {message}\n")
    
    def show_info(self, message: str) -> None:
        """Wyświetla komunikat informacyjny."""
        self.console.print(f"[blue]ℹ️  {message}[/blue]\n")
    
    def show_success(self, message: str) -> None:
        """Wyświetla komunikat sukcesu."""
        self.console.print(f"[bold green]{message}[/bold green]\n")
    
    def show_goodbye(self) -> None:
        """Wyświetla pożegnanie."""
        goodbye_text = """
[bold cyan]👋 Do zobaczenia![/bold cyan]

Dziękuję za trening. Twoje dane zostały zapisane.

[dim]Model będzie kontynuował naukę przy następnym uruchomieniu.[/dim]
        """
        
        panel = Panel(goodbye_text, title="🤖 Goodbye", 
                     border_style="cyan", box=box.DOUBLE)
        
        self.console.print(panel)
    
    def show_training_mode_start(self) -> None:
        """Wyświetla info o trybie treningu."""
        training_info = """
[bold green]🎓 Tryb Treningu Aktywny[/bold green]

W tym trybie TY kontrolujesz output - uczysz AI dokładnie tego, czego chcesz!

[yellow]Jak to działa:[/yellow]
1. Podaj INPUT (liczba)
2. Podaj OUTPUT (liczba, którą AI powinno zwrócić)
3. AI zapisuje to jako idealny przykład (LIKE 👍)
4. Powtarzaj aż nauczysz wzorca

[cyan]Przykłady:[/cyan]
  Podwajanie:  10 → 20, 5 → 10, 7 → 14
  Dodaj 100:   10 → 110, 50 → 150
  Razy 3:      5 → 15, 10 → 30

[white]Wpisz 'stop' aby zakończyć trening i wrócić do normalnego trybu.[/white]
        """
        
        panel = Panel(training_info, title="📚 Training Mode", 
                     border_style="green", box=box.DOUBLE)
        
        self.console.print(panel)
    
    def show_training_mode_end(self, count: int) -> None:
        """Wyświetla podsumowanie trybu treningu."""
        self.console.print(f"\n[green]✅ Trening zakończony! Dodano {count} przykładów.[/green]")
        self.console.print("[cyan]AI powinno teraz rozumieć Twój wzorzec.[/cyan]\n")
    
    def get_training_input(self) -> Optional[str]:
        """Pobiera input w trybie treningu."""
        try:
            user_input = Prompt.ask("\n[bold yellow]INPUT (lub 'stop' aby zakończyć)[/bold yellow]")
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def get_training_output(self) -> Optional[str]:
        """Pobiera output w trybie treningu."""
        try:
            user_output = Prompt.ask("[bold magenta]OUTPUT (idealna odpowiedź)[/bold magenta]")
            return user_output.strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def show_training_saved(self, user_input: int, user_output: int) -> None:
        """Potwierdza zapisanie przykładu treningowego."""
        self.console.print(f"[green]👍 Zapisano:[/green] {user_input} → {user_output} [dim](like)[/dim]")
    
    def show_auto_training_mode_start(self) -> None:
        """Wyświetla info o trybie auto-treningu."""
        auto_training_info = """
[bold green]🤖 Tryb Auto-Treningu Aktywny[/bold green]

AI będzie automatycznie generować przykłady treningowe!

[yellow]Jak to działa:[/yellow]
1. Podajesz ilość przykładów do wygenerowania
2. Podajesz operację matematyczną (np. *2, +100, ^2)
3. AI generuje losowe liczby i stosuje na nich operację
4. Model trenuje się na tych przykładach

[cyan]Przykłady operacji:[/cyan]
  [white]*2[/white]    - mnożenie przez 2
  [white]*3.5[/white]  - mnożenie przez 3.5
  [white]+100[/white]  - dodawanie 100
  [white]-50[/white]   - odejmowanie 50
  [white]/2[/white]    - dzielenie przez 2
  [white]^2[/white]    - potęgowanie do kwadratu
  [white]%10[/white]   - modulo 10
  [white]x*2+10[/white] - złożone wyrażenie (x to liczba wejściowa)

[bold red]⚠️  WAŻNE:[/bold red]
Model polinomialny (3. stopnia) najlepiej uczy się [bold]jednego wzorca[/bold]. Jeśli masz już dane z innymi
wzorcami w bazie, zalecamy reset przed treningiem dla najlepszych wyników!

[dim]AI nauczy się wzorca i będzie go stosować w predykcjach![/dim]
        """
        
        panel = Panel(auto_training_info, title="🤖 Auto-Training Mode", 
                     border_style="green", box=box.DOUBLE)
        
        self.console.print(panel)
    
    def confirm_auto_train_reset(self) -> bool:
        """Pyta czy zresetować dane przed auto-treningiem."""
        self.console.print("\n[bold yellow]💡 Rekomendacja:[/bold yellow]")
        self.console.print("[yellow]Dla najlepszych wyników zalecamy reset danych przed auto-treningiem.[/yellow]")
        self.console.print("[yellow]Model polinomialny (3. stopnia) najlepiej uczy się jednego wzorca na czystych danych.[/yellow]\n")
        
        return Confirm.ask(
            "[cyan]Czy zresetować dane i model przed treningiem?[/cyan]",
            default=True
        )
    
    def get_auto_train_examples_count(self) -> Optional[int]:
        """Pobiera od użytkownika ilość przykładów do wygenerowania."""
        try:
            count = IntPrompt.ask(
                "\n[bold yellow]Ile przykładów wygenerować?[/bold yellow]",
                default=50
            )
            if count <= 0:
                self.show_error("Liczba musi być większa od 0!")
                return None
            if count > 10000:
                self.show_error("Maksymalna liczba przykładów to 10000!")
                return None
            return count
        except (KeyboardInterrupt, EOFError):
            return None
    
    def get_auto_train_operation(self) -> Optional[str]:
        """Pobiera od użytkownika operację matematyczną."""
        try:
            self.console.print("\n[bold yellow]Podaj operację matematyczną:[/bold yellow]")
            self.console.print("[dim]Przykłady: *2, +100, /2, ^2, x*2+10[/dim]")
            
            operation = Prompt.ask("[bold cyan]Operacja[/bold cyan]")
            
            if not operation or operation.strip() == "":
                self.show_error("Operacja nie może być pusta!")
                return None
            
            return operation.strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def show_auto_training_progress(self, current: int, total: int, 
                                   last_input: int, last_output: int) -> None:
        """Wyświetla postęp auto-treningu."""
        percentage = (current / total) * 100
        self.console.print(
            f"[cyan]Postęp: {current}/{total} ({percentage:.1f}%)[/cyan] | "
            f"[dim]Ostatni: {last_input} → {last_output}[/dim]"
        )
    
    def show_auto_training_mode_end(self, count: int, operation: str, training_time: float) -> None:
        """Wyświetla podsumowanie trybu auto-treningu."""
        self.console.print(f"\n[green]✅ Auto-trening zakończony![/green]")
        self.console.print(f"[white]Wygenerowano i wytrenowano {count} przykładów z operacją '[cyan]{operation}[/cyan]'.[/white]")
        self.console.print(f"[dim]Czas online trainingu: {training_time:.2f}s ({training_time/count*1000:.2f}ms na przykład)[/dim]")
        self.console.print("[yellow]AI powinno teraz rozumieć ten wzorzec i stosować go w predykcjach.[/yellow]\n")
    
    def show_training_time_stats(self, online_time: float, batch_time: float, 
                                total_time: float, num_examples: int,
                                file_sizes: Dict) -> None:
        """Wyświetla szczegółowe statystyki czasowe treningu i rozmiary plików."""
        stats_table = Table(title="⏱️  Statystyki Czasowe Treningu", box=box.ROUNDED, border_style="cyan")
        
        stats_table.add_column("Etap", style="yellow", no_wrap=True)
        stats_table.add_column("Czas", style="cyan", justify="right")
        stats_table.add_column("Szczegóły", style="dim")
        
        # Online training
        online_per_example = (online_time / num_examples * 1000) if num_examples > 0 else 0
        stats_table.add_row(
            "Online Training",
            f"{online_time:.2f}s",
            f"{online_per_example:.2f}ms/przykład"
        )
        
        # Batch retrain
        stats_table.add_row(
            "Batch Retrain",
            f"{batch_time:.2f}s",
            f"100 epok przez {num_examples} przykładów"
        )
        
        # Całkowity czas
        stats_table.add_row(
            "[bold]CAŁKOWITY CZAS[/bold]",
            f"[bold green]{total_time:.2f}s[/bold green]",
            f"[bold]{total_time/60:.2f} minut[/bold]"
        )
        
        self.console.print(stats_table)
        
        # Tabela rozmiarów plików
        size_table = Table(title="💾 Rozmiary Plików", box=box.ROUNDED, border_style="magenta")
        size_table.add_column("Plik", style="yellow")
        size_table.add_column("Rozmiar", style="cyan", justify="right")
        
        # Model
        model_size = file_sizes.get('model_bytes', 0)
        model_size_str = self._format_file_size(model_size)
        size_table.add_row("Model ML (pickle)", model_size_str)
        
        # Baza danych
        db_size = file_sizes.get('database_bytes', 0)
        db_size_str = self._format_file_size(db_size)
        size_table.add_row("Baza danych (SQLite)", db_size_str)
        
        # Całkowity rozmiar
        total_size = model_size + db_size
        total_size_str = self._format_file_size(total_size)
        size_table.add_row("[bold]CAŁKOWITY ROZMIAR[/bold]", f"[bold green]{total_size_str}[/bold green]")
        
        self.console.print(size_table)
        
        # Dodatkowe info
        self.console.print(f"\n[green]✅ Model zoptymalizowany i zapisany![/green]")
        self.console.print(f"[cyan]🚀 Wydajność: {num_examples/total_time:.1f} przykładów/s[/cyan]")
        self.console.print(f"[magenta]💾 Rozmiar na przykład: {total_size/num_examples:.2f} bajtów[/magenta]\n")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Formatuje rozmiar pliku do czytelnej postaci."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def show_testing_mode_start(self) -> None:
        """Wyświetla info o trybie testowania modelu."""
        testing_info = """
[bold blue]🧪 Tryb Testowania Modelu[/bold blue]

Ten tryb automatycznie testuje jak dobrze AI zgaduje wzorzec!

[yellow]Jak to działa:[/yellow]
1. Podajesz ilość testów do wykonania
2. Podajesz wzorzec/operację (np. *2, +100)
3. AI dostaje losowe liczby i próbuje zgadnąć odpowiedź
4. Jeśli AI zgadnie poprawnie → [green]LIKE 👍[/green]
5. Jeśli AI się pomyli → [red]DISLIKE + poprawna odpowiedź[/red]
6. AI uczy się w trakcie testowania!

[cyan]Zastosowanie:[/cyan]
  • Sprawdzenie jak dobrze model nauczył się wzorca
  • Kontynuacja treningu z automatycznym feedbackiem
  • Monitoring postępu modelu (accuracy)

[dim]Model będzie się uczył podczas testowania![/dim]
        """
        
        panel = Panel(testing_info, title="🧪 Testing Mode", 
                     border_style="blue", box=box.DOUBLE)
        
        self.console.print(panel)
    
    def get_testing_examples_count(self) -> Optional[int]:
        """Pobiera od użytkownika ilość testów do wykonania."""
        try:
            count = IntPrompt.ask(
                "\n[bold yellow]Ile testów wykonać?[/bold yellow]",
                default=100
            )
            if count <= 0:
                self.show_error("Liczba musi być większa od 0!")
                return None
            if count > 10000:
                self.show_error("Maksymalna liczba testów to 10000!")
                return None
            return count
        except (KeyboardInterrupt, EOFError):
            return None
    
    def get_testing_operation(self) -> Optional[str]:
        """Pobiera od użytkownika wzorzec do testowania."""
        try:
            self.console.print("\n[bold yellow]Podaj wzorzec do testowania:[/bold yellow]")
            self.console.print("[dim]Przykłady: *2, +100, /2, ^2, x*2+10[/dim]")
            
            operation = Prompt.ask("[bold cyan]Wzorzec[/bold cyan]")
            
            if not operation or operation.strip() == "":
                self.show_error("Wzorzec nie może być pusty!")
                return None
            
            return operation.strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def show_testing_progress(self, current: int, total: int, 
                             test_input: int, ai_answer: int, 
                             correct_answer: int, is_correct: bool,
                             accuracy: float) -> None:
        """Wyświetla postęp testowania."""
        percentage = (current / total) * 100
        
        # Status: ✓ lub ✗
        status = "[green]✓[/green]" if is_correct else "[red]✗[/red]"
        
        # Koloruj accuracy
        if accuracy >= 90:
            acc_color = "green"
        elif accuracy >= 70:
            acc_color = "yellow"
        else:
            acc_color = "red"
        
        self.console.print(
            f"[cyan]Test: {current}/{total} ({percentage:.1f}%)[/cyan] | "
            f"{status} Input: {test_input} | AI: {ai_answer} | Poprawna: {correct_answer} | "
            f"[{acc_color}]Accuracy: {accuracy:.1f}%[/{acc_color}]"
        )
    
    def show_testing_mode_end(self, total_tests: int, correct: int, 
                             accuracy: float, operation: str, 
                             testing_time: float) -> None:
        """Wyświetla podsumowanie trybu testowania."""
        # Tabela wyników
        results_table = Table(title="📊 Wyniki Testowania", box=box.ROUNDED, border_style="blue")
        
        results_table.add_column("Metryka", style="yellow")
        results_table.add_column("Wartość", style="cyan", justify="right")
        
        results_table.add_row("Wykonane testy", str(total_tests))
        results_table.add_row("Poprawne odpowiedzi", f"[green]{correct}[/green]")
        results_table.add_row("Błędne odpowiedzi", f"[red]{total_tests - correct}[/red]")
        
        # Koloruj accuracy w zależności od wyniku
        if accuracy >= 90:
            acc_color = "bold green"
        elif accuracy >= 70:
            acc_color = "bold yellow"
        else:
            acc_color = "bold red"
        
        results_table.add_row("[bold]ACCURACY[/bold]", f"[{acc_color}]{accuracy:.2f}%[/{acc_color}]")
        results_table.add_row("Wzorzec", f"[magenta]{operation}[/magenta]")
        results_table.add_row("Czas", f"{testing_time:.2f}s")
        results_table.add_row("Średni czas/test", f"{testing_time/total_tests*1000:.2f}ms")
        
        self.console.print()
        self.console.print(results_table)
        
        # Komunikat końcowy
        self.console.print()
        if accuracy >= 95:
            self.console.print("[bold green]🎉 DOSKONALE! Model świetnie rozumie wzorzec![/bold green]")
        elif accuracy >= 80:
            self.console.print("[bold yellow]👍 Dobrze! Model w większości zgaduje poprawnie.[/bold yellow]")
        elif accuracy >= 50:
            self.console.print("[bold yellow]🤔 Średnio. Model potrzebuje więcej treningu.[/bold yellow]")
        else:
            self.console.print("[bold red]❌ Słabo. Model nie rozumie wzorca - potrzebny trening![/bold red]")
        
        self.console.print()

