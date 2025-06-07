import ast
import os
import time
import threading
import flet as ft
import pandas as pd
from business_logic.chatbot.react_agent import ReActAgent
from business_logic.classification.classification_wrapper import LoadedClassificationModel

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, 'resources', 'knowledge_base_categorized.csv')

# Categories explanation
LABEL_MAP = {
    1: "antisemitic ideology",
    2: "stereotypes and dehumanisation",
    3: "antisemitism against Israel or Zionism",
    4: "Holocaust or Zionism denial",
    5: "indirect antisemitism or secondary objective"
}


class ZionaApp:
    def __init__(self):
        self.knowledge_base = None
        self.classification_model = None
        self.page = None
        self.resources_loaded = False

    def load_resources(self):
        """Load knowledge base and classification model"""
        try:
            # Load knowledge base
            df = pd.read_csv(KNOWLEDGE_BASE_PATH, encoding='utf-8-sig')
            df = df[['source', 'url', 'paragraph', 'primary_categories']]
            df["primary_categories"] = df["primary_categories"].apply(lambda x: ast.literal_eval(x))
            df = df.dropna()
            df.reset_index(drop=True, inplace=True)
            self.knowledge_base = df

            # Load classification model
            # self.classification_model = LoadedClassificationModel(r"./resources/meta_model_best.pkl")
            self.resources_loaded = True
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.resources_loaded = False
            return False

    def analyze_comment_async(self, comment_text, progress_bar, status_text, result_container, analyze_button):
        """Analyze comment in a separate thread to prevent UI blocking"""
        try:
            # Update UI - Classification step
            self.page.run_thread(
                lambda: self.update_progress(progress_bar, status_text, 0.3, "Classifying comment..."))

            # Step 1: Classify the comment
            # pred = self.classification_model.predict(comment_text)
            # category_id = pred["predicted_labels"][0]
            category_id = 3
            category_name = LABEL_MAP[category_id]

            # Update UI - Response generation step
            self.page.run_thread(
                lambda: self.update_progress(progress_bar, status_text, 0.6, "Generating educational response..."))

            # Step 2: Generate the educational response
            agent = ReActAgent(self.knowledge_base)
            response = agent.generate_response(comment_text, category_id, category_name)

            # Update UI - Complete
            self.page.run_thread(lambda: self.update_progress(progress_bar, status_text, 1.0, "Analysis complete!"))

            # Display results
            self.page.run_thread(
                lambda: self.display_results(result_container, category_id, category_name, response))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.page.run_thread(lambda: self.show_error(status_text, error_msg))
        finally:
            # Re-enable button and reset text
            def reset_button():
                analyze_button.disabled = False
                analyze_button.text = "Analyze & Generate Response"
                self.page.update()

            self.page.run_thread(reset_button)

    def update_progress(self, progress_bar, status_text, value, text):
        """Update progress bar and status text"""
        progress_bar.value = value
        status_text.value = text
        self.page.update()

    def show_error(self, status_text, error_msg):
        """Show error message"""
        status_text.value = error_msg
        status_text.color = ft.Colors.RED
        self.page.update()

    def display_results(self, result_container, category_id, category_name, response):
        """Display analysis results"""
        result_container.controls.clear()

        # Classification results
        classification_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Classification Results", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(f"Category ID: {category_id}", size=16),
                    ft.Text(f"Category: {category_name}", size=16, weight=ft.FontWeight.W_500),
                ]),
                padding=20,
            ),
            elevation=3,
            margin=ft.margin.only(bottom=10)
        )

        # Resource information
        resource_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Selected Resource", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(f"Source: {response.get('source', 'N/A')}", size=14),
                    ft.Text(f"Retrieval Score: {response.get('retrieval_score', 0):.2f}", size=14),
                    ft.TextButton(
                        text=f"View Source: {response.get('url', 'N/A')}",
                        url=response.get('url', ''),
                        tooltip="Click to open source URL"
                    ) if response.get('url') else None,
                ], tight=True),
                padding=20,
            ),
            elevation=3,
            margin=ft.margin.only(bottom=10)
        )

        # Educational response
        response_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Educational Response", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=ft.Text(
                            response.get('final_response', 'No response generated'),
                            size=14,
                            selectable=True
                        ),
                        bgcolor=ft.Colors.GREY_100,
                        padding=15,
                        border_radius=8,
                    ),
                    ft.Row([
                        ft.ElevatedButton(
                            text="Copy Response",
                            icon=ft.Icons.COPY,
                            on_click=lambda _: self.copy_to_clipboard(response.get('final_response', ''))
                        ),
                    ], alignment=ft.MainAxisAlignment.END)
                ], tight=True),
                padding=20,
            ),
            elevation=3,
        )

        result_container.controls.extend([classification_card, resource_card, response_card])
        self.page.update()

    @staticmethod
    def validate_input(text):
        """Validate user input text"""
        if not isinstance(text, str):
            return False, "Input must be text"

        if not text or not text.strip():
            return False, "Please enter a comment to analyze"

        if len(text.strip()) < 3:
            return False, "Comment is too short (minimum 2 characters)"

        if len(text.strip()) > 5000:
            return False, "Comment is too long (maximum 5000 characters)"

        return True, ""

    def update_analyze_button_state(self, button, input_text, status_text=None):
        """Update analyze button state based on validation"""
        if not self.resources_loaded:
            button.disabled = True
            if status_text:
                status_text.value = "Please restart the application - resources failed to load"
                status_text.color = ft.Colors.RED
                status_text.visible = True

        is_valid, error_msg = self.validate_input(input_text)
        button.disabled = not is_valid
        if is_valid:
            button.text = "Analyze & Generate Response"
            button.disabled = False
            if status_text:
                status_text.visible = False
        else:
            button.text = "Enter Valid Comment"
            if status_text:
                status_text.value = error_msg
                status_text.color = ft.Colors.ORANGE_600
                status_text.visible = True

        self.page.update()

        # Remove notification after 6 seconds
        def remove_notification():
            time.sleep(6)
            if len(self.page.controls) > 1:
                self.page.controls.pop()
                self.page.update()

        threading.Thread(target=remove_notification, daemon=True).start()

    def show_notification(self, message, color=ft.Colors.BLUE_600):
        """Show a temporary notification message"""
        notification = ft.Text(
            message,
            size=14,
            color=color,
            weight=ft.FontWeight.W_500
        )

        # Add notification to page temporarily
        self.page.add(
            ft.Container(
                content=notification,
                bgcolor=ft.Colors.WHITE,
                padding=10,
                border_radius=8,
                shadow=ft.BoxShadow(
                    spread_radius=1,
                    blur_radius=10,
                    color=ft.Colors.BLACK12
                ),
                alignment=ft.alignment.center,
                margin=ft.margin.only(top=10, bottom=10)
            )
        )
        self.page.update()

        # Remove notification after 6 seconds
        def remove_notification():
            time.sleep(6)
            if len(self.page.controls) > 1:
                self.page.controls.pop()
                self.page.update()

        threading.Thread(target=remove_notification, daemon=True).start()

    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.page.set_clipboard(text)
        # Show a temporary success message
        self.show_notification("Response copied to clipboard!", ft.Colors.GREEN)

    def main(self, page: ft.Page):
        self.page = page
        page.title = "Ziona - Anti-Semitic Comment Analyzer"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.window_width = 800
        page.window_height = 900
        page.window_resizable = True
        page.scroll = ft.ScrollMode.AUTO

        # Header
        header = ft.Container(
            content=ft.Column([
                ft.Text(
                    "Ziona's Consulting App",
                    size=32,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_800
                ),
                ft.Text(
                    "Analyze problematic comments and get educational responses",
                    size=16,
                    color=ft.Colors.GREY_700
                ),
            ], tight=True),
            padding=ft.padding.only(bottom=20)
        )

        # Comment input with validation
        comment_input = ft.TextField(
            label="Enter a potentially anti-Semitic/anti-Israeli comment",
            multiline=True,
            min_lines=3,
            max_lines=8,
            border_color=ft.Colors.BLUE_400,
            focused_border_color=ft.Colors.BLUE_600,
            counter_text="",
            max_length=5000,
        )

        # Progress and status
        progress_bar = ft.ProgressBar(value=0, visible=False)
        status_text = ft.Text("", size=14, color=ft.Colors.BLUE_600, visible=False)
        validation_text = ft.Text("", size=12, color=ft.Colors.ORANGE_600, visible=False)

        # Results container
        result_container = ft.Column(spacing=10)

        def on_analyze_click(e):
            # Double-check validation before processing
            is_valid, error_msg = self.validate_input(comment_input.value)
            if not is_valid or not self.resources_loaded:
                self.show_notification(error_msg or "Resources not available", ft.Colors.RED)
                return

            # Show progress indicators
            progress_bar.visible = True
            status_text.visible = True
            validation_text.visible = False
            progress_bar.value = 0.1
            status_text.value = "Initializing analysis..."
            status_text.color = ft.Colors.BLUE_600
            result_container.controls.clear()

            # Disable button during processing
            analyze_button.disabled = True
            analyze_button.text = "Processing..."
            page.update()

            # Start analysis in separate thread
            threading.Thread(
                target=self.analyze_comment_async,
                args=(comment_input.value.strip(), progress_bar, status_text, result_container, analyze_button),
                daemon=True
            ).start()

        def on_input_change(e):
            # Update button state whenever input changes
            self.update_analyze_button_state(analyze_button, comment_input.value, validation_text)

        # Set input change handler
        comment_input.on_change = on_input_change

        # Analyze button (initially disabled)
        analyze_button = ft.ElevatedButton(
            text="Enter Valid Comment",
            icon=ft.Icons.ANALYTICS,
            on_click=on_analyze_click,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_600,
                padding=ft.padding.symmetric(horizontal=20, vertical=10)
            ),
            height=50
        )

        # Main layout
        main_content = ft.Column([
            header,
            comment_input,
            validation_text,
            ft.Container(height=10),  # Spacer
            analyze_button,
            ft.Container(height=20),  # Spacer
            progress_bar,
            status_text,
            ft.Container(height=10),  # Spacer
            result_container,
        ], spacing=0, scroll=ft.ScrollMode.AUTO)

        # Add to page
        page.add(
            ft.Container(
                content=main_content,
                padding=30,
                expand=True
            )
        )

        # Load resources on startup and set initial button state
        resources_loaded = self.load_resources()

        # Set initial button state
        self.update_analyze_button_state(analyze_button, "", validation_text)

        if not resources_loaded:
            self.show_notification(
                "Warning: Could not load resources. Please check file paths.",
                ft.Colors.RED
            )


def main():
    app = ZionaApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()
