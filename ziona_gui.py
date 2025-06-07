import ast
import os
import threading
import flet as ft
import pandas as pd
from business_logic.chatbot.react_agent import ReActAgent
from business_logic.classification.classification_wrapper import LoadedClassificationModel

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, 'resources', 'knowledge_base.csv')

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

    def load_resources(self):
        """Load knowledge base and classification model"""
        try:
            # Load knowledge base
            df = pd.read_csv(KNOWLEDGE_BASE_PATH)
            df = df[['source', 'url', 'paragraph', 'categories']]
            df = df.dropna(subset=['source', 'url', 'paragraph'])
            df.reset_index(drop=True, inplace=True)
            df["primary_categories"] = df["primary_categories"].apply(lambda x: ast.literal_eval(x))
            self.knowledge_base = df

            # Load classification model
            self.classification_model = LoadedClassificationModel(r"./resources/meta_model_best.pkl")
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            return False

    def analyze_comment_async(self, comment_text, progress_bar, status_text, result_container):
        """Analyze comment in a separate thread to prevent UI blocking"""
        try:
            # Update UI - Classification step
            self.page.run_in_thread(
                lambda: self.update_progress(progress_bar, status_text, 0.3, "Classifying comment..."))

            # Step 1: Classify the comment
            pred = self.classification_model.predict(comment_text)
            category_id = pred["predicted_labels"][0]
            category_name = LABEL_MAP[category_id]

            # Update UI - Response generation step
            self.page.run_in_thread(
                lambda: self.update_progress(progress_bar, status_text, 0.6, "Generating educational response..."))

            # Step 2: Generate the educational response
            agent = ReActAgent(self.knowledge_base)
            response = agent.generate_response(comment_text, category_id, category_name)

            # Update UI - Complete
            self.page.run_in_thread(lambda: self.update_progress(progress_bar, status_text, 1.0, "Analysis complete!"))

            # Display results
            self.page.run_in_thread(
                lambda: self.display_results(result_container, category_id, category_name, response))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.page.run_in_thread(lambda: self.show_error(status_text, error_msg))

    def update_progress(self, progress_bar, status_text, value, text):
        """Update progress bar and status text"""
        progress_bar.value = value
        status_text.value = text
        self.page.update()

    def show_error(self, status_text, error_msg):
        """Show error message"""
        status_text.value = error_msg
        status_text.color = ft.colors.RED
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
                        bgcolor=ft.colors.GREY_100,
                        padding=15,
                        border_radius=8,
                    ),
                    ft.Row([
                        ft.ElevatedButton(
                            text="Copy Response",
                            icon=ft.icons.COPY,
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

    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.page.set_clipboard(text)
        self.page.show_snack_bar(ft.SnackBar(content=ft.Text("Response copied to clipboard!")))

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
                    color=ft.colors.BLUE_800
                ),
                ft.Text(
                    "Analyze problematic comments and get educational responses",
                    size=16,
                    color=ft.colors.GREY_700
                ),
            ], tight=True),
            padding=ft.padding.only(bottom=20)
        )

        # Comment input
        comment_input = ft.TextField(
            label="Enter a potentially anti-Semitic/anti-Israeli comment",
            multiline=True,
            min_lines=3,
            max_lines=8,
            border_color=ft.colors.BLUE_400,
            focused_border_color=ft.colors.BLUE_600,
        )

        # Progress and status
        progress_bar = ft.ProgressBar(value=0, visible=False)
        status_text = ft.Text("", size=14, color=ft.colors.BLUE_600, visible=False)

        # Results container
        result_container = ft.Column(spacing=10)

        def on_analyze_click(e):
            if not comment_input.value or not comment_input.value.strip():
                page.show_snack_bar(
                    ft.SnackBar(content=ft.Text("Please enter a comment to analyze"))
                )
                return

            # Show progress indicators
            progress_bar.visible = True
            status_text.visible = True
            progress_bar.value = 0.1
            status_text.value = "Initializing analysis..."
            status_text.color = ft.colors.BLUE_600
            result_container.controls.clear()
            page.update()

            # Start analysis in separate thread
            threading.Thread(
                target=self.analyze_comment_async,
                args=(comment_input.value.strip(), progress_bar, status_text, result_container),
                daemon=True
            ).start()

        # Analyze button
        analyze_button = ft.ElevatedButton(
            text="Analyze & Generate Response",
            icon=ft.icons.ANALYTICS,
            on_click=on_analyze_click,
            style=ft.ButtonStyle(
                color=ft.colors.WHITE,
                bgcolor=ft.colors.BLUE_600,
                padding=ft.padding.symmetric(horizontal=20, vertical=10)
            ),
            height=50
        )

        # Main layout
        main_content = ft.Column([
            header,
            comment_input,
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

        # Load resources on startup
        if not self.load_resources():
            page.show_snack_bar(
                ft.SnackBar(
                    content=ft.Text("Warning: Could not load resources. Please check file paths."),
                    bgcolor=ft.colors.ORANGE_400
                )
            )


def main():
    app = ZionaApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()