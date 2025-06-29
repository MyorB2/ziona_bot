import ast
import os
import threading
import time
import flet as ft
import pandas as pd
from datetime import datetime

from sympy import vfield

from business_logic.chatbot.react_agent import ReActAgent
from business_logic.classification.classification_wrapper import ClassificationModel

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, 'resources', 'knowledge_base_categorized.csv')

# Categories explanation
LABEL_MAP = {
    0: "Non Antisemitic",
    1: "Antisemitic Ideology",
    2: "Stereotypes and Dehumanization",
    3: "Antisemitism Against Israel/Zionism",
    4: "Holocaust Denial/Distortion",
    5: "Indirect Antisemitism",
}


def normalize_categories_column(row):
    norm_list = ast.literal_eval(row)
    if len(norm_list) == 0:
        norm_list = None
    return norm_list


class ZionaApp:
    def __init__(self):
        self.is_valid_comment_df = None
        self.comments_df = None
        self.knowledge_base = None
        self.classification_model = None
        self.page = None
        self.resources_loaded = False
        self.current_mode = "single"  # "single" or "batch"

        self.help_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("How to Use Ziona's Consulting App", size=18, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Welcome to Ziona's Anti-Semitic Comment Analyzer!",
                            size=16, weight=ft.FontWeight.W_500),
                    ft.Container(height=10),

                    ft.Text("Requirements for first use:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text("• Follow the guid on 'https://www.youtube.com/watch?app=desktop&v=F5Erab2EMhU'",
                            size=12),
                    ft.Text("• from minute 0:52 to 2:30 to do the following actions:", size=12),
                    ft.Text("• Download Ollama framework, ", size=12),
                    ft.Text("• Open the terminal (write 'cmd' on the search bar)", size=12),
                    ft.Text("• Activate Ollama", size=12),
                    ft.Text("• Then follow the guide from minute 3:38 to 4:10 to download a specific LLM,",
                            size=12),
                    ft.Text("• you need to run the command 'ollama run llama3'.", size=12),
                    ft.Container(height=10),

                    ft.Text("Requirements for every use:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text("• Activate Ollama by just clicking the Ollama App on your desktop.", size=12),
                    ft.Container(height=10),

                    ft.Text("Single Comment Mode:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text("• Enter a potentially problematic comment in the text field", size=12),
                    ft.Text("• Click 'Analyze & Generate Response' to get classification and educational response",
                            size=12),
                    ft.Text("• View results including category, sources, and suggested responses", size=12),
                    ft.Text("• Copy responses to clipboard for use", size=12),
                    ft.Container(height=10),

                    ft.Text("Multiple Comments Mode:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text("• Upload an Excel file (.xlsx or .xls) with comments", size=12),
                    ft.Text("• File must have a 'comment' column (case insensitive)", size=12),
                    ft.Text("• Maximum 100 comments per file", size=12),
                    ft.Text("• Results are saved as CSV file with all analyses", size=12),
                    ft.Container(height=10),

                    ft.Text("Tips:", size=14, weight=ft.FontWeight.BOLD),
                    ft.Text("• Comments should be between 3-5000 characters", size=12),
                    ft.Text("• Processing may take a few moments depending on complexity", size=12),
                    ft.Text("• All generated responses are educational and constructive", size=12),

                ], tight=True, scroll=ft.ScrollMode.AUTO),
                height=400,
                width=500
            ),
            actions=[
                ft.TextButton("Got it!", on_click=lambda e: self.page.close(self.help_dialog))
            ]
        )

    def load_resources(self):
        """Load knowledge base and classification model"""
        try:
            # Load knowledge base
            df = pd.read_csv(KNOWLEDGE_BASE_PATH, encoding='utf-8-sig')
            df = df[['source', 'url', 'paragraph', 'primary_categories']]
            df = df.dropna()
            df["primary_categories"] = df["primary_categories"].apply(lambda x: normalize_categories_column(x))
            df = df.dropna()
            df.reset_index(drop=True, inplace=True)
            self.knowledge_base = df

            # Load classification model
            self.classification_model = ClassificationModel()
            self.resources_loaded = True
            return True
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.resources_loaded = False
            return False

    def analyze_comment_async(self, comment_text, result_container):
        """Analyze comment in a separate thread to prevent UI blocking"""
        try:
            # Update UI - Classification step
            self.page.run_thread(
                lambda: self.update_progress(0.3, "Classifying comment..."))

            # Step 1: Classify the comment
            pred = self.classification_model.predict(comment_text)
            category_id = pred["predicted_labels"][0]
            category_name = LABEL_MAP[category_id]
            # Update UI - Response generation step
            self.page.run_thread(
                lambda: self.update_progress(0.6, "Generating educational response..."))

            # Step 2: Generate the educational response
            if category_id == 0:
                response = {
                    "final_response": "No response is needed",
                    "paragraph": "N/A",
                    "source": "N/A",
                    "url": "N/A"
                }
            else:
                agent = ReActAgent(self.knowledge_base)
                response = agent.generate_response(comment_text, category_id, category_name)

            # Update UI - Complete
            self.page.run_thread(lambda: self.update_progress(1.0, "Analysis complete!"))

            # Display results
            self.page.run_thread(
                lambda: self.display_results(result_container, category_id, category_name, response))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.page.run_thread(lambda: self.show_error(error_msg))
        finally:
            # Re-enable button and reset text
            def reset_button():
                self.analyze_button.disabled = False
                self.analyze_button.text = "Analyze & Generate Response"
                self.page.update()

            self.page.run_thread(reset_button)

    def update_progress(self, progress_bar_value, status_text_value):
        """Update progress bar and status text"""
        self.progress_bar.value = progress_bar_value
        self.status_text.value = status_text_value
        self.page.update()

    def show_error(self, error_msg):
        """Show error message"""
        self.status_text.value = error_msg
        self.status_text.color = ft.Colors.RED
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
                        url=response.get('url', 'N/A'),
                        tooltip="Click to open source URL"
                    ) if response.get('url') else "N/A",
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
                            response.get('final_response', 'N/A'),
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
                            on_click=lambda _: self.copy_to_clipboard(response.get('final_response', 'N/A'))
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
    def read_excel_file(excel_path):
        # Try reading the file with and without headers
        try:
            # Attempt to read with headers
            df = pd.read_excel(excel_path)

            # We expect a header to be only one word.
            # If all columns are longer then one word, treat as no headers
            if all(len(str(col).split()) > 1 for col in df.columns):
                raise ValueError("Detected no proper headers, switching to header=None")
            df.columns = ["comment"]
            df["comment"].dropna()
            print("Excel read with headers...")
            return df
        except Exception:
            # Read without headers
            df = pd.read_excel(excel_path, header=None)
            df.columns = ["comment"]
            df["comment"].dropna()
            print("Excel read without headers...")
            return df

    def validate_excel_file(self, file_path):
        """Validate uploaded Excel file"""
        try:
            self.comments_df = self.read_excel_file(file_path)

            if self.comments_df.empty:
                return False, "Excel file is empty"

            if len(self.comments_df["comment"]) == 0:
                return False, "No valid comments found in the file"

            if len(self.comments_df["comment"]) > 100:
                return False, "Too many comments (maximum 100 allowed)"

            return True, f"Found {len(self.comments_df["comment"])} valid comments to analyze"

        except Exception as e:
            return False, f"Error reading Excel file: {str(e)}"

    def process_batch_analysis(self, result_container):
        """Process batch analysis of Excel file"""
        try:
            # Read Excel file

            # Filter valid comments
            total_comments = len(self.comments_df)

            results = []

            i = 0
            for index, row in self.comments_df.iterrows():
                comment_text = str(row["comment"]).strip()

                # Update progress
                current_progress = i / total_comments
                self.page.run_thread(
                    lambda idx=index: self.update_progress(
                        current_progress,
                        f"Processing comment {i}/{total_comments}..."
                    )
                )
                i += 1

                try:
                    # Classify comment
                    pred = self.classification_model.predict(comment_text)
                    category_id = pred["predicted_labels"][0]
                    category_name = LABEL_MAP[category_id]

                    # Generate response
                    if category_id == 0:
                        response = {
                            "final_response": "No response is needed",
                            "paragraph": "N/A",
                            "source": "N/A",
                            "url": "N/A"
                        }
                    else:
                        agent = ReActAgent(self.knowledge_base)
                        response = agent.generate_response(comment_text, category_id, category_name)

                    # Store result
                    result = {
                        'original_row': index + 1,
                        'comment': comment_text,
                        'category_id': category_id,
                        'category_name': category_name,
                        'educational_response': response.get('final_response', 'N/A'),
                        'source': response.get('source', 'N/A'),
                        'source_url': response.get('url', 'N/A'),
                        'retrieval_score': response.get('retrieval_score', 0)
                    }
                    results.append(result)

                except Exception as e:
                    # Handle individual comment errors
                    result = {
                        'original_row': index + 1,
                        'comment': comment_text,
                        'category_id': 'ERROR',
                        'category_name': 'Analysis Failed',
                        'educational_response': f'Error processing comment: {str(e)}',
                        'source': 'N/A',
                        'source_url': 'N/A',
                        'retrieval_score': 0
                    }
                    results.append(result)

            # Save results to CSV
            output_filename = f"ziona_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(os.getcwd(), output_filename)

            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            # Update UI with completion
            self.page.run_thread(
                lambda: self.update_progress(1.0, "Batch analysis complete!")
            )

            # Display batch results
            self.page.run_thread(
                lambda: self.display_batch_results(result_container, results, output_path)
            )

        except Exception as e:
            error_msg = f"Error during batch analysis: {str(e)}"
            self.page.run_thread(lambda: self.show_error(error_msg))
        finally:
            # Re-enable button
            def reset_button():
                self.analyze_button.disabled = False
                self.analyze_button.text = "Analyze Excel File"
                self.page.update()

            self.page.run_thread(reset_button)

    def display_batch_results(self, result_container, results, output_path):
        """Display batch analysis results"""
        result_container.controls.clear()

        # Summary card
        summary_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Batch Analysis Complete", size=20, weight=ft.FontWeight.BOLD),
                    ft.Text(f"Total comments processed: {len(results)}", size=16),
                    ft.Text(f"Results saved to: {os.path.basename(output_path)}", size=14),
                    ft.Row([
                        ft.ElevatedButton(
                            text="Open Results Folder",
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=lambda _: os.startfile(os.path.dirname(output_path))
                        ),
                        ft.ElevatedButton(
                            text="Copy File Path",
                            icon=ft.Icons.COPY,
                            on_click=lambda _: self.copy_to_clipboard(output_path)
                        ),
                    ], alignment=ft.MainAxisAlignment.START)
                ]),
                padding=20,
            ),
            elevation=3,
            margin=ft.margin.only(bottom=10)
        )

        # Category breakdown
        category_counts = {}
        for result in results:
            cat = result['category_name']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        breakdown_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Category Breakdown", size=18, weight=ft.FontWeight.BOLD),
                    *[ft.Text(f"• {cat}: {count} comments", size=14)
                      for cat, count in category_counts.items()]
                ]),
                padding=20,
            ),
            elevation=3,
            margin=ft.margin.only(bottom=10)
        )

        # Sample results preview
        preview_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Sample Results", size=18, weight=ft.FontWeight.BOLD),
                    *[
                        ft.Container(
                            content=ft.Column([
                                ft.Text(f"Comment {i + 1}: {result['comment'][:100]}...",
                                        size=12, weight=ft.FontWeight.W_500),
                                ft.Text(f"Category: {result['category_name']}", size=11),
                                ft.Text(f"Response: {result['educational_response'][:150]}...", size=11),
                            ]),
                            padding=10,
                            bgcolor=ft.Colors.GREY_50,
                            border_radius=5,
                            margin=ft.margin.only(bottom=5)
                        )
                        for i, result in enumerate(results[:3])
                    ]
                ]),
                padding=20,
            ),
            elevation=3,
        )

        result_container.controls.extend([summary_card, breakdown_card, preview_card])
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

    def update_analyze_button_state(self, button, input_text):
        """Update analyze button state based on validation"""
        if not self.resources_loaded:
            button.disabled = True
            if self.status_text:
                self.status_text.value = "Please restart the application - resources failed to load"
                self.status_text.color = ft.Colors.RED
                self.status_text.visible = True

        is_valid, error_msg = self.validate_input(input_text)
        button.disabled = not is_valid
        if is_valid:
            button.text = "Analyze & Generate Response"
            button.disabled = False
            if self.status_text:
                self.status_text.visible = False
        else:
            button.text = "Enter Valid Comment"
            if self.status_text:
                self.status_text.value = error_msg
                self.status_text.color = ft.Colors.ORANGE_600
                self.status_text.visible = True

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
        page.window_height = 1000
        page.window_resizable = True
        page.scroll = ft.ScrollMode.AUTO

        # AppBar with help icon
        page.appbar = ft.AppBar(
            center_title=True,
            actions=[
                ft.IconButton(
                    icon=ft.Icons.HELP_OUTLINE,
                    icon_color=ft.Colors.BLUE_800,
                    tooltip="Help & Instructions",
                    on_click=lambda e: page.open(self.help_dialog)
                )
            ]
        )

        # Header
        header = ft.Container(
            content=ft.Column([
                ft.Text(
                    "Ziona's Consulting App",
                    size=28,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_800
                ),
                ft.Text(
                    "Hi! I'm Ziona, I will analyze the problematic comments and suggest educational responses",
                    size=16,
                    color=ft.Colors.GREY_700
                ),
            ], tight=True),
            padding=ft.padding.only(bottom=20)
        )

        # Mode selection tabs
        def on_tab_change(e):
            self.current_mode = "single" if e.control.selected_index == 0 else "batch"
            single_mode_content.visible = (self.current_mode == "single")
            batch_mode_content.visible = (self.current_mode == "batch")
            result_container.controls.clear()
            page.update()

        mode_tabs = ft.Tabs(
            selected_index=0,
            on_change=on_tab_change,
            tabs=[
                ft.Tab(text="Single Comment", icon=ft.Icons.CHAT),
                ft.Tab(text="Multiple Comments", icon=ft.Icons.TABLE_VIEW),
            ],
        )

        # === SINGLE MODE CONTENT ===
        comment_input = ft.TextField(
            label="Enter a potentially anti-Semitic/ anti-Israeli comment",
            multiline=True,
            min_lines=3,
            max_lines=8,
            border_color=ft.Colors.BLUE_400,
            focused_border_color=ft.Colors.BLUE_600,
            counter_text="",
            max_length=5000,
        )

        validation_text = ft.Text("", size=12, color=ft.Colors.ORANGE_600, visible=False)

        self.analyze_button = ft.ElevatedButton(
            text="Enter Valid Comment",
            icon=ft.Icons.ANALYTICS,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_600,
                padding=ft.padding.symmetric(horizontal=20, vertical=10)
            ),
            height=50
        )

        single_mode_content = ft.Column([
            comment_input,
            validation_text,
            ft.Container(height=10),
            self.analyze_button,
        ], visible=True)

        # === BATCH MODE CONTENT ===
        file_picker = ft.FilePicker()
        page.overlay.append(file_picker)

        selected_file_text = ft.Text("No file selected", size=14, color=ft.Colors.GREY_600)
        file_validation_text = ft.Text("", size=12, color=ft.Colors.ORANGE_600, visible=False)

        def on_file_picked(e: ft.FilePickerResultEvent):
            if e.files:
                file_path = e.files[0].path
                selected_file_text.value = f"Selected: {os.path.basename(file_path)}"
                selected_file_text.color = ft.Colors.GREEN_700

                # Validate file
                self.status_text.visible = False
                self.is_valid_comment_df, message = self.validate_excel_file(file_path)
                if self.is_valid_comment_df:
                    file_validation_text.value = message
                    file_validation_text.color = ft.Colors.GREEN_600
                    file_validation_text.visible = True
                    self.batch_analyze_button.disabled = not self.resources_loaded
                    self.batch_analyze_button.text = "Analyze Excel File" if self.resources_loaded else "Resources not loaded"
                else:
                    file_validation_text.value = message
                    file_validation_text.color = ft.Colors.RED
                    file_validation_text.visible = True
                    self.batch_analyze_button.disabled = True
                    self.batch_analyze_button.text = "Invalid File"

                page.update()

        file_picker.on_result = on_file_picked

        upload_button = ft.ElevatedButton(
            text="Select Excel File",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda _: file_picker.pick_files(
                dialog_title="Select Excel file with comments",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["xlsx", "xls"]
            ),
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREEN_600,
                padding=ft.padding.symmetric(horizontal=20, vertical=10)
            ),
            height=50
        )

        self.batch_analyze_button = ft.ElevatedButton(
            text="Select File First",
            icon=ft.Icons.BATCH_PREDICTION,
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_600,
                padding=ft.padding.symmetric(horizontal=20, vertical=10)
            ),
            height=50
        )

        file_instructions = ft.Container(
            content=ft.Column([
                ft.Text("Excel File Requirements:", size=14, weight=ft.FontWeight.BOLD),
                ft.Text("• File must contain a column named 'comment' (case insensitive)", size=12),
                ft.Text("• Maximum 100 comments per file", size=12),
                ft.Text("• Supported formats: .xlsx, .xls", size=12),
                ft.Text("• Results will be saved as CSV file", size=12),
            ]),
            bgcolor=ft.Colors.BLUE_50,
            padding=15,
            border_radius=8,
            border=ft.border.all(1, ft.Colors.BLUE_200)
        )

        batch_mode_content = ft.Column([
            file_instructions,
            ft.Container(height=10),
            upload_button,
            selected_file_text,
            file_validation_text,
            ft.Container(height=10),
            self.batch_analyze_button,
        ], visible=False)

        # Progress and status
        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.status_text = ft.Text("", size=14, color=ft.Colors.BLUE_600, visible=False)

        # Results container
        result_container = ft.Column(spacing=10)

        # Event handlers
        def on_single_analyze_click(e):
            is_valid, error_msg = self.validate_input(comment_input.value)
            if not is_valid or not self.resources_loaded:
                self.show_notification(error_msg or "Resources not available", ft.Colors.RED)
                return

            self.progress_bar.visible = True
            self.status_text.visible = True
            validation_text.visible = False
            self.progress_bar.value = 0.1
            self.status_text.value = "Initializing analysis..."
            self.status_text.color = ft.Colors.BLUE_600
            result_container.controls.clear()

            self.analyze_button.disabled = True
            self.analyze_button.text = "Processing..."
            page.update()

            threading.Thread(
                target=self.analyze_comment_async,
                args=(comment_input.value.strip(), result_container),
                daemon=True
            ).start()

        def on_batch_analyze_click(e):
            if not file_picker.result or not file_picker.result.files:
                self.show_notification("Please select an Excel file first", ft.Colors.RED)
                return

            file_path = file_picker.result.files[0].path

            if not self.is_valid_comment_df or not self.resources_loaded:
                self.show_notification("Invalid file or resources not available", ft.Colors.RED)
                return

            self.progress_bar.visible = True
            self.status_text.visible = True
            file_validation_text.visible = False
            self.progress_bar.value = 0.1
            self.status_text.value = "Starting analysis..."
            self.status_text.color = ft.Colors.BLUE_600
            result_container.controls.clear()

            self.batch_analyze_button.disabled = True
            self.batch_analyze_button.text = "Processing..."
            page.update()

            threading.Thread(
                target=self.process_batch_analysis,
                args=([result_container]),
                daemon=True
            ).start()

        def on_input_change(e):
            self.update_analyze_button_state(self.analyze_button, comment_input.value)

        # Set event handlers
        comment_input.on_change = on_input_change
        self.analyze_button.on_click = on_single_analyze_click
        self.batch_analyze_button.on_click = on_batch_analyze_click

        # Main layout
        main_content = ft.Column([
            header,
            mode_tabs,
            ft.Container(height=20),
            single_mode_content,
            batch_mode_content,
            ft.Container(height=20),
            self.progress_bar,
            self.status_text,
            ft.Container(height=10),
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
        self.update_analyze_button_state(self.analyze_button, "")

        if not resources_loaded:
            self.show_notification(
                "Warning: Could not load resources. Please check file paths.",
                ft.Colors.RED
            )


def main():
    app = ZionaApp()
    ft.app(target=ZionaApp().main)


if __name__ == "__main__":
    main()
