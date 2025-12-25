#!/usr/bin/env python3
"""
Custom Data Annotation Interface for Ansible Log Error Annotations
"""

import gradio as gr
import json
import logging
import os
from datetime import datetime
from typing import Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import psycopg2

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class DataAnnotationApp:
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "annotation.json")
        self.current_index = 0
        self.data = []
        self.all_data = []  # Store all data before cluster filtering
        self.feedback_data = []
        self.show_cluster_sample = False  # Toggle state for cluster sampling

        # Get table name from environment variable
        self.table_name = os.getenv("ALERTS_TABLE_NAME", "grafanaalert")

        # Initialize sync engine
        self.engine = create_engine(
            os.getenv("DATABASE_URL")
            .replace("+asyncpg", "")
            .replace("postgresql", "postgresql+psycopg2")
        )

        # Ensure data directory exists
        os.makedirs(self.feedback_dir, exist_ok=True)

        self.load_data()
        self.load_feedback()

    def load_data(self):
        """Load the pipeline output data from PostgreSQL."""
        logger.debug(f"Loading data from table: {self.table_name}")
        try:
            with Session(self.engine) as session:
                # Use raw SQL to query the table dynamically
                query = text(f"""
                    SELECT 
                        id,
                        "logMessage",
                        "logSummary", 
                        "stepByStepSolution",
                        "contextForStepByStepSolution",
                        "logCluster",
                        "log_labels",
                        "needMoreContext"
                    FROM {self.table_name}
                    ORDER BY id
                """)

                result = session.execute(query)
                rows = result.fetchall()

                # Convert to the expected data format
                self.all_data = []
                for row in rows:
                    # Parse labels JSON if it exists
                    labels = row.log_labels if hasattr(row, "log_labels") else {}

                    data_entry = {
                        "id": row.id,
                        "filename": labels.get("filename", "unknown")
                        if isinstance(labels, dict)
                        else "unknown",
                        "line_number": labels.get("line_number", "")
                        if isinstance(labels, dict)
                        else "",
                        "logMessage": row.logMessage or "No log content",
                        "summary": row.logSummary or "No summary available",
                        "context_for_solution": row.contextForStepByStepSolution
                        or "No context available",
                        "step_by_step_solution": row.stepByStepSolution
                        or "No solution available",
                        "log_cluster": row.logCluster
                        if hasattr(row, "logCluster")
                        else None,
                        "need_more_context": row.needMoreContext
                        if hasattr(row, "needMoreContext")
                        else False,
                        "labels": labels if isinstance(labels, dict) else {},
                    }
                    self.all_data.append(data_entry)

                # Initialize data with all entries
                self.data = self.all_data.copy()

                logger.info(
                    f"Loaded {len(self.all_data)} data entries from table '{self.table_name}'"
                )
        except psycopg2.errors.UndefinedTable as _:
            logger.warning(
                f"Table '{self.table_name}' not initiated. Make sure the data is ingested into the database"
            )
            self.all_data = []
            self.data = []
        except Exception as e:
            logger.error(
                f"Error loading data from database table '{self.table_name}': {e}"
            )
            self.all_data = []
            self.data = []

    def load_feedback(self):
        """Load existing feedback data."""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, "r") as f:
                    self.feedback_data = json.load(f)
            else:
                self.feedback_data = []
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            self.feedback_data = []

    def toggle_cluster_sampling(
        self, show_sample: bool
    ) -> Tuple[str, str, str, str, str, str, str, str, str, bool, bool, bool, str, str]:
        """Toggle between showing all rows or one sample per cluster."""
        self.show_cluster_sample = show_sample

        if show_sample:
            # Group by cluster and take one sample from each
            cluster_samples = {}
            for entry in self.all_data:
                cluster_id = entry.get("log_cluster")
                # If no cluster, treat each entry as its own cluster
                if cluster_id is None:
                    cluster_id = f"_no_cluster_{entry.get('id')}"

                # Keep only the first entry from each cluster
                if cluster_id not in cluster_samples:
                    cluster_samples[cluster_id] = entry

            self.data = list(cluster_samples.values())
            logger.info(
                f"Cluster sampling enabled: showing {len(self.data)} samples from {len(self.all_data)} total entries"
            )
        else:
            # Show all data
            self.data = self.all_data.copy()
            logger.info(
                f"Cluster sampling disabled: showing all {len(self.data)} entries"
            )

        # Reset to first entry
        self.current_index = 0
        return self.get_current_entry()

    def save_feedback(
        self,
        feedback: str,
        golden_solution: str = "",
        expected_behavior: str = "",
        is_context_correct: bool = False,
        need_more_context: bool = False,
        need_more_context_reason: str = "",
    ) -> str:
        """Save feedback, golden solution, expected behavior, and context info for current data entry."""
        if not self.data:
            logger.debug("save_feedback called with no data available")
            return "No data available"

        current_entry = self.data[self.current_index]
        logger.debug(f"Saving feedback for entry index {self.current_index}")

        # Create feedback entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "index": self.current_index,
            "filename": current_entry.get("filename", ""),
            "line_number": current_entry.get("line_number", ""),
            "feedback": feedback,
            "golden_stepByStepSolution": golden_solution,
            "expected_behavior": expected_behavior,
            "golden_is_context_correct": is_context_correct,
            "golden_need_more_context": need_more_context,
            "golden_need_more_context_reason": need_more_context_reason,
            "logMessage": current_entry.get("logMessage", "No line context"),
            "logSummary": current_entry.get("summary", ""),
            "stepByStepSolution": current_entry.get("step_by_step_solution", ""),
            "contextForStepByStepSolution": current_entry.get(
                "context_for_solution", ""
            ),
        }

        # Remove any existing feedback for this entry
        self.feedback_data = [
            f for f in self.feedback_data if f["index"] != self.current_index
        ]

        # Add new feedback if not empty (either feedback, golden_solution, expected_behavior, or context info)
        if (
            feedback.strip()
            or golden_solution.strip()
            or expected_behavior.strip()
            or is_context_correct
            or need_more_context
            or need_more_context_reason.strip()
        ):
            self.feedback_data.append(feedback_entry)

        # Save to file
        try:
            with open(self.feedback_file, "w") as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.info(
                f"Feedback saved for entry {self.current_index + 1} (total: {len(self.feedback_data)} entries)"
            )
            return f"Feedback saved for entry {self.current_index + 1}"
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return f"Error saving feedback: {e}"

    def get_current_entry(
        self,
    ) -> Tuple[str, str, str, str, str, str, str, str, str, bool, bool, bool, str, str]:
        """Get current data entry for display."""
        if not self.data:
            return (
                "No data",
                "No data",
                "No data",
                "No data",
                "",
                "",
                "",
                "0 / 0",
                "",
                False,
                False,
                False,
                "",
                "{}",
            )

        entry = self.data[self.current_index]

        # Format error log with syntax highlighting
        log_content = entry.get("logMessage", "No log content")

        # Format summary
        summary = entry.get("summary", "No summary available")

        # Get context for solution
        context_for_solution = entry.get(
            "context_for_solution",
            "No context available",
        )

        # For now, use a placeholder for step-by-step solution
        # In the future, you can extend this to fetch from your database or add it to your JSON
        step_by_step = entry.get(
            "step_by_step_solution",
            "Step-by-step solution not available for this entry.\n\n"
            "This would typically contain:\n"
            "1. Problem identification\n"
            "2. Root cause analysis\n"
            "3. Recommended solution steps\n"
            "4. Prevention measures",
        )

        # Get need_more_context from DB
        db_need_more_context = entry.get("need_more_context", False)

        # Get existing feedback, golden solution, expected behavior, and context info for this entry
        existing_feedback = ""
        existing_golden_solution = ""
        existing_expected_behavior = ""
        existing_is_context_correct = False
        existing_need_more_context = False
        existing_need_more_context_reason = ""
        for f in self.feedback_data:
            if f["index"] == self.current_index:
                existing_feedback = f.get("feedback", "")
                existing_golden_solution = f.get("golden_stepByStepSolution", "")
                existing_expected_behavior = f.get("expected_behavior", "")
                existing_is_context_correct = f.get("golden_is_context_correct", False)
                existing_need_more_context = f.get("golden_need_more_context", False)
                existing_need_more_context_reason = f.get(
                    "golden_need_more_context_reason", ""
                )
                break

        # Navigation info
        nav_info = f"{self.current_index + 1} / {len(self.data)}"

        # Format labels for display
        labels = entry.get("labels", {})
        labels_json = json.dumps(labels, indent=2)

        return (
            log_content,
            summary,
            context_for_solution,
            step_by_step,
            existing_feedback,
            existing_golden_solution,
            existing_expected_behavior,
            nav_info,
            step_by_step,  # raw text for copying
            db_need_more_context,  # from DB
            existing_is_context_correct,  # from user annotation
            existing_need_more_context,  # from user annotation
            existing_need_more_context_reason,  # from user annotation
            labels_json,  # labels in JSON format
        )

    def navigate(
        self, direction: int
    ) -> Tuple[str, str, str, str, str, str, str, str, str, bool, bool, bool, str, str]:
        """Navigate through data entries."""
        if not self.data:
            return self.get_current_entry()

        self.current_index = max(
            0, min(len(self.data) - 1, self.current_index + direction)
        )
        return self.get_current_entry()

    def go_to_index(
        self, index: int
    ) -> Tuple[str, str, str, str, str, str, str, str, str, bool, bool, bool, str, str]:
        """Jump to specific index."""
        if not self.data:
            return self.get_current_entry()

        self.current_index = max(0, min(len(self.data) - 1, index))
        return self.get_current_entry()

    def get_feedback_table(self) -> str:
        """Generate HTML table of all feedback."""
        if not self.feedback_data:
            return "<div style='padding: 20px; text-align: center; color: #94a3b8; background-color: #1e293b; border: 1px solid #475569; border-radius: 8px;'>No feedback entries yet</div>"

        # Sort by timestamp (most recent first)
        sorted_feedback = sorted(
            self.feedback_data, key=lambda x: x["timestamp"], reverse=True
        )

        html = """
        <div style="max-height: 300px; overflow-y: auto; border: 1px solid #475569; border-radius: 8px; background-color: #1e293b;">
            <table style="width: 100%; border-collapse: collapse; font-size: 12px; background-color: #1e293b; color: #e2e8f0;">
                <thead style="background-color: #334155; position: sticky; top: 0;">
                    <tr>
                        <th style="padding: 8px; border: 1px solid #475569; color: #f1f5f9; font-weight: 600;">Index</th>
                        <th style="padding: 8px; border: 1px solid #475569; color: #f1f5f9; font-weight: 600;">File</th>
                        <th style="padding: 8px; border: 1px solid #475569; color: #f1f5f9; font-weight: 600;">Log</th>
                        <th style="padding: 8px; border: 1px solid #475569; color: #f1f5f9; font-weight: 600;">Feedback</th>
                        <th style="padding: 8px; border: 1px solid #475569; color: #f1f5f9; font-weight: 600;">Time</th>
                    </tr>
                </thead>
                <tbody>
        """

        for feedback in sorted_feedback:
            timestamp = datetime.fromisoformat(feedback["timestamp"]).strftime(
                "%m/%d %H:%M"
            )
            feedback_text = (
                feedback["feedback"][:50] + "..."
                if len(feedback["feedback"]) > 50
                else feedback["feedback"]
            )

            html += f"""
                <tr style="border-bottom: 1px solid #475569; background-color: #1e293b;" onmouseover="this.style.backgroundColor='#334155'" onmouseout="this.style.backgroundColor='#1e293b'">
                    <td style="padding: 8px; border: 1px solid #475569; color: #e2e8f0;">{feedback["index"] + 1}</td>
                    <td style="padding: 8px; border: 1px solid #475569; color: #e2e8f0;" title="{feedback["filename"]}">{feedback["filename"][:20]}...</td>
                    <td style="padding: 8px; border: 1px solid #475569; color: #e2e8f0;" title="{feedback["logMessage"]}">{feedback["logMessage"]}</td>
                    <td style="padding: 8px; border: 1px solid #475569; color: #e2e8f0;" title="{feedback["feedback"]}">{feedback_text}</td>
                    <td style="padding: 8px; border: 1px solid #475569; color: #e2e8f0;">{timestamp}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """
        return html


def create_app():
    """Create the Gradio interface."""
    # Initialize the app
    feedback_dir = "data/feedback"
    app = DataAnnotationApp(feedback_dir)

    # Custom CSS for dark theme
    css = """
    .summary-box {
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #475569 !important;
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        line-height: 1.6;
    }
    .basic_box {
        font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace; 
        font-size: 12px; 
        line-height: 1.5; 
        background-color: #0f172a !important; 
        color: #e2e8f0 !important; 
        padding: 8px; 
        border-radius: 8px; 
        border: 1px solid #334155 !important;
        max-height: 420px;
    }
    .feedback-box {
        min-height: 200px;
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
    }
    .nav-button {
        min-width: 60px;
        margin: 2px;
        background-color: #475569 !important;
        color: #f1f5f9 !important;
        border: 1px solid #64748b !important;
        padding: 4px 8px !important;
        font-size: 0.9em !important;
    }
    .nav-button:hover {
        background-color: #64748b !important;
        color: #ffffff !important;
    }
    /* Dark theme for feedback table */
    table {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    th {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    td {
        border-color: #475569 !important;
    }
    tr:hover {
        background-color: #334155 !important;
    }
    /* Override any light theme remnants */
    .gradio-container {
        background-color: #0f172a !important;
    }
    /* Need More Context badge */
    .need-context-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.9em;
        font-weight: 600;
        margin-left: 8px;
    }
    .need-context-true {
        background-color: #3b82f6;
        color: #f1f5f9;
    }
    .need-context-false {
        background-color: #475569;
        color: #94a3b8;
    }
    """

    with gr.Blocks(
        css=css,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ).set(
            body_background_fill="*neutral_950",
            body_text_color="*neutral_100",
            block_background_fill="*neutral_900",
            block_border_color="*neutral_700",
            input_background_fill="*neutral_800",
            button_primary_background_fill="*primary_600",
            button_primary_text_color="white",
        ),
        title="Ansible Log Annotation Interface",
    ) as interface:
        gr.Markdown("# Ansible Log Data Annotation Interface")
        gr.Markdown(
            "Annotate pipeline outputs with feedback on failure modes and solution quality."
        )

        # Compact navigation controls - all in one row
        with gr.Row():
            cluster_sample_toggle = gr.Checkbox(
                label="One sample per cluster",
                value=False,
                interactive=True,
                scale=2,
            )
            prev_btn = gr.Button(
                "← Prev",
                elem_classes="nav-button",
                scale=1,
            )
            nav_info = gr.Textbox(
                label="Position",
                interactive=False,
                value="Loading...",
                container=False,
                scale=1,
            )
            next_btn = gr.Button(
                "Next →",
                elem_classes="nav-button",
                scale=1,
            )
            jump_input = gr.Number(
                label="Jump to",
                minimum=1,
                value=1,
                step=1,
                precision=0,
                scale=1,
                container=False,
            )
            jump_btn = gr.Button(
                "Go",
                scale=1,
            )

        # Main content area - Reorganized into rows

        # Row 1: Inputs
        gr.Markdown("## 📥 Inputs")
        error_log = gr.Textbox(
            elem_classes="basic_box",
            label="Error Log",
            lines=5,
            max_lines=20,
            interactive=False,
        )

        # Row 2: Outputs (toggleable)

        with gr.Row():
            gr.Markdown("## 🤖 AI-Generated Outputs")
            show_outputs_toggle = gr.Checkbox(
                label="🤖 Show AI-Generated Outputs",
                value=True,
                interactive=True,
            )

        # Need More Context indicator from DB
        db_need_more_context_display = gr.HTML(
            value="<div class='need-context-badge need-context-false'>🤖 AI Assessment - Need More Context: Loading...</div>",
            label="AI Assessment",
        )

        with gr.Group(visible=True) as outputs_section:
            with gr.Row():
                show_summary_toggle = gr.Checkbox(
                    label="Show Summary",
                    value=True,
                    interactive=True,
                    scale=1,
                )
                show_context_toggle = gr.Checkbox(
                    label="Show Context for Solution",
                    value=True,
                    interactive=True,
                    scale=1,
                )
                show_solution_toggle = gr.Checkbox(
                    label="Show Step-by-Step Solution",
                    value=True,
                    interactive=True,
                    scale=1,
                )
            with gr.Column():
                with gr.Column():
                    summary_title = gr.Markdown(
                        "### 🦾 Generated Summary", visible=True
                    )
                    summary = gr.Textbox(
                        lines=2,
                        max_lines=5,
                        elem_classes="basic_box",
                        visible=True,
                        show_label=False,
                    )

                with gr.Column():
                    context_title = gr.Markdown(
                        "### 🔍 Context for Solution", visible=True
                    )
                    context_for_solution = gr.Textbox(
                        elem_classes="basic_box",
                        show_label=False,
                        lines=8,
                        max_lines=8,
                        interactive=False,
                    )

                with gr.Column():
                    with gr.Row():
                        solution_title = gr.Markdown(
                            "### 🦾 Step-by-Step Solution", visible=True
                        )
                        copy_solution_btn = gr.Button(
                            "📋 Copy",
                            size="sm",
                            scale=0,
                            min_width=80,
                            visible=True,
                        )
                    step_by_step = gr.Markdown(
                        value="",
                        label="🤖 Generated Step-by-Step Solution",
                        elem_classes="basic_box",
                        visible=True,
                    )
                    # Hidden textbox to store raw markdown for copying
                    step_by_step_raw = gr.Textbox(
                        value="",
                        visible=False,
                        elem_id="step_by_step_raw",
                    )

        # Row 3: Feedback columns
        gr.Markdown("## 📝 Human Annotations")

        with gr.Row():
            annotation_view_toggle = gr.Radio(
                choices=[
                    "Feedback & Failure Mode",
                    "Golden Solution",
                    "Expected Behavior",
                    "Context",
                ],
                value="Feedback & Failure Mode",
                label="📝 Human Annotations",
                interactive=True,
                scale=1,
            )

        with gr.Row():
            with gr.Column(scale=1):
                feedback_text = gr.Textbox(
                    label="Feedback & Failure Mode Analysis",
                    lines=15,
                    placeholder="Describe any issues with the summary or solution:\n"
                    "- Is the summary accurate?\n"
                    "- Is the solution appropriate?\n"
                    "- What failure modes do you observe?\n"
                    "- Any missing information?",
                    elem_classes="feedback-box",
                    visible=True,
                )

                golden_solution_text = gr.Textbox(
                    label="Golden Step-by-Step Solution (Optional)",
                    lines=15,
                    placeholder="Provide your golden/ideal step-by-step solution:\n"
                    "1. Clear problem identification\n"
                    "2. Root cause analysis\n"
                    "3. Step-by-step solution\n"
                    "4. Prevention measures\n\n"
                    "This will be saved alongside your feedback for comparison with AI-generated solutions.",
                    elem_classes="feedback-box",
                    visible=False,
                )

                expected_behavior_text = gr.Textbox(
                    label="What Should Happen to Produce the Right Solution",
                    lines=15,
                    placeholder="Describe what should happen to produce the correct solution:\n"
                    "- What are the expected actions or steps?\n"
                    "- What information or context is needed?\n"
                    "- What should the ideal outcome look like?\n"
                    "- What conditions need to be met?\n\n"
                    "This helps document the expected behavior needed to generate the right solution.",
                    elem_classes="feedback-box",
                    visible=False,
                )

                # Context tab
                need_more_context_column = gr.Column(visible=False)
                with need_more_context_column:
                    gr.Markdown("### 🔍 Context Assessment")
                    is_context_correct_toggle = gr.Checkbox(
                        label="Is the returned context correct?",
                        value=False,
                        interactive=True,
                    )
                    need_more_context_toggle = gr.Checkbox(
                        label="Need More Context",
                        value=False,
                        interactive=True,
                    )
                    need_more_context_reason = gr.Textbox(
                        label="Explain Why More Context is Needed",
                        lines=10,
                        placeholder="Explain what additional context or information is needed:\n"
                        "- What specific information is missing?\n"
                        "- What context would help provide a better solution?\n"
                        "- What additional details are required?\n"
                        "- How would this context improve the solution?",
                        elem_classes="feedback-box",
                    )

        # Save feedback button and status
        with gr.Row():
            save_feedback_btn = gr.Button(
                "💾 Save Feedback", variant="primary", scale=1
            )
            feedback_status = gr.Textbox(
                label="Status", interactive=False, lines=1, scale=2
            )

        # Labels section at the bottom
        gr.Markdown("## 🏷️ Log Labels")
        labels_display = gr.Code(
            label="All Labels (JSON)",
            language="json",
            interactive=False,
            lines=10,
            value="{}",
        )

        # Initialize the interface
        def init_interface():
            result = app.get_current_entry()
            # Extract the db_need_more_context value (index 9) from database
            db_need_more_context = result[9]
            # Create HTML badge for need_more_context display
            badge_class = (
                "need-context-true" if db_need_more_context else "need-context-false"
            )
            badge_text = "Yes" if db_need_more_context else "No"
            badge_html = f"<div class='need-context-badge {badge_class}'>🤖 AI Assessment - Need More Context: {badge_text}</div>"
            # Return all values except db_need_more_context (index 9), plus the badge_html
            # result[:9] = first 9 items, result[10:13] = items 10-12 (is_context_correct, need_more_context, reason), result[13] = labels
            return result[:9] + result[10:13] + (badge_html, result[13])

        # Event handlers
        def handle_save_feedback(
            feedback,
            golden_solution,
            expected_behavior,
            is_context_correct,
            need_more_context,
            need_more_context_reason,
        ):
            status = app.save_feedback(
                feedback,
                golden_solution,
                expected_behavior,
                is_context_correct,
                need_more_context,
                need_more_context_reason,
            )
            return status

        def handle_navigate_prev(
            show_outputs, show_summary, show_context, show_solution
        ):
            (
                log_content,
                summary_content,
                context_content,
                step_content,
                feedback,
                golden,
                expected,
                nav,
                raw_step,
                db_need_more_context,
                user_is_context_correct,
                user_need_more_context,
                user_need_more_context_reason,
                labels_json,
            ) = app.navigate(-1)
            # Create HTML badge for need_more_context display
            badge_class = (
                "need-context-true" if db_need_more_context else "need-context-false"
            )
            badge_text = "Yes" if db_need_more_context else "No"
            badge_html = f"<div class='need-context-badge {badge_class}'>🤖 AI Assessment - Need More Context: {badge_text}</div>"
            # Return content updates with visibility + preserved toggle states
            return (
                log_content,  # error_log
                gr.update(visible=show_summary),  # summary_title
                gr.update(
                    value=summary_content, visible=show_summary
                ),  # summary with visibility
                gr.update(visible=show_context),  # context_title
                gr.update(
                    value=context_content, visible=show_context
                ),  # context_for_solution with visibility
                gr.update(visible=show_solution),  # solution_title
                gr.update(
                    value=step_content, visible=show_solution
                ),  # step_by_step with visibility
                feedback,  # feedback_text
                golden,  # golden_solution_text
                expected,  # expected_behavior_text
                nav,  # nav_info
                show_outputs,  # preserve show_outputs_toggle
                show_summary,  # preserve show_summary_toggle
                show_context,  # preserve show_context_toggle
                show_solution,  # preserve show_solution_toggle
                gr.update(visible=show_outputs),  # outputs_section visibility
                raw_step,  # step_by_step_raw
                badge_html,  # db_need_more_context_display
                user_is_context_correct,  # is_context_correct_toggle
                user_need_more_context,  # need_more_context_toggle
                user_need_more_context_reason,  # need_more_context_reason
                labels_json,  # labels_display
            )

        def handle_navigate_next(
            show_outputs, show_summary, show_context, show_solution
        ):
            (
                log_content,
                summary_content,
                context_content,
                step_content,
                feedback,
                golden,
                expected,
                nav,
                raw_step,
                db_need_more_context,
                user_is_context_correct,
                user_need_more_context,
                user_need_more_context_reason,
                labels_json,
            ) = app.navigate(1)
            # Create HTML badge for need_more_context display
            badge_class = (
                "need-context-true" if db_need_more_context else "need-context-false"
            )
            badge_text = "Yes" if db_need_more_context else "No"
            badge_html = f"<div class='need-context-badge {badge_class}'>🤖 AI Assessment - Need More Context: {badge_text}</div>"
            # Return content updates with visibility + preserved toggle states
            return (
                log_content,  # error_log
                gr.update(visible=show_summary),  # summary_title
                gr.update(
                    value=summary_content, visible=show_summary
                ),  # summary with visibility
                gr.update(visible=show_context),  # context_title
                gr.update(
                    value=context_content, visible=show_context
                ),  # context_for_solution with visibility
                gr.update(visible=show_solution),  # solution_title
                gr.update(
                    value=step_content, visible=show_solution
                ),  # step_by_step with visibility
                feedback,  # feedback_text
                golden,  # golden_solution_text
                expected,  # expected_behavior_text
                nav,  # nav_info
                show_outputs,  # preserve show_outputs_toggle
                show_summary,  # preserve show_summary_toggle
                show_context,  # preserve show_context_toggle
                show_solution,  # preserve show_solution_toggle
                gr.update(visible=show_outputs),  # outputs_section visibility
                raw_step,  # step_by_step_raw
                badge_html,  # db_need_more_context_display
                user_is_context_correct,  # is_context_correct_toggle
                user_need_more_context,  # need_more_context_toggle
                user_need_more_context_reason,  # need_more_context_reason
                labels_json,  # labels_display
            )

        def handle_jump(index, show_outputs, show_summary, show_context, show_solution):
            if index is not None:
                (
                    log_content,
                    summary_content,
                    context_content,
                    step_content,
                    feedback,
                    golden,
                    expected,
                    nav,
                    raw_step,
                    db_need_more_context,
                    user_is_context_correct,
                    user_need_more_context,
                    user_need_more_context_reason,
                    labels_json,
                ) = app.go_to_index(int(index) - 1)
            else:
                (
                    log_content,
                    summary_content,
                    context_content,
                    step_content,
                    feedback,
                    golden,
                    expected,
                    nav,
                    raw_step,
                    db_need_more_context,
                    user_is_context_correct,
                    user_need_more_context,
                    user_need_more_context_reason,
                    labels_json,
                ) = app.get_current_entry()
            # Create HTML badge for need_more_context display
            badge_class = (
                "need-context-true" if db_need_more_context else "need-context-false"
            )
            badge_text = "Yes" if db_need_more_context else "No"
            badge_html = f"<div class='need-context-badge {badge_class}'>🤖 AI Assessment - Need More Context: {badge_text}</div>"
            # Return content updates with visibility + preserved toggle states
            return (
                log_content,  # error_log
                gr.update(visible=show_summary),  # summary_title
                gr.update(
                    value=summary_content, visible=show_summary
                ),  # summary with visibility
                gr.update(visible=show_context),  # context_title
                gr.update(
                    value=context_content, visible=show_context
                ),  # context_for_solution with visibility
                gr.update(visible=show_solution),  # solution_title
                gr.update(
                    value=step_content, visible=show_solution
                ),  # step_by_step with visibility
                feedback,  # feedback_text
                golden,  # golden_solution_text
                expected,  # expected_behavior_text
                nav,  # nav_info
                show_outputs,  # preserve show_outputs_toggle
                show_summary,  # preserve show_summary_toggle
                show_context,  # preserve show_context_toggle
                show_solution,  # preserve show_solution_toggle
                gr.update(visible=show_outputs),  # outputs_section visibility
                raw_step,  # step_by_step_raw
                badge_html,  # db_need_more_context_display
                user_is_context_correct,  # is_context_correct_toggle
                user_need_more_context,  # need_more_context_toggle
                user_need_more_context_reason,  # need_more_context_reason
                labels_json,  # labels_display
            )

        def handle_cluster_toggle(show_sample):
            result = app.toggle_cluster_sampling(show_sample)
            # Extract the db_need_more_context value (index 9) from database
            db_need_more_context = result[9]
            # Create HTML badge for need_more_context display
            badge_class = (
                "need-context-true" if db_need_more_context else "need-context-false"
            )
            badge_text = "Yes" if db_need_more_context else "No"
            badge_html = f"<div class='need-context-badge {badge_class}'>🤖 AI Assessment - Need More Context: {badge_text}</div>"
            # Return all values except db_need_more_context (index 9), plus the badge_html
            # result[:9] = first 9 items, result[10:13] = items 10-12 (is_context_correct, need_more_context, reason), result[13] = labels
            return result[:9] + result[10:13] + (badge_html, result[13])

        def handle_outputs_toggle(show_outputs):
            return gr.update(visible=show_outputs)

        def handle_summary_toggle(show_summary):
            return (
                gr.update(visible=show_summary),  # summary_title
                gr.update(visible=show_summary),  # summary
            )

        def handle_context_toggle(show_context):
            return (
                gr.update(visible=show_context),  # context_title
                gr.update(visible=show_context),  # context_for_solution
            )

        def handle_solution_toggle(show_solution):
            return (
                gr.update(visible=show_solution),  # solution_title
                gr.update(visible=show_solution),  # step_by_step
                gr.update(visible=show_solution),  # copy_solution_btn
            )

        def handle_annotation_view_toggle(view_selection):
            """Toggle between feedback, golden solution, expected behavior, and context views."""
            show_feedback = view_selection == "Feedback & Failure Mode"
            show_golden = view_selection == "Golden Solution"
            show_expected = view_selection == "Expected Behavior"
            show_need_context = view_selection == "Context"
            return (
                gr.update(visible=show_feedback),  # feedback_text
                gr.update(visible=show_golden),  # golden_solution_text
                gr.update(visible=show_expected),  # expected_behavior_text
                gr.update(visible=show_need_context),  # need_more_context_column
            )

        # Bind events
        interface.load(
            init_interface,
            outputs=[
                error_log,
                summary,
                context_for_solution,
                step_by_step,
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                nav_info,
                step_by_step_raw,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                db_need_more_context_display,
                labels_display,
            ],
        )

        prev_btn.click(
            handle_navigate_prev,
            inputs=[
                show_outputs_toggle,
                show_summary_toggle,
                show_context_toggle,
                show_solution_toggle,
            ],
            outputs=[
                error_log,
                summary_title,
                summary,
                context_title,
                context_for_solution,
                solution_title,
                step_by_step,
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                nav_info,
                show_outputs_toggle,
                show_summary_toggle,
                show_context_toggle,
                show_solution_toggle,
                outputs_section,
                step_by_step_raw,
                db_need_more_context_display,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                labels_display,
            ],
        )

        next_btn.click(
            handle_navigate_next,
            inputs=[
                show_outputs_toggle,
                show_summary_toggle,
                show_context_toggle,
                show_solution_toggle,
            ],
            outputs=[
                error_log,
                summary_title,
                summary,
                context_title,
                context_for_solution,
                solution_title,
                step_by_step,
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                nav_info,
                show_outputs_toggle,
                show_summary_toggle,
                show_context_toggle,
                show_solution_toggle,
                outputs_section,
                step_by_step_raw,
                db_need_more_context_display,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                labels_display,
            ],
        )

        jump_btn.click(
            handle_jump,
            inputs=[
                jump_input,
                show_outputs_toggle,
                show_summary_toggle,
                show_context_toggle,
                show_solution_toggle,
            ],
            outputs=[
                error_log,
                summary_title,
                summary,
                context_title,
                context_for_solution,
                solution_title,
                step_by_step,
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                nav_info,
                show_outputs_toggle,
                show_summary_toggle,
                show_context_toggle,
                show_solution_toggle,
                outputs_section,
                step_by_step_raw,
                db_need_more_context_display,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                labels_display,
            ],
        )

        save_feedback_btn.click(
            handle_save_feedback,
            inputs=[
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
            ],
            outputs=[feedback_status],
        )

        cluster_sample_toggle.change(
            handle_cluster_toggle,
            inputs=[cluster_sample_toggle],
            outputs=[
                error_log,
                summary,
                context_for_solution,
                step_by_step,
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                nav_info,
                step_by_step_raw,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                db_need_more_context_display,
                labels_display,
            ],
        )

        show_outputs_toggle.change(
            handle_outputs_toggle,
            inputs=[show_outputs_toggle],
            outputs=[outputs_section],
        )

        show_summary_toggle.change(
            handle_summary_toggle,
            inputs=[show_summary_toggle],
            outputs=[summary_title, summary],
        )

        show_context_toggle.change(
            handle_context_toggle,
            inputs=[show_context_toggle],
            outputs=[context_title, context_for_solution],
        )

        show_solution_toggle.change(
            handle_solution_toggle,
            inputs=[show_solution_toggle],
            outputs=[solution_title, step_by_step, copy_solution_btn],
        )

        copy_solution_btn.click(
            None,
            inputs=[step_by_step_raw],
            outputs=[],
            js="(text) => {navigator.clipboard.writeText(text); return text;}",
        )

        annotation_view_toggle.change(
            handle_annotation_view_toggle,
            inputs=[annotation_view_toggle],
            outputs=[
                feedback_text,
                golden_solution_text,
                expected_behavior_text,
                need_more_context_column,
            ],
        )

    return interface


demo = create_app()
if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7861)),
        share=False,
        debug=True,
    )
