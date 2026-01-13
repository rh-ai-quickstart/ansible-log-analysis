#!/usr/bin/env python3
"""
Custom Data Annotation Interface for Ansible Log Error Annotations
"""

import gradio as gr
import json
import logging
import os
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import psycopg2

from test_end_to_end import run_evaluation

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - annotation interface - %(name)s - %(levelname)s - %(message)s",
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

        # Evaluation results storage: dict keyed by filename
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.eval_summary: Dict[str, Any] = {"total": 0, "success": 0, "run": False}

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
                    logger.info(f"Loading feedback from {self.feedback_file}")
                    self.feedback_data = json.load(f)

                # Restore evaluation results from saved feedback data
                self._restore_eval_results_from_feedback()
            else:
                self.feedback_data = []
                logger.info(f"No feedback file found at {self.feedback_file}")
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            self.feedback_data = []

    def _restore_eval_results_from_feedback(self) -> None:
        """Restore evaluation results from saved feedback data."""
        eval_count = 0
        success_count = 0

        for entry in self.feedback_data:
            if "eval_metrics" in entry:
                filename = entry.get("filename", "")
                self.evaluation_results[filename] = {
                    "success": entry.get("eval_success", False),
                    "metrics": entry.get("eval_metrics", []),
                }
                eval_count += 1
                if entry.get("eval_success", False):
                    success_count += 1

        if eval_count > 0:
            self.eval_summary = {
                "total": eval_count,
                "success": success_count,
                "run": True,
            }
            logger.info(f"Restored {eval_count} evaluation results from feedback data")

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

    def run_evaluation_on_feedback(
        self, filename_filter: str = None
    ) -> Tuple[str, str]:
        """Run deepeval evaluation on feedback data with golden solutions.

        Args:
            filename_filter: Optional filename to evaluate only a single entry.
                            If None, evaluates all entries with golden solutions.
        """
        if not self.feedback_data:
            return "No feedback data available", self.get_eval_summary_html()

        # Filter feedback entries that have golden solutions
        if filename_filter:
            # Single entry mode - find the specific entry
            entries_with_golden = [
                f
                for f in self.feedback_data
                if f.get("filename") == filename_filter
                and f.get("golden_stepByStepSolution", "").strip()
            ]
            if not entries_with_golden:
                return (
                    f"No golden solution found for entry: {filename_filter}",
                    self.get_eval_summary_html(),
                )
        else:
            # All entries mode
            entries_with_golden = [
                f
                for f in self.feedback_data
                if f.get("golden_stepByStepSolution", "").strip()
            ]
            if not entries_with_golden:
                return (
                    "No entries with golden solutions found",
                    self.get_eval_summary_html(),
                )

        # Convert to DataFrame with expected column names
        df_data = []
        for entry in entries_with_golden:
            df_data.append(
                {
                    "file_name": entry.get(
                        "filename", f"entry_{entry.get('index', 0)}"
                    ),
                    "logMessage": entry.get("logMessage", ""),
                    "stepByStepSolution": entry.get("stepByStepSolution", ""),
                    "golden_stepByStepSolution": entry.get(
                        "golden_stepByStepSolution", ""
                    ),
                }
            )

        df = pd.DataFrame(df_data)
        print(df.head())
        try:
            logger.info(f"Running evaluation on {len(df)} entries...")
            results_df = run_evaluation(df)

            # Process results and store in evaluation_results dict
            # For single entry mode, don't clear existing results
            if not filename_filter:
                self.evaluation_results = {}

            new_success_count = 0
            for _, row in results_df.iterrows():
                file_name = row.get("file_name") or row.get("name", "unknown")
                success = row.get("success", False)
                metrics_data = row.get("metrics_data", [])
                # Extract metrics info
                metrics_list = []
                if metrics_data:
                    for metric in metrics_data:
                        # Handle both dict and object access
                        if hasattr(metric, "name"):
                            metrics_list.append(
                                {
                                    "name": metric.name,
                                    "score": metric.score,
                                    "reason": metric.reason,
                                    "success": metric.success,
                                }
                            )
                        elif isinstance(metric, dict):
                            metrics_list.append(
                                {
                                    "name": metric.get("name", "Unknown"),
                                    "score": metric.get("score"),
                                    "reason": metric.get("reason"),
                                    "success": metric.get("success", False),
                                }
                            )

                self.evaluation_results[file_name] = {
                    "success": success,
                    "metrics": metrics_list,
                }

                if success:
                    new_success_count += 1

            # Update summary - recalculate from all stored results
            total_count = len(self.evaluation_results)
            total_success = sum(
                1 for r in self.evaluation_results.values() if r.get("success", False)
            )
            self.eval_summary = {
                "total": total_count,
                "success": total_success,
                "run": True,
            }

            # Safely update annotation.json with evaluation results
            self._update_annotation_with_eval_results()

            if filename_filter:
                status = "✅ Passed" if new_success_count > 0 else "❌ Failed"
                return f"Evaluation complete: {status}", self.get_eval_summary_html()
            else:
                return (
                    f"Evaluation complete: {total_success}/{total_count} tests passed",
                    self.get_eval_summary_html(),
                )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return f"Evaluation failed: {str(e)}", self.get_eval_summary_html()

    def _update_annotation_with_eval_results(self) -> None:
        """Safely update annotation.json with evaluation results, one entry at a time."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to save")
            return

        updated_count = 0

        # Update each feedback entry with its evaluation results
        for i, entry in enumerate(self.feedback_data):
            filename = entry.get("filename", "")

            # Find matching evaluation result
            eval_result = self.evaluation_results.get(filename)

            if eval_result:
                # Format metrics for storage
                metrics_formatted = []
                for metric in eval_result.get("metrics", []):
                    metrics_formatted.append(
                        {
                            "name": metric.get("name", "Unknown"),
                            "score": metric.get("score"),
                            "reason": metric.get("reason", ""),
                            "success": metric.get("success", False),
                        }
                    )

                # Update entry with evaluation results
                self.feedback_data[i]["eval_success"] = eval_result.get(
                    "success", False
                )
                self.feedback_data[i]["eval_metrics"] = metrics_formatted
                self.feedback_data[i]["eval_timestamp"] = datetime.now().isoformat()

                updated_count += 1
                logger.debug(f"Updated entry {i} ({filename}) with eval results")

        # Save updated feedback data to file
        try:
            # Write to a temp file first, then rename for safety
            temp_file = self.feedback_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.feedback_data, f, indent=2)

            # Rename temp file to actual file (atomic on most systems)
            os.replace(temp_file, self.feedback_file)

            logger.info(
                f"Updated annotation.json with {updated_count} evaluation results"
            )
        except Exception as e:
            logger.error(f"Failed to save evaluation results to annotation.json: {e}")
            # Try to clean up temp file if it exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def get_eval_summary_html(self) -> str:
        """Generate HTML for the global evaluation summary."""
        if not self.eval_summary.get("run", False):
            return """<div style='padding: 12px; text-align: center; color: #94a3b8; 
                       background-color: #1e293b; border: 1px solid #475569; border-radius: 8px;'>
                       No evaluation run yet. Click "Run Evaluation" to start.</div>"""

        total = self.eval_summary.get("total", 0)
        success = self.eval_summary.get("success", 0)
        percentage = (success / total * 100) if total > 0 else 0

        # Color based on success rate
        if percentage >= 80:
            color = "#22c55e"  # green
        elif percentage >= 50:
            color = "#eab308"  # yellow
        else:
            color = "#ef4444"  # red

        return f"""
        <div style='padding: 16px; background-color: #1e293b; border: 1px solid #475569; 
                    border-radius: 8px; display: flex; align-items: center; gap: 20px;'>
            <div style='font-size: 1.2em; font-weight: 600; color: {color};'>
                {success} / {total} tests passed ({percentage:.1f}%)
            </div>
            <div style='flex-grow: 1; background-color: #334155; border-radius: 4px; height: 8px;'>
                <div style='width: {percentage}%; background-color: {color}; height: 100%; 
                            border-radius: 4px; transition: width 0.3s;'></div>
            </div>
        </div>
        """

    def get_current_entry_eval_html(self) -> str:
        """Get evaluation results HTML for the current entry."""
        if not self.data:
            return """<div style='padding: 12px; text-align: center; color: #94a3b8; 
                       background-color: #1e293b; border: 1px solid #475569; border-radius: 8px;'>
                       No data available</div>"""

        current_entry = self.data[self.current_index]
        filename = current_entry.get("filename", "")

        # Try to find evaluation results - first from memory, then from saved feedback data
        eval_result = self.evaluation_results.get(filename)

        # If not in memory, check if it's saved in feedback_data
        if not eval_result:
            for feedback_entry in self.feedback_data:
                if (
                    feedback_entry.get("filename") == filename
                    and "eval_metrics" in feedback_entry
                ):
                    eval_result = {
                        "success": feedback_entry.get("eval_success", False),
                        "metrics": feedback_entry.get("eval_metrics", []),
                    }
                    break

        if not eval_result:
            return """<div style='padding: 12px; text-align: center; color: #94a3b8; 
                       background-color: #1e293b; border: 1px solid #475569; border-radius: 8px;'>
                       No evaluation results for this entry (no golden solution or not evaluated)</div>"""

        success = eval_result.get("success", False)
        metrics = eval_result.get("metrics", [])

        # Build metrics HTML
        metrics_html = ""
        for metric in metrics:
            name = metric.get("name", "Unknown")
            score = metric.get("score")
            reason = metric.get("reason", "No reason provided")
            metric_success = metric.get("success", False)

            score_display = f"{score:.2f}" if score is not None else "N/A"
            icon = "✅" if metric_success else "❌"
            score_color = "#22c55e" if metric_success else "#ef4444"

            metrics_html += f"""
            <div style='margin-bottom: 12px; padding: 12px; background-color: #0f172a; 
                        border-radius: 6px; border-left: 3px solid {score_color};'>
                <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 6px;'>
                    <span>{icon}</span>
                    <span style='font-weight: 600; color: #f1f5f9;'>{name}</span>
                    <span style='color: {score_color}; font-weight: 600;'>[GEval] ({score_display})</span>
                </div>
                <div style='color: #94a3b8; font-size: 0.9em; padding-left: 24px;'>
                    <strong>Reason:</strong> {reason[:500]}{"..." if len(reason) > 500 else ""}
                </div>
            </div>
            """

        overall_icon = "✅" if success else "❌"
        overall_color = "#22c55e" if success else "#ef4444"

        return f"""
        <div style='padding: 16px; background-color: #1e293b; border: 1px solid #475569; border-radius: 8px;'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 12px; 
                        padding-bottom: 12px; border-bottom: 1px solid #475569;'>
                <span style='font-size: 1.2em;'>{overall_icon}</span>
                <span style='font-weight: 600; color: {overall_color};'>
                    Overall: {"PASSED" if success else "FAILED"}
                </span>
            </div>
            {metrics_html if metrics_html else "<div style='color: #94a3b8;'>No metrics available</div>"}
        </div>
        """


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

        # Global Evaluation Summary Row

        # with gr.Row():
        #     with gr.Column():
        #         run_eval_btn = gr.Button(
        #             "🧪 Run Evaluation on labeled data",
        #             variant="secondary",
        #             scale=1,
        #         )
        #         eval_summary_display = gr.HTML(
        #             value=app.get_eval_summary_html(),
        #             label="Evaluation Summary",
        #         )
        #     eval_status = gr.Textbox(
        #         label="Evaluation Status",
        #         interactive=False,
        #         value="Not run yet",
        #         scale=2,
        #     )

        # Compact navigation controls - all in one row
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 Evaluation Results")
                with gr.Row():
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

                with gr.Row():
                    cluster_sample_toggle = gr.Checkbox(
                        label="One sample per cluster",
                        value=False,
                        interactive=True,
                        scale=2,
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
            with gr.Column():
                show_evals_toggle = gr.Checkbox(
                    label="Show Evals",
                    value=True,
                    interactive=True,
                )
                with gr.Row() as eval_interface:
                    with gr.Column():
                        run_eval_btn = gr.Button(
                            "🧪 Run Evaluation on labeled data",
                            variant="secondary",
                            scale=1,
                        )
                        eval_summary_display = gr.HTML(
                            value=app.get_eval_summary_html(),
                            label="Evaluation Summary",
                        )
                    eval_status = gr.Textbox(
                        label="Evaluation Status",
                        interactive=False,
                        value="Not run yet",
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
                show_evaluation_toggle = gr.Checkbox(
                    label="Show Evaluation Results",
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

                # Per-log evaluation results section
                with gr.Column(visible=True) as eval_section:
                    with gr.Row():
                        gr.Markdown(
                            "### 📊 Evaluation Results for This Entry", visible=True
                        )
                        run_single_eval_btn = gr.Button(
                            "🧪 Evaluate This Entry",
                            size="sm",
                            scale=0,
                            min_width=150,
                        )
                    eval_results_display = gr.HTML(
                        value=app.get_current_entry_eval_html(),
                        label="Evaluation Results",
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
            # Get evaluation results for current entry
            eval_results_html = app.get_current_entry_eval_html()
            # Return all values except db_need_more_context (index 9), plus the badge_html and eval_results
            # result[:9] = first 9 items, result[10:13] = items 10-12 (is_context_correct, need_more_context, reason), result[13] = labels
            return (
                result[:9] + result[10:13] + (badge_html, result[13], eval_results_html)
            )

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

        def _build_nav_response(nav_result):
            """Build UI response tuple from navigation result.

            Note: We only update content values here - toggle handlers control visibility exclusively.
            This avoids Gradio issues with combining value and visibility updates.
            """
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
            ) = nav_result

            badge_class = (
                "need-context-true" if db_need_more_context else "need-context-false"
            )
            badge_text = "Yes" if db_need_more_context else "No"
            badge_html = f"<div class='need-context-badge {badge_class}'>🤖 AI Assessment - Need More Context: {badge_text}</div>"
            eval_results_html = app.get_current_entry_eval_html()

            return (
                log_content,
                summary_content,
                context_content,
                step_content,
                feedback,
                golden,
                expected,
                nav,
                raw_step,
                badge_html,
                user_is_context_correct,
                user_need_more_context,
                user_need_more_context_reason,
                labels_json,
                eval_results_html,
            )

        def handle_navigate_prev():
            return _build_nav_response(app.navigate(-1))

        def handle_navigate_next():
            return _build_nav_response(app.navigate(1))

        def handle_jump(index):
            nav_result = (
                app.go_to_index(int(index) - 1)
                if index is not None
                else app.get_current_entry()
            )
            return _build_nav_response(nav_result)

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
            # Get evaluation results for current entry
            eval_results_html = app.get_current_entry_eval_html()
            # Return all values except db_need_more_context (index 9), plus the badge_html and eval_results
            # result[:9] = first 9 items, result[10:13] = items 10-12 (is_context_correct, need_more_context, reason), result[13] = labels
            return (
                result[:9] + result[10:13] + (badge_html, result[13], eval_results_html)
            )

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

        def handle_evaluation_toggle(show_evaluation):
            return gr.update(visible=show_evaluation)  # eval_section

        def handle_evals_toggle(show_evals):
            """Toggle visibility of both eval_interface, eval_section, and show_evaluation_toggle."""
            return (
                gr.update(visible=show_evals),  # eval_interface
                gr.update(visible=show_evals),  # eval_section
                gr.update(visible=show_evals),  # show_evaluation_toggle
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

        def handle_run_evaluation():
            """Run evaluation on feedback data and update displays."""
            status_msg, summary_html = app.run_evaluation_on_feedback()
            eval_results_html = app.get_current_entry_eval_html()
            return status_msg, summary_html, eval_results_html

        def handle_run_single_evaluation():
            """Run evaluation on just the current entry."""
            if not app.data:
                return (
                    "No data available",
                    app.get_eval_summary_html(),
                    app.get_current_entry_eval_html(),
                )

            current_entry = app.data[app.current_index]
            filename = current_entry.get("filename", "")

            status_msg, summary_html = app.run_evaluation_on_feedback(
                filename_filter=filename
            )
            eval_results_html = app.get_current_entry_eval_html()
            return status_msg, summary_html, eval_results_html

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
                eval_results_display,
            ],
        )

        prev_btn.click(
            handle_navigate_prev,
            inputs=[],
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
                db_need_more_context_display,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                labels_display,
                eval_results_display,
            ],
        )

        next_btn.click(
            handle_navigate_next,
            inputs=[],
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
                db_need_more_context_display,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                labels_display,
                eval_results_display,
            ],
        )

        jump_btn.click(
            handle_jump,
            inputs=[jump_input],
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
                db_need_more_context_display,
                is_context_correct_toggle,
                need_more_context_toggle,
                need_more_context_reason,
                labels_display,
                eval_results_display,
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
                eval_results_display,
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

        show_evaluation_toggle.change(
            handle_evaluation_toggle,
            inputs=[show_evaluation_toggle],
            outputs=[eval_section],
        )

        show_evals_toggle.change(
            handle_evals_toggle,
            inputs=[show_evals_toggle],
            outputs=[eval_interface, eval_section, show_evaluation_toggle],
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

        run_eval_btn.click(
            handle_run_evaluation,
            inputs=[],
            outputs=[eval_status, eval_summary_display, eval_results_display],
        )

        run_single_eval_btn.click(
            handle_run_single_evaluation,
            inputs=[],
            outputs=[eval_status, eval_summary_display, eval_results_display],
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
