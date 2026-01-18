import os
import logging
import gradio as gr
import httpx
import markdown
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.name = "ui"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL")

# Log Categories (as defined in README)
EXPERT_CLASSES = [
    "Select All",
    "Cloud Infrastructure / AWS Engineers",
    "Kubernetes / OpenShift Cluster Admins",
    "DevOps / CI/CD Engineers (Ansible + Automation Platform)",
    "Networking / Security Engineers",
    "System Administrators / OS Engineers",
    "Application Developers / GitOps / Platform Engineers",
    "Identity & Access Management (IAM) Engineers",
    "Other / Miscellaneous",
]

# Global variable to store all alerts
all_alerts: List[Dict[str, Any]] = []


def extract_unique_label_keys(alerts: List[Dict[str, Any]]) -> List[str]:
    """Extract unique label keys from all alerts."""
    label_keys = set()
    for alert in alerts:
        labels = alert.get("log_labels", {})
        label_keys.update(labels.keys())
    return sorted(list(label_keys))


def extract_unique_label_values(
    alerts: List[Dict[str, Any]], label_key: str
) -> List[str]:
    """Extract unique values for a specific label key from all alerts."""
    label_values = set()
    for alert in alerts:
        labels = alert.get("log_labels", {})
        if label_key in labels:
            label_values.add(labels[label_key])
    return sorted(list(label_values))


def filter_alerts_by_label(
    alerts: List[Dict[str, Any]], label_key: str, label_value: str
) -> List[Dict[str, Any]]:
    """Filter alerts by a specific label key-value pair."""
    if not label_key or not label_value:
        return alerts

    filtered_alerts = []
    for alert in alerts:
        labels = alert.get("log_labels", {})
        if labels.get(label_key) == label_value:
            filtered_alerts.append(alert)

    return filtered_alerts


async def fetch_all_alerts() -> List[Dict[str, Any]]:
    """Fetch all Grafana alerts from the backend."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/grafana-alert/")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching all alerts: {e}")
        return []


async def fetch_alerts_by_expert_class(expert_class: str) -> List[Dict[str, Any]]:
    """Fetch alerts filtered by expert class from the backend."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URL}/grafana-alert/by-expert-class/?expert_class={expert_class}"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching alerts for expert class {expert_class}: {e}")
        return []


async def fetch_unique_clusters_by_expert_class(
    expert_class: str,
) -> List[Dict[str, Any]]:
    """Fetch unique log clusters for an expert class from the backend."""
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(
                f"Fetching unique clusters: {BACKEND_URL}/grafana-alert/unique-clusters/?expert_class={expert_class}"
            )
            response = await client.get(
                f"{BACKEND_URL}/grafana-alert/unique-clusters/?expert_class={expert_class}"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(
            f"Error fetching unique clusters for expert class {expert_class}: {e}"
        )
        return []


async def fetch_alerts_by_expert_class_and_cluster(
    expert_class: str, log_cluster: str
) -> List[Dict[str, Any]]:
    """Fetch alerts filtered by expert class and log cluster from the backend."""
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(
                f"Fetching alerts by class and cluster: {BACKEND_URL}/grafana-alert/by-expert-class-and-log-cluster/?expert_class={expert_class}&log_cluster={log_cluster}"
            )
            response = await client.get(
                f"{BACKEND_URL}/grafana-alert/by-expert-class-and-log-cluster/?expert_class={expert_class}&log_cluster={log_cluster}"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(
            f"Error fetching alerts for expert class {expert_class} and cluster {log_cluster}: {e}"
        )
        return []


def format_alerts_for_display(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format alerts data for display in Gradio."""
    if not alerts:
        return []

    formatted_data = []
    for i, alert in enumerate(alerts):
        # Parse timestamp for sorting
        timestamp = alert.get("logTimestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                sort_timestamp = dt
            except (ValueError, TypeError):
                formatted_timestamp = str(timestamp)
                sort_timestamp = datetime.min
        else:
            formatted_timestamp = "Unknown"
            sort_timestamp = datetime.min

        summary = alert.get("logSummary", "No summary available")
        expert_classification = alert.get("expertClassification", "Unclassified")
        log_cluster = alert.get("logCluster", "No cluster")

        formatted_data.append(
            {
                "Index": i,
                "Summary": summary,
                "Timestamp": formatted_timestamp,  # Keep for details view
                "Classification": expert_classification,  # Keep for details view
                "Log Cluster": log_cluster,  # Keep for details view
                "Sort_Timestamp": sort_timestamp,  # For sorting purposes
                "Full Alert": alert,  # Store full alert data for later use
            }
        )

    # Sort by timestamp (newest first)
    formatted_data.sort(key=lambda x: x["Sort_Timestamp"], reverse=True)

    # Reassign indices after sorting
    for i, item in enumerate(formatted_data):
        item["Index"] = i

    return formatted_data


def on_expert_change(expert: str):
    """Handle expert class dropdown change - now shows clusters first."""
    if not expert or expert == "Select an expert":
        empty_html = generate_logs_html([])
        return (
            empty_html,
            gr.update(choices=["No label key"], value="No label key"),
            gr.update(choices=["No label value"], value="No label value"),
        )

    import asyncio

    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Store the current expert class globally
        global \
            current_expert_class, \
            current_view_mode, \
            current_cluster_data, \
            current_category_alerts, \
            current_label_keys

        current_expert_class = expert
        current_view_mode = "clusters"  # Switch to cluster view mode

        # Handle "Select All" case - now shows clusters like other expert classes
        if expert == "Select All":
            alerts = loop.run_until_complete(fetch_all_alerts())
            current_category_alerts = alerts
            current_label_keys = extract_unique_label_keys(alerts)
            current_view_mode = "clusters"  # Use cluster view for "Select All"

            # Create unique cluster data from all alerts
            unique_clusters = {}
            for alert in alerts:
                cluster_key = alert.get("logCluster", "No cluster")
                if cluster_key not in unique_clusters:
                    unique_clusters[cluster_key] = alert

            # Format cluster data for display
            cluster_alerts = list(unique_clusters.values())
            current_cluster_data = format_alerts_for_display(cluster_alerts)

            # Generate HTML for clusters
            logs_html = generate_clusters_html(current_cluster_data, expert)
        else:
            # Fetch unique clusters for this expert class
            cluster_alerts = loop.run_until_complete(
                fetch_unique_clusters_by_expert_class(expert)
            )

            if cluster_alerts:
                # Format cluster data for display
                current_cluster_data = format_alerts_for_display(cluster_alerts)
                # Generate HTML for clusters
                logs_html = generate_clusters_html(current_cluster_data, expert)

                # Also fetch all alerts for label filtering
                all_alerts = loop.run_until_complete(
                    fetch_alerts_by_expert_class(expert)
                )
                current_category_alerts = all_alerts
                current_label_keys = extract_unique_label_keys(all_alerts)
            else:
                current_cluster_data = []
                current_category_alerts = []
                current_label_keys = []
                logs_html = generate_clusters_html([], expert)

        # Update label key dropdown
        label_key_choices = ["No label key"] + current_label_keys
        label_key_update = gr.update(choices=label_key_choices, value="No label key")
        label_value_update = gr.update(
            choices=["No label value"], value="No label value"
        )

        return logs_html, label_key_update, label_value_update
    finally:
        loop.close()


def on_label_key_change(label_key: str):
    """Handle label key dropdown change."""
    global current_category_alerts

    if not label_key or label_key == "No label key" or not current_category_alerts:
        return gr.update(choices=["No label value"], value="No label value")

    # Extract unique values for the selected label key
    label_values = extract_unique_label_values(current_category_alerts, label_key)
    label_value_choices = ["No label value"] + label_values

    return gr.update(choices=label_value_choices, value="No label value")


# Removed on_cluster_change function - clusters are now directly expandable


def on_label_filter_change(label_key: str, label_value: str):
    """Handle label filtering when label key or value changes."""
    global current_category_alerts, current_alerts_data, current_view_mode

    # Only apply label filtering when in logs view mode
    if current_view_mode != "logs" or not current_category_alerts:
        return generate_logs_html([])

    # Apply label filtering if both key and value are selected
    if (
        label_key
        and label_key != "No label key"
        and label_value
        and label_value != "No label value"
    ):
        filtered_alerts = filter_alerts_by_label(
            current_category_alerts, label_key, label_value
        )
    else:
        filtered_alerts = current_category_alerts

    # Format and update display
    formatted_data = format_alerts_for_display(filtered_alerts)
    current_alerts_data = formatted_data

    # Generate HTML for logs
    logs_html = generate_logs_html(formatted_data)

    return logs_html


def generate_clusters_html(
    cluster_data: List[Dict[str, Any]], expert_class: str
) -> str:
    """Generate HTML for displaying expandable log clusters with logs underneath."""
    if not cluster_data:
        return """
        <div style="text-align: center; padding: 3rem; color: #94a3b8;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
            <h3 style="color: #cbd5e1; margin-bottom: 0.5rem;">No clusters found</h3>
            <p style="margin: 0;">No log clusters available for this expert class</p>
        </div>
        """

    import asyncio

    # Set up async loop to fetch logs for each cluster
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    html_parts = []

    try:
        for i, cluster_data_item in enumerate(cluster_data):
            full_alert = cluster_data_item.get("Full Alert", cluster_data_item)
            summary = cluster_data_item.get(
                "Summary", full_alert.get("logSummary", "No summary available")
            )
            log_cluster = cluster_data_item.get(
                "Log Cluster", full_alert.get("logCluster", "No cluster")
            )

            # Fetch logs for this specific cluster
            try:
                # For "Select All", filter from already-fetched alerts instead of API call
                if expert_class == "Select All":
                    cluster_alerts = [
                        alert
                        for alert in current_category_alerts
                        if alert.get("logCluster") == log_cluster
                    ]
                else:
                    cluster_alerts = loop.run_until_complete(
                        fetch_alerts_by_expert_class_and_cluster(
                            expert_class, log_cluster
                        )
                    )
                cluster_logs_formatted = (
                    format_alerts_for_display(cluster_alerts) if cluster_alerts else []
                )
                logs_count = len(cluster_logs_formatted)
            except Exception as e:
                logger.error(f"Error fetching logs for cluster {log_cluster}: {e}")
                cluster_logs_formatted = []
                logs_count = 0

            # Generate HTML for the logs in this cluster
            cluster_logs_html = ""
            if cluster_logs_formatted:
                # Generate individual log items for this cluster
                for j, log_data in enumerate(cluster_logs_formatted):
                    log_full_alert = log_data.get("Full Alert", {})
                    log_summary = log_data.get("Summary", "No summary available")
                    log_timestamp = log_data.get("Timestamp", "Unknown")
                    log_classification = log_data.get("Classification", "Unclassified")

                    # Get classification color and badge
                    classification_color = (
                        "#10b981" if log_classification != "Unclassified" else "#f59e0b"
                    )
                    class_badge = "✅" if log_classification != "Unclassified" else "❓"

                    # Format labels
                    labels_html = ""
                    if log_full_alert.get("log_labels"):
                        labels_list = [
                            f'<span style="display: inline-block; background: rgba(30, 41, 59, 0.8); color: #e2e8f0; border: 1px solid #475569; padding: 0.25rem 0.5rem; border-radius: 0.375rem; margin: 0.125rem; font-size: 0.875rem;"><strong style="color: #cbd5e1;">{k}:</strong> {v}</span>'
                            for k, v in log_full_alert.get("log_labels", {}).items()
                        ]
                        labels_html = "".join(labels_list)
                    else:
                        labels_html = (
                            '<span style="color: #94a3b8;">No labels available</span>'
                        )

                    log_message = log_full_alert.get(
                        "logMessage", "No log message available"
                    )
                    step_by_step_solution = log_full_alert.get("stepByStepSolution", "")

                    # Convert markdown to HTML if solution exists
                    if step_by_step_solution and step_by_step_solution.strip():
                        step_by_step_solution_html = markdown.markdown(
                            step_by_step_solution.strip(),
                            extensions=["fenced_code", "tables", "nl2br"],
                        )
                    else:
                        step_by_step_solution_html = ""

                    # Create individual log item within cluster
                    log_item_html = f"""
                    <div class="cluster-log-item" style="margin: 0.5rem 0; border-left: 3px solid #3b82f6; background: rgba(15, 23, 42, 0.6);">
                        <!-- Hidden checkbox for log toggle -->
                        <input type="checkbox" id="cluster-{i}-log-{
                        j
                    }" style="display: none;">
                        
                        <!-- Log Summary (clickable) -->
                        <label for="cluster-{i}-log-{
                        j
                    }" class="cluster-log-summary" style="display: block; padding: 0.75rem 1rem; cursor: pointer; transition: all 0.2s ease; border-radius: 0 0.375rem 0.375rem 0;">
                            <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                                <div style="flex-shrink: 0; margin-top: 0.125rem;">
                                    <span style="font-size: 1.25rem;">{
                        class_badge
                    }</span>
                                </div>
                                <div style="flex: 1; min-width: 0;">
                                    <p style="margin: 0; font-size: 0.9rem; line-height: 1.4; color: #f1f5f9; font-weight: 500;">{
                        log_summary
                    }</p>
                                    <div style="display: flex; justify-content: between; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
                                        <span style="font-size: 0.8rem; color: #94a3b8;">⌚ {
                        log_timestamp
                    }</span>
                                        <span style="background: {
                        classification_color
                    }; color: white; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 500;">{
                        log_classification
                    }</span>
                                        <span class="cluster-log-toggle-text" style="color: #64748b; font-size: 0.75rem; margin-left: auto;">▼ Details</span>
                                    </div>
                                </div>
                            </div>
                        </label>
                        
                        <!-- Log Details (expandable) -->
                        <div class="cluster-log-details" style="background: rgba(7, 14, 25, 0.8); border-top: 1px solid #475569; padding: 0 1rem; max-height: 0; overflow: hidden; transition: max-height 0.3s ease, padding 0.3s ease;">
                            <div style="padding: 1rem 0;">
                                <div style="margin-bottom: 1rem;">
                                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                                        <span style="font-size: 1rem;">📄</span>
                                        <strong style="color: #f1f5f9; font-size: 0.95rem;">Full Log Message</strong>
                                    </div>
                                    <div style="background: rgba(15, 23, 42, 0.9); color: #e2e8f0; border: 1px solid #374151; border-radius: 0.375rem; padding: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.4; white-space: pre-wrap;">{
                        log_message
                    }</div>
                                </div>
                                
                                {
                        f'''
                                <div style="margin-bottom: 1rem;">
                                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                                        <span style="font-size: 1rem;">🔧</span>
                                        <strong style="color: #f1f5f9; font-size: 0.95rem;">Solution</strong>
                                    </div>
                                    <div class="markdown-content" style="background: rgba(16, 185, 129, 0.1); color: #e2e8f0; border: 1px solid #10b981; border-radius: 0.375rem; padding: 1rem; font-size: 0.8rem; line-height: 1.5;">
                                        {step_by_step_solution_html}
                                    </div>
                                </div>
                                '''
                        if step_by_step_solution_html
                        else ""
                    }
                                
                                <div>
                                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                                        <span style="font-size: 1rem;">🏷️</span>
                                        <strong style="color: #f1f5f9; font-size: 0.95rem;">Labels</strong>
                                    </div>
                                    <div style="line-height: 1.6; font-size: 0.8rem;">
                                        {labels_html}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                    cluster_logs_html += log_item_html

            else:
                cluster_logs_html = '<div style="padding: 1rem; text-align: center; color: #94a3b8; font-size: 0.875rem;">No logs found for this cluster</div>'

            # Create expandable cluster item
            cluster_item_html = f"""
            <div class="cluster-expandable-item" style="margin-bottom: 1.5rem;">
                <!-- Hidden checkbox for cluster toggle -->
                <input type="checkbox" id="cluster-toggle-{i}" style="display: none;">
                
                <!-- Cluster Summary (clickable) -->
                <label for="cluster-toggle-{i}" class="cluster-header" style="display: block; background: rgba(30, 41, 59, 0.8); border: 2px solid #475569; border-radius: 0.75rem; padding: 1.25rem; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.3); backdrop-filter: blur(10px);">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <!-- Cluster icon -->
                        <div style="flex-shrink: 0;">
                            <div style="background: #3b82f6; color: white; padding: 0.75rem; border-radius: 50%; font-size: 1.25rem; display: flex; align-items: center; justify-content: center;">
                                🎯
                            </div>
                        </div>
                        
                        <!-- Cluster info -->
                        <div style="flex: 1; min-width: 0;">
                            <h4 style="margin: 0 0 0.5rem 0; font-size: 1.125rem; font-weight: 600; color: #f1f5f9;">
                                {summary}
                            </h4>
                            <div style="margin-top: 0.75rem; display: flex; align-items: center; gap: 0.75rem;">
                                <span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 500; border: 1px solid #10b981;">
                                    📊 {logs_count} logs
                                </span>
                                <span class="cluster-toggle-text" style="color: #94a3b8; font-size: 0.875rem;">▼ Click to expand logs</span>
                            </div>
                        </div>
                        
                        <!-- Arrow indicator -->
                        <div style="flex-shrink: 0; color: #94a3b8; font-size: 1.5rem; transition: transform 0.2s ease;">
                            ▼
                        </div>
                    </div>
                </label>
                
                <!-- Cluster Logs (expandable content) -->
                <div class="cluster-logs-content" style="background: rgba(15, 23, 42, 0.9); border: 2px solid #475569; border-top: none; border-radius: 0 0 0.75rem 0.75rem; max-height: 0; overflow: hidden; transition: max-height 0.4s ease, padding 0.4s ease; backdrop-filter: blur(10px);">
                    <div style="padding: 1rem;">
                        {cluster_logs_html}
                    </div>
                </div>
            </div>
            """
            html_parts.append(cluster_item_html)

    finally:
        loop.close()

    # Add CSS for cluster and log toggle functionality
    cluster_css = """
    <style>
        /* Cluster toggle functionality */
        .cluster-expandable-item input[type="checkbox"] {
            display: none !important;
        }
        
        /* Default state: cluster content hidden */
        .cluster-logs-content {
            max-height: 0 !important;
            overflow: hidden !important;
            padding: 0 !important;
            border-width: 0 !important;
            transition: all 0.4s ease !important;
        }
        
        /* When cluster checkbox is checked: show content */
        input[id^="cluster-toggle-"]:checked ~ .cluster-logs-content {
            max-height: none !important;
            padding: 0 !important;
            border-width: 2px !important;
            border-color: #475569 !important;
            border-style: solid !important;
            border-top: none !important;
        }
        
        /* Rotate arrow when cluster expanded */
        input[id^="cluster-toggle-"]:checked ~ label > div > div:last-child {
            transform: rotate(180deg);
        }
        
        /* Change text when cluster expanded */
        input[id^="cluster-toggle-"]:checked ~ label .cluster-toggle-text::before {
            content: "▲ Click to collapse logs";
        }
        
        input[id^="cluster-toggle-"]:not(:checked) ~ label .cluster-toggle-text::before {
            content: "▼ Click to expand logs";
        }
        
        .cluster-toggle-text {
            display: none;
        }
        
        .cluster-toggle-text::before {
            display: inline;
            color: #94a3b8;
            font-size: 0.875rem;
        }
        
        /* Log item toggle functionality within clusters */
        .cluster-log-item input[type="checkbox"] {
            display: none !important;
        }
        
        /* Default state: log details hidden */
        .cluster-log-details {
            max-height: 0 !important;
            overflow: hidden !important;
            padding: 0 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        /* When log checkbox is checked: show details */
        input[id^="cluster-"][id*="-log-"]:checked ~ .cluster-log-details {
            max-height: none !important;
            padding: 0 1rem !important;
        }
        
        /* Change log toggle text */
        input[id^="cluster-"][id*="-log-"]:checked ~ label .cluster-log-toggle-text::before {
            content: "▲ Hide";
        }
        
        input[id^="cluster-"][id*="-log-"]:not(:checked) ~ label .cluster-log-toggle-text::before {
            content: "▼ Details";
        }
        
        .cluster-log-toggle-text {
            display: none;
        }
        
        .cluster-log-toggle-text::before {
            display: inline;
            color: #64748b;
            font-size: 0.75rem;
        }
        
        /* Enhanced hover effects */
        .cluster-header:hover {
            border-color: #3b82f6 !important;
            box-shadow: 0 4px 8px rgba(59,130,246,0.25) !important;
            transform: translateY(-1px);
            background: rgba(30, 41, 59, 0.95) !important;
        }
        
        .cluster-log-summary:hover {
            background: rgba(30, 41, 59, 0.4) !important;
        }
        
        /* Animations */
        .cluster-expandable-item {
            animation: fadeIn 0.3s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """

    return cluster_css + "\n".join(html_parts)


def generate_logs_html(alerts_data: List[Dict[str, Any]]) -> str:
    """Generate HTML for expandable log items."""
    if not alerts_data:
        return """
        <div style="text-align: center; padding: 3rem; color: #94a3b8;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
            <h3 style="color: #cbd5e1; margin-bottom: 0.5rem;">No logs found</h3>
            <p style="margin: 0;">Select a category to view log summaries</p>
        </div>
        """

    html_parts = []
    for i, alert_data in enumerate(alerts_data):
        full_alert = alert_data.get("Full Alert", {})
        summary = alert_data.get("Summary", "No summary available")
        timestamp = alert_data.get("Timestamp", "Unknown")
        expert_classification = alert_data.get("Classification", "Unclassified")

        # Get classification color and badge
        classification_color = (
            "#10b981" if expert_classification != "Unclassified" else "#f59e0b"
        )
        class_badge = "✅" if expert_classification != "Unclassified" else "❓"

        # Use full summary without truncation
        display_summary = summary

        # Format labels
        labels_html = ""
        if full_alert.get("log_labels"):
            labels_list = [
                f'<span style="display: inline-block; background: rgba(30, 41, 59, 0.8); color: #e2e8f0; border: 1px solid #475569; padding: 0.25rem 0.5rem; border-radius: 0.375rem; margin: 0.125rem; font-size: 0.875rem;"><strong style="color: #cbd5e1;">{k}:</strong> {v}</span>'
                for k, v in full_alert.get("log_labels", {}).items()
            ]
            labels_html = "".join(labels_list)
        else:
            labels_html = '<span style="color: #94a3b8;">No labels available</span>'

        log_message = full_alert.get("logMessage", "No log message available")
        step_by_step_solution = full_alert.get("stepByStepSolution", "")

        # Convert markdown to HTML if solution exists
        if step_by_step_solution and step_by_step_solution.strip():
            step_by_step_solution_html = markdown.markdown(
                step_by_step_solution.strip(),
                extensions=["fenced_code", "tables", "nl2br"],
            )
        else:
            step_by_step_solution_html = ""

        # Create the expandable log item using CSS-only toggle
        log_item_html = f"""
        <div class="log-item" style="margin-bottom: 1rem;">
            <!-- Hidden checkbox for toggle functionality -->
            <input type="checkbox" id="toggle-{i}" style="display: none;">
            
            <!-- Log Summary (clickable label) -->
            <label for="toggle-{
            i
        }" class="log-summary" style="display: block; background: rgba(30, 41, 59, 0.8); border: 2px solid #475569; border-radius: 0.75rem; padding: 1.25rem; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.3); backdrop-filter: blur(10px);">
                <div style="display: flex; align-items: flex-start; gap: 1rem;">
                    <!-- Status indicator -->
                    <div style="flex-shrink: 0; margin-top: 0.25rem;">
                        <span style="font-size: 1.5rem;">{class_badge}</span>
                    </div>
                    
                    <!-- Main content -->
                    <div style="flex: 1; min-width: 0;">
                        <div style="display: flex; justify-content: between; align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem;">
                            <div style="flex: 1;">
                                <p style="margin: 0; font-size: 1rem; line-height: 1.5; color: #f1f5f9; font-weight: 500;">{
            display_summary
        }</p>
                            </div>
                            <div style="flex-shrink: 0; text-align: right;">
                                <div style="font-size: 0.875rem; color: #cbd5e1; margin-bottom: 0.25rem;">⏰ {
            timestamp
        }</div>
                                <span style="background: {
            classification_color
        }; color: white; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.75rem; font-weight: 500;">{
            expert_classification
        }</span>
                </div>
            </div>
            
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span style="background: #10b981; color: white; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.75rem; font-weight: 500;">📋 Event Summary</span>
                            <span class="toggle-text" style="color: #94a3b8; font-size: 0.875rem;">▼ Click to expand details</span>
                        </div>
                    </div>
                </div>
            </label>
            
            <!-- Log Details (shown when checkbox is checked) -->
            <div class="log-details-content" style="background: rgba(15, 23, 42, 0.9); border: 2px solid #475569; border-top: none; border-radius: 0 0 0.75rem 0.75rem; padding: 1.5rem; max-height: 0; overflow: hidden; transition: max-height 0.3s ease, padding 0.3s ease; backdrop-filter: blur(10px);">
                <div style="border-bottom: 2px solid #475569; padding-bottom: 1rem; margin-bottom: 1.5rem;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.25rem;">⏰</span>
                                <strong style="color: #f1f5f9;">Timestamp</strong>
                            </div>
                            <code style="background: rgba(30, 41, 59, 0.8); color: #e2e8f0; padding: 0.5rem; border-radius: 0.375rem; font-size: 0.875rem; display: block; border: 1px solid #475569;">{
            timestamp
        }</code>
                        </div>
                        <div>
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.25rem;">📋</span>
                                <strong style="color: #f1f5f9;">Event Summary</strong>
                            </div>
                            <span style="background: #10b981; color: white; padding: 0.5rem 0.75rem; border-radius: 0.5rem; font-size: 0.875rem; font-weight: 500; display: inline-block;">Displayed Above</span>
                        </div>
                    </div>
                </div>
                
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                        <span style="font-size: 1.25rem;">📄</span>
                        <strong style="color: #f1f5f9; font-size: 1.1rem;">Full Log Message</strong>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.8); color: #e2e8f0; border: 1px solid #475569; border-radius: 0.5rem; padding: 1rem; font-family: 'JetBrains Mono', 'Monaco', 'Menlo', monospace; font-size: 0.875rem; line-height: 1.5; white-space: pre-wrap;">{
            log_message
        }</div>
                </div>
                
                {
            f'''
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                        <span style="font-size: 1.25rem;">🔧</span>
                        <strong style="color: #f1f5f9; font-size: 1.1rem;">Step-by-Step Solution</strong>
                    </div>
                    <div class="markdown-content" style="background: rgba(16, 185, 129, 0.1); color: #e2e8f0; border: 2px solid #10b981; border-radius: 0.5rem; padding: 1.5rem; font-size: 0.875rem; line-height: 1.6; position: relative;">
                        <div style="position: absolute; top: 0.5rem; right: 0.5rem; background: #10b981; color: white; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 600;">SOLUTION</div>
                        {step_by_step_solution_html}
                    </div>
                </div>
                '''
            if step_by_step_solution_html
            else ""
        }
                
                <div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                        <span style="font-size: 1.25rem;">🏷️</span>
                        <strong style="color: #f1f5f9; font-size: 1.1rem;">Labels</strong>
                    </div>
                    <div style="line-height: 1.8;">
                        {labels_html}
                    </div>
                </div>
            </div>
    </div>
        </div>
        """

        html_parts.append(log_item_html)

    # Add CSS for toggle functionality
    toggle_css = """
    <style>
        /* CSS-only toggle functionality */
        input[type="checkbox"] {
            display: none !important;
        }
        
        /* Default state: details hidden */
        .log-details-content {
            max-height: 0 !important;
            overflow: hidden !important;
            padding: 0 1.5rem !important;
            border-width: 0 !important;
            transition: all 0.4s ease !important;
        }
        
        /* When checkbox is checked: show details */
        input[type="checkbox"]:checked ~ .log-details-content {
            max-height: none !important;
            padding: 1.5rem !important;
            border-width: 2px !important;
            border-color: #475569 !important;
            border-style: solid !important;
            border-top: none !important;
        }
        
        /* Change arrow when expanded */
        input[type="checkbox"]:checked ~ label .toggle-text::before {
            content: "▲ Click to collapse details";
        }
        
        input[type="checkbox"]:not(:checked) ~ label .toggle-text::before {
            content: "▼ Click to expand details";
        }
        
        .toggle-text {
            display: none;
        }
        
        .toggle-text::before {
            display: inline;
            color: #94a3b8;
            font-size: 0.875rem;
        }
        
        /* Enhanced hover effects for clickable labels */
        .log-summary:hover {
            border-color: #3b82f6 !important;
            box-shadow: 0 4px 8px rgba(59,130,246,0.25) !important;
            transform: translateY(-1px);
            background: rgba(30, 41, 59, 0.95) !important;
        }
        
        /* Smooth transitions */
        .log-summary {
            transition: all 0.2s ease !important;
        }
    </style>
    """

    return toggle_css + "\n".join(html_parts)


# Global variables to store current alerts data and filtering state
current_alerts_data = []
current_category_alerts = []  # Store alerts from current category
current_label_keys = []  # Store available label keys
current_expert_class = ""  # Store current expert class
current_view_mode = "clusters"  # "clusters" or "logs"
current_cluster_data = []  # Store unique cluster data for current expert
current_selected_cluster = ""  # Store currently selected cluster


def create_interface():
    """Create and configure the Gradio interface."""

    # Custom CSS for modern, beautiful dark theme
    custom_css = """
    /* CSS Custom Property for consistent max-width */
    :root {
        --app-max-width: 1080px;
    }
    
    /* Main container styling */
    .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #e2e8f0 !important;
        overflow: visible !important;
        max-width: var(--app-max-width) !important;
        margin: 0 auto !important;
        padding: 0 1rem !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Ensure parent elements allow dropdown overflow */
    .gradio-container > * {
        overflow: visible !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    .gradio-container .block {
        overflow: visible !important;
        max-width: 100% !important;
    }
    
    .gradio-container .wrap {
        overflow: visible !important;
        max-width: 100% !important;
    }
    
    /* Ensure all Gradio rows and columns respect the container width */
    .gradio-container .gradio-row,
    .gradio-container .gradio-column {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin: 1rem 0;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    .card {
        max-width: 100% !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }
    
    /* Input styling */
    .gradio-dropdown {
        border-radius: 0.75rem !important;
        border: 2px solid #475569 !important;
        background: #334155 !important;
        color: #e2e8f0 !important;
        transition: all 0.2s ease !important;
        position: relative !important;
    }
    
    .gradio-dropdown:hover,
    .gradio-dropdown:focus,
    .gradio-dropdown:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Dropdown menu styling */
    .gradio-dropdown .wrap {
        position: relative !important;
    }
    
    .gradio-dropdown .dropdown {
        position: absolute !important;
        background: #334155 !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .gradio-dropdown .dropdown .item {
        color: #e2e8f0 !important;
        background: transparent !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .gradio-dropdown .dropdown .item:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #f1f5f9 !important;
    }
    
    /* Specific targeting for dropdown containers */
    .gradio-container .block.gradio-dropdown {
        position: relative !important;
    }
    
    .gradio-container .dropdown-content {
        position: absolute !important;
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Additional Gradio dropdown fixes */
    .gradio-container [data-testid="dropdown"] {
        position: relative !important;
    }
    
    .gradio-container [data-testid="dropdown"] .dropdown-menu {
        position: absolute !important;
        background: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    .gradio-container [data-testid="dropdown"] .dropdown-option {
        color: #e2e8f0 !important;
        background: transparent !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
        border-bottom: 1px solid rgba(71, 85, 105, 0.3) !important;
    }
    
    .gradio-container [data-testid="dropdown"] .dropdown-option:last-child {
        border-bottom: none !important;
    }
    
    .gradio-container [data-testid="dropdown"] .dropdown-option:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #f1f5f9 !important;
    }
    

    
    /* Log items styling */
    .logs-container {
        padding: 0.5rem;
        position: relative;
        max-width: 100% !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }
    
    .log-item {
        margin-bottom: 1rem;
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .log-summary {
        transition: all 0.2s ease;
        position: relative;
    }
    
    .log-summary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.25);
    }
    
    .log-details {
        border-top: 2px solid #475569;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        position: relative;
    }
    
    /* Button styling */
    button {
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    /* Markdown content styling */
    .markdown {
        line-height: 1.6;
    }
    
    /* Enhanced markdown styling for solutions */
    .log-details-content h1,
    .log-details-content h2,
    .log-details-content h3,
    .log-details-content h4,
    .log-details-content h5,
    .log-details-content h6 {
        color: #f1f5f9 !important;
        margin: 1rem 0 0.5rem 0 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #475569 !important;
        padding-bottom: 0.25rem !important;
    }
    
    .log-details-content h1 { font-size: 1.5rem !important; }
    .log-details-content h2 { font-size: 1.25rem !important; }
    .log-details-content h3 { font-size: 1.125rem !important; }
    .log-details-content h4 { font-size: 1rem !important; }
    
    .log-details-content p {
        margin: 0.75rem 0 !important;
        color: #e2e8f0 !important;
    }
    
    .log-details-content ul,
    .log-details-content ol {
        margin: 0.75rem 0 !important;
        padding-left: 1.5rem !important;
    }
    
    .log-details-content li {
        margin: 0.25rem 0 !important;
        color: #e2e8f0 !important;
    }
    
    .log-details-content code {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #fbbf24 !important;
        padding: 0.125rem 0.375rem !important;
        border-radius: 0.25rem !important;
        font-size: 0.875rem !important;
        border: 1px solid #475569 !important;
    }
    
    .log-details-content pre {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        overflow-x: auto !important;
    }
    
    .log-details-content pre code {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: #e2e8f0 !important;
    }
    
    .log-details-content blockquote {
        border-left: 4px solid #3b82f6 !important;
        background: rgba(59, 130, 246, 0.1) !important;
        padding: 0.75rem 1rem !important;
        margin: 1rem 0 !important;
        border-radius: 0.375rem !important;
        color: #e2e8f0 !important;
    }
    
    .log-details-content table {
        border-collapse: collapse !important;
        margin: 1rem 0 !important;
        width: 100% !important;
    }
    
    .log-details-content th,
    .log-details-content td {
        border: 1px solid #475569 !important;
        padding: 0.5rem !important;
        text-align: left !important;
    }
    
    .log-details-content th {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    .log-details-content td {
        color: #e2e8f0 !important;
    }
    
    .log-details-content strong {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    .log-details-content em {
        color: #cbd5e1 !important;
        font-style: italic !important;
    }
    
    /* Markdown content styling for solutions */
    .markdown-content h1,
    .markdown-content h2,
    .markdown-content h3,
    .markdown-content h4,
    .markdown-content h5,
    .markdown-content h6 {
        color: #f1f5f9 !important;
        margin: 1rem 0 0.5rem 0 !important;
        font-weight: 600 !important;
    }
    
    .markdown-content p {
        margin: 0.5rem 0 !important;
    }
    
    .markdown-content ul,
    .markdown-content ol {
        margin: 0.5rem 0 !important;
        padding-left: 1.5rem !important;
    }
    
    .markdown-content li {
        margin: 0.25rem 0 !important;
    }
    
    .markdown-content code {
        background: rgba(15, 23, 42, 0.9) !important;
        color: #fbbf24 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 0.25rem !important;
        font-family: 'JetBrains Mono', 'Monaco', 'Menlo', monospace !important;
        font-size: 0.85em !important;
        border: 1px solid #475569 !important;
    }
    
    .markdown-content pre {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        margin: 0.75rem 0 !important;
        overflow-x: auto !important;
    }
    
    .markdown-content pre code {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: #e2e8f0 !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
        display: block !important;
        white-space: pre !important;
    }
    
    .markdown-content strong {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    .markdown-content blockquote {
        border-left: 3px solid #10b981 !important;
        background: rgba(16, 185, 129, 0.05) !important;
        padding: 0.5rem 1rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 0 0.25rem 0.25rem 0 !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1.5rem;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 0.75rem;
        border: 1px solid #475569;
        backdrop-filter: blur(10px);
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #10b981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    """

    # JavaScript to auto-redirect to dark theme if not already set
    head_js = """
    <script>
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    </script>
    """

    with gr.Blocks(
        title="Ansible Logs Viewer",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ).set(
            body_background_fill="*neutral_950",
            body_text_color="*neutral_200",
            block_background_fill="*neutral_900",
            block_border_color="*neutral_700",
            input_background_fill="*neutral_800",
            button_primary_background_fill="*primary_600",
            button_primary_text_color="white",
        ),
        css=custom_css,
        head=head_js,
    ) as demo:
        # Beautiful header
        with gr.Column(elem_classes=["main-header"]):
            gr.HTML("""
            <div class="animate-fade-in">
                <h1>🚀 Ansible Logs Viewer</h1>
                <p>Advanced log analysis and monitoring dashboard</p>
            </div>
            """)

        # Filters section
        with gr.Column(elem_classes=["card"]):
            gr.HTML('<h3 class="section-header">🎯 Filters & Controls</h3>')

            with gr.Row():
                expert_dropdown = gr.Dropdown(
                    choices=["Select an expert"] + EXPERT_CLASSES,
                    value="Select an expert",
                    label="📂 Expert Class",
                    info="Choose an expert class to filter and analyze alerts",
                    elem_classes=["expert-selector"],
                    scale=3,
                )

                label_key_dropdown = gr.Dropdown(
                    choices=["No label key"],
                    value="No label key",
                    label="🏷️ Label Key",
                    info="Filter by specific label keys",
                    elem_classes=["filter-dropdown"],
                    scale=1,
                )

                label_value_dropdown = gr.Dropdown(
                    choices=["No label value"],
                    value="No label value",
                    label="🎯 Label Value",
                    info="Filter by label values",
                    elem_classes=["filter-dropdown"],
                    scale=1,
                )

        # Main content area
        with gr.Column():
            gr.HTML('<h3 class="section-header">📊 Interactive Log Viewer</h3>')
            logs_display = gr.HTML(
                value="""
                <div style="text-align: center; padding: 3rem; color: #94a3b8;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
                    <h3 style="color: #cbd5e1; margin-bottom: 0.5rem;">No logs found</h3>
                    <p style="margin: 0;">Select a category to view log summaries with expandable details</p>
                        </div>
                        """,
                elem_classes=["logs-container"],
            )

        # Event handlers
        expert_dropdown.change(
            fn=on_expert_change,
            inputs=[expert_dropdown],
            outputs=[logs_display, label_key_dropdown, label_value_dropdown],
        )

        label_key_dropdown.change(
            fn=on_label_key_change,
            inputs=[label_key_dropdown],
            outputs=[label_value_dropdown],
        )

        label_value_dropdown.change(
            fn=on_label_filter_change,
            inputs=[label_key_dropdown, label_value_dropdown],
            outputs=[logs_display],
        )

        # Footer
        with gr.Column(elem_classes=["footer"]):
            gr.HTML(f"""
            <div style="text-align: center;">
                <div class="status-indicator status-online"></div>
                <strong>Backend Connected:</strong> <code>{BACKEND_URL}</code>
                <br><br>
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                    🔒 Secure • 🚀 Fast • 📊 Real-time Analytics
                </p>
            </div>
            """)

    return demo


demo = create_interface()


def main():
    """Main function to launch the Gradio app."""
    logger.info("🚀 Starting Ansible Logs Viewer...")
    logger.info(f"Backend URL: {BACKEND_URL}")

    # Create and launch the interface

    # Launch the app
    demo.launch(
        server_name=os.getenv(
            "GRADIO_SERVER_NAME", "0.0.0.0"
        ),  # Allow external connections
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),  # Default Gradio port
        share=False,  # Set to True for public sharing
        debug=True,  # Enable debug mode
    )


if __name__ == "__main__":
    main()
