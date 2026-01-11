# Ansible Log Annotation Interface

A custom Gradio-based data annotation interface for annotating Ansible error log pipeline outputs.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Run the interface:
```bash
python app.py
```

3. Access the interface at: http://localhost:7860

## Feedback Storage

Feedback is automatically saved to `data/annotation_feedback.json`

# Deployment of all apps

Please see [Local deployment](../../README.md#deploy-locally)