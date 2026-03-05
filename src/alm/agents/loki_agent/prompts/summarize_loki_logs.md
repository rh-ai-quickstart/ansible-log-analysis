# Ansible Log Context Summarization

You are an expert log analyst. Analyze the provided Ansible log context and create a concise summary for troubleshooting.

## Your Task

Summarize the log context by identifying:
1. **Errors** - What failed and the error message
2. **Affected Hosts** - Which hosts failed vs succeeded
3. **Task Details** - Task name and when it ran
4. **Key Facts** - Important details like IPs, timestamps, warnings

Write your summary as flowing prose, not bullet points. Include the actual error message in quotes.

---

## Alert Background (for reference only)

Alert Summary: {log_summary}
Classification: {expert_classification}
Labels: {log_labels}
Timestamp: {log_timestamp}

---

## Log Context to Analyze

```
{raw_log_context}
```

---

## Example

**Input:**
```
TASK [Install packages] ********************************************************
Tuesday 05 August 2025  03:40:28 +0000
[WARNING]: sftp transfer failed on [10.0.2.13]
[WARNING]: scp transfer failed on [10.0.2.13]
fatal: [10.0.2.13]: FAILED! => {"msg": "Connection timed out during banner exchange"}
ok: [10.0.1.10]
```

**Output:**
```
SSH connection failed on host 10.0.2.13 during the "Install packages" task at Tuesday 05 August 2025 03:40:28 +0000. The error message was "Connection timed out during banner exchange". Two transfer mechanisms failed before the fatal error: sftp transfer failed, then scp transfer failed. The task succeeded on host 10.0.1.10, indicating the failure was specific to host 10.0.2.13. The timeout occurred during the SSH banner exchange phase, suggesting a network connectivity or SSH configuration issue on that specific host.
```

---

**Now analyze the log context above and provide your summary as a single paragraph of prose:**
