"""
Utility functions for monitoring Kubernetes job status (for fail-fast coordination).
"""

import asyncio
import os
import shutil
import subprocess

try:
    import httpx
except ImportError:
    httpx = None


async def check_other_job_status(
    job_name: str, namespace: str, check_interval: int = 30
):
    """
    Periodically check if another job has failed.
    If the other job fails, exit immediately to fail this job too.

    Args:
        job_name: Name of the job to monitor
        namespace: Kubernetes namespace
        check_interval: How often to check (seconds)

    Returns:
        True if job is still running or completed successfully
        Raises SystemExit if job has failed
    """
    # Check if we're in a Kubernetes environment
    if not os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount"):
        # Not in Kubernetes, skip monitoring
        return True

    try:
        # Use oc or kubectl to check job status
        # Try oc first (OpenShift), then kubectl
        cmd = None
        if shutil.which("oc"):
            cmd = [
                "oc",
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                'jsonpath={.status.conditions[?(@.type=="Failed")].status}',
            ]
        elif shutil.which("kubectl"):
            cmd = [
                "kubectl",
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                'jsonpath={.status.conditions[?(@.type=="Failed")].status}',
            ]
        else:
            # No CLI available, skip monitoring
            return True

        if cmd:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                failed_status = result.stdout.strip()
                if failed_status == "True":
                    error_msg = (
                        f"Other job '{job_name}' has failed. This job must also fail."
                    )
                    print(f"\n✗ ERROR: {error_msg}")
                    raise SystemExit(1)
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        # If we can't check, continue (don't fail on monitoring errors)
        print(f"⚠ Warning: Could not check other job status: {e}")
        pass

    return True


async def wait_for_job_complete(
    job_name: str, namespace: str, max_wait_time: int = 600, check_interval: int = 10
):
    """
    Wait for a Kubernetes job to complete (successfully or with failure).
    This function BLOCKS until the job completes.

    Args:
        job_name: Name of the job to wait for
        namespace: Kubernetes namespace
        max_wait_time: Maximum time to wait in seconds (default: 10 minutes)
        check_interval: How often to check job status (seconds)

    Raises:
        TimeoutError: If job does not complete within max_wait_time
        SystemExit: If job fails
    """
    # Check if we're in a Kubernetes environment
    service_account_path = "/var/run/secrets/kubernetes.io/serviceaccount"
    if not os.path.exists(service_account_path):
        # Not in Kubernetes, skip waiting
        print(f"⚠ Not in Kubernetes environment, skipping wait for job '{job_name}'")
        return

    print(f"\nWaiting for job '{job_name}' to complete...")
    elapsed = 0

    # Try to use Kubernetes API via HTTP first (works without CLI tools)
    if httpx is not None:
        try:
            # Read service account token and namespace
            with open(f"{service_account_path}/token", "r") as f:
                token = f.read().strip()
            with open(f"{service_account_path}/namespace", "r") as f:
                pod_namespace = f.read().strip()

            # Use pod's namespace if namespace not provided
            if not namespace or namespace == "default":
                namespace = pod_namespace

            # Kubernetes API endpoint (in-cluster)
            api_host = os.getenv("KUBERNETES_SERVICE_HOST")
            api_port = os.getenv("KUBERNETES_SERVICE_PORT", "443")
            api_url = f"https://{api_host}:{api_port}/apis/batch/v1/namespaces/{namespace}/jobs/{job_name}"

            async with httpx.AsyncClient(
                verify=f"{service_account_path}/ca.crt", timeout=10.0
            ) as client:
                headers = {"Authorization": f"Bearer {token}"}

                while elapsed < max_wait_time:
                    try:
                        response = await client.get(api_url, headers=headers)
                        if response.status_code == 200:
                            job_data = response.json()
                            status = job_data.get("status", {})
                            conditions = status.get("conditions", [])

                            # Check for completion
                            for condition in conditions:
                                if (
                                    condition.get("type") == "Complete"
                                    and condition.get("status") == "True"
                                ):
                                    print(f"✓ Job '{job_name}' completed successfully!")
                                    return
                                if (
                                    condition.get("type") == "Failed"
                                    and condition.get("status") == "True"
                                ):
                                    error_msg = f"Job '{job_name}' has failed. This job must also fail."
                                    print(f"\n✗ ERROR: {error_msg}")
                                    raise SystemExit(1)

                        # Job is still running
                        if elapsed % 30 == 0:  # Print every 30 seconds
                            print(
                                f"  Job '{job_name}' still running... (elapsed: {elapsed}s/{max_wait_time}s)"
                            )

                    except httpx.RequestError as e:
                        if elapsed % 30 == 0:
                            print(
                                f"  Error checking job status: {e}, retrying... (elapsed: {elapsed}s/{max_wait_time}s)"
                            )

                    await asyncio.sleep(check_interval)
                    elapsed += check_interval

                # Timeout reached
                error_msg = (
                    f"Job '{job_name}' did not complete within {max_wait_time} seconds."
                )
                print(f"\n✗ ERROR: {error_msg}")
                raise TimeoutError(error_msg)

        except Exception as e:
            # Fall back to CLI tools if API access fails
            print(f"⚠ Kubernetes API access failed: {e}, trying CLI tools...")

    # Fallback: Use oc or kubectl CLI tools if available
    try:
        cmd_base = None
        if shutil.which("oc"):
            cmd_base = ["oc"]
        elif shutil.which("kubectl"):
            cmd_base = ["kubectl"]
        else:
            error_msg = f"No oc/kubectl available and Kubernetes API access failed. Cannot wait for job '{job_name}'."
            print(f"\n✗ ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        while elapsed < max_wait_time:
            # Check if job is complete
            cmd = cmd_base + [
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                'jsonpath={.status.conditions[?(@.type=="Complete")].status}',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                complete_status = result.stdout.strip()
                if complete_status == "True":
                    print(f"✓ Job '{job_name}' completed successfully!")
                    return

            # Check if job has failed
            cmd = cmd_base + [
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                'jsonpath={.status.conditions[?(@.type=="Failed")].status}',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                failed_status = result.stdout.strip()
                if failed_status == "True":
                    error_msg = f"Job '{job_name}' has failed. This job must also fail."
                    print(f"\n✗ ERROR: {error_msg}")
                    raise SystemExit(1)

            # Job is still running
            if elapsed % 30 == 0:  # Print every 30 seconds
                print(
                    f"  Job '{job_name}' still running... (elapsed: {elapsed}s/{max_wait_time}s)"
                )

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        # Timeout reached
        error_msg = f"Job '{job_name}' did not complete within {max_wait_time} seconds."
        print(f"\n✗ ERROR: {error_msg}")
        raise TimeoutError(error_msg)

    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.SubprocessError,
    ) as e:
        error_msg = f"Error checking job status: {e}"
        print(f"\n✗ ERROR: {error_msg}")
        raise RuntimeError(error_msg)


async def monitor_other_job_async(
    job_name: str, namespace: str, check_interval: int = 30
):
    """
    Background task to monitor another job's status.
    Runs in parallel with main work.

    Args:
        job_name: Name of the job to monitor
        namespace: Kubernetes namespace
        check_interval: How often to check (seconds)
    """
    while True:
        try:
            await check_other_job_status(job_name, namespace, check_interval)
            await asyncio.sleep(check_interval)
        except SystemExit:
            raise
        except Exception as e:
            # Don't fail on monitoring errors, just log and continue
            print(f"⚠ Warning: Error monitoring other job: {e}")
            await asyncio.sleep(check_interval)
