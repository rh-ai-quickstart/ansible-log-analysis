from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.evaluate import evaluate
import pandas as pd
from deepeval.evaluate.configs import DisplayConfig, ErrorConfig

import os
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase
from typing import List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - annotation interface - %(name)s - %(levelname)s - %(message)s",
)

llm_vllm = LocalModel(
    model=os.environ.get("OPENAI_MODEL"),
    base_url=os.environ.get("OPENAI_API_ENDPOINT"),
    api_key=os.environ.get("OPENAI_API_TOKEN"),
    temperature=float(os.environ.get("OPENAI_TEMPERATURE")),
)


# Metric 1: Root Cause Accuracy - Does the actual output identify the same underlying problem?
root_cause_metric = GEval(
    name="Root Cause Accuracy",
    criteria="Evaluate whether the actual output correctly identifies the same root cause/underlying problem as the expected output",
    evaluation_steps=[
        "1. Extract the root cause from both outputs",
        "2. Compare the root cause: Is the fundamental reason the error occurred according to each?",
        "3. Score: 1 = completely different root causes identified, 6 = partially overlapping causes (related but not exact), 8 = similar root cause identified, 10 = identical root cause identified",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=llm_vllm,
)

# Metric 2: Solution Steps Alignment - Do the remediation steps match?
same_steps_metric = GEval(
    name="Solution Steps Alignment",
    criteria="Evaluate whether the remediation/fix steps in the actual output align with those in the expected output",
    evaluation_steps=[
        "1. Identify the key remediation actions in both outputs: What specific steps/commands/changes are recommended to fix the issue?",
        "2. Compare the core approach: Do both suggest the same type of fix (e.g., both suggest modifying a config file vs one suggests restart and other suggests reinstall)?",
        "3. Check for command/action equivalence: Are the same commands, file paths, or configuration changes mentioned in both?",
        "4. Evaluate step ordering: Do the steps follow a similar logical sequence (diagnosis → fix → verification)?",
        "5. Allow for reasonable variations: Extra helpful steps or minor ordering differences should not penalize heavily",
        "6. Score: 1 = completely different remediation approaches, 6 = partially similar steps with key differences, 8 = similar steps identified, 10 = same remediation steps/approach",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=llm_vllm,
)

# Metric 2: Answer Quality - Overall quality of the solution
answer_quality_metric = GEval(
    name="Answer Quality",
    criteria="Evaluate the overall quality, completeness, and usefulness of the troubleshooting solution",
    evaluation_steps=[
        "1. Check if the solution provides a clear root cause analysis that explains WHY the error occurred",
        "2. Verify that the solution includes concrete, actionable steps (not vague suggestions) with actual commands/code examples",
        "3. Assess if the solution has proper structure: root cause → fix steps → verification → prevention",
        "4. Check if verification steps are included to confirm the fix worked",
        "5. Evaluate if preventive measures are suggested to avoid future occurrences",
        "6. Determine if the explanation is clear and would help an engineer understand and execute the fix",
        "7. Assign a score: 0 (poor quality, vague, incomplete) to 1 (excellent quality, comprehensive, actionable)",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=llm_vllm,
)

# Metric 3: OpenShift Specificity - Uses OpenShift-specific tools and approaches
openshift_specificity_metric = GEval(
    name="OpenShift Specificity",
    criteria="Determine if the solution correctly uses OpenShift-specific tools, commands, and approaches instead of generic Kubernetes solutions",
    evaluation_steps=[
        "1. Check if the solution uses 'oc' commands instead of 'kubectl' commands throughout",
        "2. Verify if OpenShift-specific features are referenced when applicable: Routes (not Ingress), DeploymentConfigs, Security Context Constraints (SCC), MachineConfig resources",
        "3. Check if the solution uses 'oc debug node/' for node access and troubleshooting instead of direct SSH when appropriate",
        "4. Ensure the solution accounts for OpenShift's stricter security policies compared to vanilla Kubernetes",
        "5. Assign a score: 0 (uses kubectl or generic K8s, ignores OpenShift specifics) to 1 (fully OpenShift-native approach with proper oc commands and OpenShift features)",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=llm_vllm,
)


def create_test_cases(df: pd.DataFrame) -> List[LLMTestCase]:
    test_cases = []
    for _, row in df.iterrows():
        test_cases.append(
            LLMTestCase(
                input=row["logMessage"],
                actual_output=row["stepByStepSolution"],
                expected_output=row["golden_stepByStepSolution"],
                name=row["file_name"],
            )
        )
    return test_cases


def run_evaluation(df: pd.DataFrame):
    test_cases = create_test_cases(df)
    results = evaluate(
        test_cases=test_cases,
        metrics=[root_cause_metric, same_steps_metric],
        display_config=DisplayConfig(print_results=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    test_results_df = pd.DataFrame(results.test_results)
    return pd.merge(
        df, test_results_df, left_on="file_name", right_on="name", how="left"
    )
