# FinLongDocQA v1.1 Update

FinLongDocQA has been updated to **v1.1**.

This update refines the gold evidence-page annotations in `dataset_qa.jsonl`. The dataset schema and example set remain unchanged: v1.1 contains the same **7,527 examples** as the original release.

The update mainly revises the `page_numbers` field. For examples where the corrected evidence affected the written rationale or computation, the corresponding `thoughts`, `python_code`, and `answer` fields were also updated.

We recommend using **FinLongDocQA v1.1** for future experiments and evidence-grounded evaluation.
