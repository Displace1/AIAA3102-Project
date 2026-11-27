import json

from superagent.llm.client import LLMClientError


class BudgetedLLMClient:
    def __init__(
        self,
        llm_client,
        budget_manager,
    ):
        self.llm_client = llm_client
        self.budget_manager = budget_manager

    def chat(
        self,
        messages,
        purpose,
        temperature=0.2,
        max_tokens=500,
    ):
        estimated_input_tokens = (
            self._conservative_token_estimate(messages)
        )

        authorization = (
            self.budget_manager.authorize_call(
                estimated_input_tokens=(
                    estimated_input_tokens
                ),
                maximum_output_tokens=max_tokens,
                purpose=purpose,
            )
        )

        result = self.llm_client.chat(
            messages=messages,
            purpose=purpose,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = result.get("usage") or {}

        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get(
            "completion_tokens"
        )

        if input_tokens is None or output_tokens is None:
            raise LLMClientError(
                "The API response did not include token usage. "
                "The call cannot be audited or costed."
            )

        cost_record = (
            self.budget_manager.record_successful_call(
                call_id=result["call_id"],
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

        return {
            **result,
            "authorization": authorization,
            "cost": cost_record,
        }

    @staticmethod
    def _conservative_token_estimate(messages):
        serialized = json.dumps(
            messages,
            ensure_ascii=False,
        )

        # Conservative pre-call estimate:
        # treat each character as one potential token.
        return max(1, len(serialized))