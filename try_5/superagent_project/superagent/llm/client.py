import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        endpoint,
        model,
        log_path,
        api_key_environment_variable="AIGC_API_KEY",
        timeout_seconds=120,
        max_retries=3,
        retry_base_seconds=1,
        retry_max_seconds=8,
    ):
        self.endpoint = endpoint
        self.model = model
        self.log_path = Path(log_path)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds

        self.api_key = os.getenv(
            api_key_environment_variable
        )

        if not self.api_key:
            raise LLMClientError(
                "API key environment variable is missing: "
                f"{api_key_environment_variable}"
            )

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.session = requests.Session()

    def chat(
        self,
        messages,
        purpose,
        temperature=0.2,
        max_tokens=500,
    ):
        call_id = str(uuid.uuid4())
        started_at = datetime.now().astimezone()
        start_time = time.perf_counter()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        attempt_errors = []

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(10, self.timeout_seconds),
                )

                if (
                    response.status_code
                    in RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    attempt_errors.append(
                        f"HTTP {response.status_code}"
                    )
                    self._sleep_before_retry(attempt)
                    continue

                response.raise_for_status()
                result = response.json()
                break

            except requests.HTTPError as error:
                last_error = error
                attempt_errors.append(str(error))

                status_code = getattr(error.response, "status_code", None)
                retryable = status_code in RETRYABLE_STATUS_CODES
                if retryable and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                self._log_failure(
                    call_id, started_at, start_time, purpose, payload,
                    attempt, error, attempt_errors
                )
                raise LLMClientError(
                    f"LLM API request failed: {error}"
                ) from error

            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
                attempt_errors.append(str(error))

                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                self._log_failure(
                    call_id, started_at, start_time, purpose, payload,
                    attempt, error, attempt_errors
                )
                raise LLMClientError(
                    f"LLM API request failed: {error}"
                ) from error

            except (requests.RequestException, ValueError) as error:
                last_error = error
                attempt_errors.append(str(error))
                self._log_failure(
                    call_id, started_at, start_time, purpose, payload,
                    attempt, error, attempt_errors
                )
                raise LLMClientError(
                    f"LLM API response failed: {error}"
                ) from error
        else:
            raise LLMClientError(
                f"LLM API request failed: {last_error}"
            )

        try:
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as error:
            raise LLMClientError(
                "API response did not contain a valid message."
            ) from error

        usage = result.get("usage") or {}

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        log_record = {
            "call_id": call_id,
            "timestamp": started_at.isoformat(),
            "purpose": purpose,
            "model": self.model,
            "endpoint": self.endpoint,
            "status": "success",
            "attempts": attempt,
            "latency_seconds": (
                time.perf_counter() - start_time
            ),
            "request": payload,
            "response_id": result.get("id"),
            "response_content": content,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
            "raw_usage": usage,
        }

        self._write_log(log_record)

        return {
            "call_id": call_id,
            "content": content,
            "finish_reason": finish_reason,
            "usage": log_record["usage"],
            "raw_response": result,
        }

    def _sleep_before_retry(self, attempt):
        delay = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** (attempt - 1)),
        )
        time.sleep(delay)

    def _log_failure(
        self,
        call_id,
        started_at,
        start_time,
        purpose,
        payload,
        attempt,
        error,
        attempt_errors,
    ):
        self._write_log(
            {
                "call_id": call_id,
                "timestamp": started_at.isoformat(),
                "purpose": purpose,
                "model": self.model,
                "endpoint": self.endpoint,
                "status": "failed",
                "attempts": attempt,
                "latency_seconds": time.perf_counter() - start_time,
                "request": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "attempt_errors": attempt_errors,
            }
        )

    def _write_log(self, record):
        # Authorization headers and API keys are never written.
        with self.log_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
