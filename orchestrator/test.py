# ============================================================
# COMPLETE REMOTE MODEL DEBUG CELL
# ============================================================

from __future__ import annotations

import json
import os
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

REPAIR_MODEL_URL = os.environ.get(
    "REPAIR_MODEL_URL",
    "https://marsupial-abnormal-unselect.ngrok-free.dev",
).rstrip("/")

REMOTE_TIMEOUT_SECONDS = (
    float(os.environ["REPAIR_MODEL_TIMEOUT"])
    if os.environ.get("REPAIR_MODEL_TIMEOUT")
    else 120.0
)

DEBUG_MAX_NEW_TOKENS = 128
DEBUG_TEMPERATURE = 0.0


# ------------------------------------------------------------
# DEBUG STATE
# ------------------------------------------------------------

LAST_REMOTE_DEBUG: dict[str, Any] = {}


def _request_id() -> str:
    return (
        datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%f"
        )
        + "-"
        + str(os.getpid())
    )


def _reset_debug(request_id: str) -> None:
    global LAST_REMOTE_DEBUG

    LAST_REMOTE_DEBUG = {
        "request_id": request_id,
        "status": "started",
        "url": None,
        "language": None,
        "prompt_chars": None,
        "max_new_tokens": None,
        "temperature": None,
        "http_status": None,
        "raw_response": None,
        "parsed_response": None,
        "model_response": None,
        "error": None,
        "traceback": None,
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "finished_at": None,
    }


def _finish_debug(status: str) -> None:
    LAST_REMOTE_DEBUG["status"] = status
    LAST_REMOTE_DEBUG["finished_at"] = (
        datetime.now(timezone.utc).isoformat()
    )


def _log(
    request_id: str,
    message: str,
    value: Any = None,
) -> None:
    prefix = f"[remote-debug:{request_id}]"

    if value is None:
        print(
            f"{prefix} {message}",
            flush=True,
        )
    else:
        print(
            f"{prefix} {message}: {value!r}",
            flush=True,
        )


def show_last_remote_debug() -> dict[str, Any]:
    print(
        json.dumps(
            LAST_REMOTE_DEBUG,
            indent=2,
            default=str,
        ),
        flush=True,
    )

    return dict(LAST_REMOTE_DEBUG)


# ------------------------------------------------------------
# REMOTE HEALTH CHECK
# ------------------------------------------------------------

def debug_remote_health() -> dict[str, Any]:
    request_id = _request_id()
    url = f"{REPAIR_MODEL_URL}/health"

    print("=" * 100)
    print("REMOTE MODEL HEALTH CHECK")
    print("=" * 100)
    print("Request ID:", request_id)
    print("URL:", url)

    request = urllib.request.Request(
        url,
        headers={
            "ngrok-skip-browser-warning": "1",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            body_text = response.read().decode(
                "utf-8",
                errors="replace",
            )

            status = response.status

        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = body_text

        result = {
            "success": True,
            "request_id": request_id,
            "url": url,
            "status": status,
            "response": body,
        }

        print("HTTP status:", status)
        print("Response:")
        print(json.dumps(body, indent=2, default=str))
        print("=" * 100)

        return result

    except Exception as error:
        result = {
            "success": False,
            "request_id": request_id,
            "url": url,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }

        print("HEALTH CHECK FAILED")
        traceback.print_exc()
        print("=" * 100)

        return result


# ------------------------------------------------------------
# DIRECT /GENERATE TEST
# ------------------------------------------------------------

def debug_remote_generation(
    language: str = "java",
    prompt: str | None = None,
    max_new_tokens: int = DEBUG_MAX_NEW_TOKENS,
    temperature: float = DEBUG_TEMPERATURE,
) -> dict[str, Any]:
    if prompt is None:
        prompt = """// ISSUE
// Incorrect loop condition

// RULES
// Return only the corrected Java method.
// Preserve the method signature exactly.
// Preserve all identifiers exactly.
// Do not add comments or explanations.
// Change only the loop condition.

public Order findOrder(int orderId) {
    for (int i = 0; i < orders.size() - 1; i++) {
        if (orders.get(i).getOrderId() == orderId) {
            return orders.get(i);
        }
    }

    return null;
}

<FILL_ME>
"""

    request_id = _request_id()
    url = f"{REPAIR_MODEL_URL}/generate"

    _reset_debug(request_id)

    LAST_REMOTE_DEBUG["url"] = url
    LAST_REMOTE_DEBUG["language"] = language
    LAST_REMOTE_DEBUG["prompt_chars"] = len(prompt)
    LAST_REMOTE_DEBUG["max_new_tokens"] = max_new_tokens
    LAST_REMOTE_DEBUG["temperature"] = temperature

    payload = {
        "language": language,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    print("\n" + "=" * 100)
    print("DIRECT REMOTE GENERATION TEST")
    print("=" * 100)

    _log(request_id, "URL", url)
    _log(request_id, "payload", payload)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "1",
        },
        method="POST",
    )

    started = time.monotonic()

    try:
        _log(
            request_id,
            "sending POST request",
        )

        with urllib.request.urlopen(
            request,
            timeout=REMOTE_TIMEOUT_SECONDS,
        ) as response:
            http_status = response.status
            raw_text = response.read().decode(
                "utf-8",
                errors="replace",
            )

        elapsed = time.monotonic() - started

        LAST_REMOTE_DEBUG["http_status"] = http_status
        LAST_REMOTE_DEBUG["raw_response"] = raw_text

        _log(
            request_id,
            "HTTP status",
            http_status,
        )

        _log(
            request_id,
            "elapsed seconds",
            round(elapsed, 2),
        )

        print("\n" + "-" * 100)
        print("RAW HTTP RESPONSE")
        print("-" * 100)
        print(raw_text)
        print("-" * 100)

        try:
            response_data = json.loads(raw_text)
        except json.JSONDecodeError as error:
            LAST_REMOTE_DEBUG["error"] = str(error)
            LAST_REMOTE_DEBUG["traceback"] = (
                traceback.format_exc()
            )

            _finish_debug("invalid_json")

            raise RuntimeError(
                f"Remote response was not valid JSON: {error}"
            ) from error

        LAST_REMOTE_DEBUG["parsed_response"] = response_data

        if isinstance(response_data, dict):
            reply = None

            for key in (
                "response",
                "generated_text",
                "text",
                "output",
            ):
                value = response_data.get(key)

                if isinstance(value, str):
                    reply = value
                    break

            if reply is None:
                raise RuntimeError(
                    "Response JSON does not contain a string "
                    "field named response, generated_text, text, "
                    f"or output. Keys: {list(response_data.keys())}"
                )

        elif isinstance(response_data, str):
            reply = response_data

        else:
            raise RuntimeError(
                "Unexpected response type: "
                f"{type(response_data).__name__}"
            )

        LAST_REMOTE_DEBUG["model_response"] = reply

        print("\n" + "-" * 100)
        print("MODEL RESPONSE")
        print("-" * 100)
        print(reply)
        print("-" * 100)

        _finish_debug("success")

        print("RESULT: model generated a response")
        print("Response characters:", len(reply))
        print("Response lines:", len(reply.splitlines()))
        print("=" * 100)

        return {
            "success": True,
            "request_id": request_id,
            "http_status": http_status,
            "response": response_data,
            "model_response": reply,
            "debug": dict(LAST_REMOTE_DEBUG),
        }

    except urllib.error.HTTPError as error:
        detail = error.read().decode(
            "utf-8",
            errors="replace",
        )

        LAST_REMOTE_DEBUG["http_status"] = error.code
        LAST_REMOTE_DEBUG["raw_response"] = detail
        LAST_REMOTE_DEBUG["error"] = (
            f"HTTP {error.code}: {detail}"
        )
        LAST_REMOTE_DEBUG["traceback"] = (
            traceback.format_exc()
        )

        _finish_debug("http_error")

        print("\n" + "-" * 100)
        print("REMOTE SERVER HTTP ERROR")
        print("-" * 100)
        print("HTTP status:", error.code)
        print("Response body:")
        print(detail)
        print("-" * 100)
        traceback.print_exc()
        print("=" * 100)

        return {
            "success": False,
            "request_id": request_id,
            "http_status": error.code,
            "error": str(error),
            "raw_response": detail,
            "debug": dict(LAST_REMOTE_DEBUG),
        }

    except Exception as error:
        LAST_REMOTE_DEBUG["error"] = str(error)
        LAST_REMOTE_DEBUG["traceback"] = (
            traceback.format_exc()
        )

        _finish_debug("client_error")

        print("\n" + "-" * 100)
        print("CLIENT-SIDE REMOTE GENERATION ERROR")
        print("-" * 100)
        print(type(error).__name__, str(error))
        traceback.print_exc()
        print("=" * 100)

        return {
            "success": False,
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "debug": dict(LAST_REMOTE_DEBUG),
        }


# ------------------------------------------------------------
# FULL CODING-AGENT REMOTE PATCH TEST
# ------------------------------------------------------------

def debug_remote_patch_from_file(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    issue: str,
    language: str = "java",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Tests the exact remote repair path using a real repository file.

    Line numbers are 1-based.
    """

    request_id = _request_id()
    url = f"{REPAIR_MODEL_URL}/generate"

    print("\n" + "=" * 100)
    print("FULL REMOTE PATCH TEST")
    print("=" * 100)
    print("Request ID:", request_id)
    print("Repository:", repo_path)
    print("File:", file_path)
    print("Language:", language)
    print("Lines:", start_line, "-", end_line)

    source_path = Path(repo_path) / file_path

    if not source_path.is_file():
        result = {
            "success": False,
            "error": f"File not found: {source_path}",
        }

        print(result["error"])
        print("=" * 100)

        return result

    try:
        original = source_path.read_text(
            encoding="utf-8"
        )

        lines = original.splitlines()

        function_lines = lines[
            start_line - 1:end_line
        ]

        function_text = "\n".join(function_lines)

        prompt = f"""// ISSUE
// {issue.strip()}

// RULES
// Make the smallest possible change.
// Preserve the method signature exactly.
// Preserve all variable names exactly.
// Preserve all method names exactly.
// Preserve all capitalization exactly.
// Do not add comments or explanations.
// Return only the complete corrected Java method.
// Change only the code necessary to fix the issue.

// BUGGY JAVA METHOD
{function_text}

// FILL THE CORRECTED METHOD
<FILL_ME>
"""

        payload = {
            "language": language,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "1",
            },
            method="POST",
        )

        print("\nPROMPT SENT TO MODEL")
        print("-" * 100)
        print(prompt)
        print("-" * 100)

        started = time.monotonic()

        print("Sending request...", flush=True)

        with urllib.request.urlopen(
            request,
            timeout=REMOTE_TIMEOUT_SECONDS,
        ) as response:
            status = response.status
            raw_response = response.read().decode(
                "utf-8",
                errors="replace",
            )

        elapsed = time.monotonic() - started

        print("HTTP status:", status)
        print("Elapsed seconds:", round(elapsed, 2))

        print("\nRAW SERVER RESPONSE")
        print("-" * 100)
        print(raw_response)
        print("-" * 100)

        response_data = json.loads(raw_response)

        if isinstance(response_data, dict):
            reply = next(
                (
                    response_data[key]
                    for key in (
                        "response",
                        "generated_text",
                        "text",
                        "output",
                    )
                    if isinstance(
                        response_data.get(key),
                        str,
                    )
                ),
                None,
            )
        elif isinstance(response_data, str):
            reply = response_data
        else:
            reply = None

        if not reply:
            return {
                "success": False,
                "stage": "response_extraction",
                "response": response_data,
                "raw_response": raw_response,
            }

        print("\nEXTRACTED MODEL RESPONSE")
        print("-" * 100)
        print(reply)
        print("-" * 100)

        repaired_function = reply.strip()

        if repaired_function.startswith(
            "```"
        ):
            repaired_lines = (
                repaired_function.splitlines()
            )

            if repaired_lines:
                repaired_lines = repaired_lines[1:]

            if (
                repaired_lines
                and repaired_lines[-1].strip()
                == "```"
            ):
                repaired_lines = repaired_lines[:-1]

            repaired_function = "\n".join(
                repaired_lines
            ).strip()

        repaired_lines = repaired_function.splitlines()

        if not repaired_lines:
            return {
                "success": False,
                "stage": "empty_model_response",
                "model_response": reply,
            }

        repaired_source = "\n".join(
            lines[: start_line - 1]
            + repaired_lines
            + lines[end_line:]
        )

        print("\nREPAIRED SOURCE PREVIEW")
        print("-" * 100)
        print(repaired_source)
        print("-" * 100)

        if repaired_source == original:
            return {
                "success": False,
                "stage": "unchanged_response",
                "model_response": reply,
            }

        try:
            from src.agents.coding_agent.function_patch import (
                make_diff,
            )

            patch = make_diff(
                file_path,
                original,
                repaired_source,
            )

        except Exception as error:
            print("PATCH CREATION FAILED")
            traceback.print_exc()

            return {
                "success": False,
                "stage": "patch_creation",
                "model_response": reply,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }

        print("\nGENERATED PATCH")
        print("-" * 100)
        print(patch)
        print("-" * 100)
        print("RESULT: model response received and patch created")
        print("=" * 100)

        return {
            "success": True,
            "stage": "patch_created",
            "model_response": reply,
            "patch": patch,
            "raw_response": raw_response,
            "elapsed_seconds": round(elapsed, 2),
        }

    except urllib.error.HTTPError as error:
        detail = error.read().decode(
            "utf-8",
            errors="replace",
        )

        print("\nREMOTE HTTP ERROR")
        print("-" * 100)
        print("HTTP status:", error.code)
        print(detail)
        print("-" * 100)
        traceback.print_exc()
        print("=" * 100)

        return {
            "success": False,
            "stage": "http_request",
            "http_status": error.code,
            "error": str(error),
            "raw_response": detail,
            "traceback": traceback.format_exc(),
        }

    except Exception as error:
        print("\nFULL PATCH DEBUG FAILED")
        print("-" * 100)
        print(type(error).__name__, str(error))
        traceback.print_exc()
        print("-" * 100)
        print("=" * 100)

        return {
            "success": False,
            "stage": "unexpected_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# RUN TESTS
# ============================================================

health_result = debug_remote_health()

if health_result.get("success"):
    generation_result = debug_remote_generation(
        language="java",
        max_new_tokens=128,
        temperature=0.0,
    )

    print("\nFINAL DEBUG SUMMARY")
    print("=" * 100)
    print(
        json.dumps(
            {
                "health_success": health_result.get(
                    "success"
                ),
                "generation_success": generation_result.get(
                    "success"
                ),
                "generation_stage": generation_result.get(
                    "debug",
                    {},
                ).get("status"),
                "model_response_received": bool(
                    generation_result.get(
                        "model_response"
                    )
                ),
                "http_status": generation_result.get(
                    "http_status"
                ),
            },
            indent=2,
            default=str,
        )
    )
    print("=" * 100)
else:
    print(
        "Generation test skipped because /health failed.",
        flush=True,
    )