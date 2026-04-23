from __future__ import annotations

from typing import Any

from src.llm.types import LLMBackend, LLMRequest, LLMResponse, TokenUsage


class LlamaCppBackend(LLMBackend):
    """llama-cpp-python adapter that implements the project LLM contract."""

    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        chat_format: str | None = None,
        verbose: bool = False,
        model: Any | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be a non-empty string.")
        if n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer.")
        if n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be greater than or equal to -1.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if top_p <= 0:
            raise ValueError("top_p must be a positive number.")

        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._stop = list(stop or [])
        self._seed = seed
        self._chat_format = chat_format
        self._verbose = verbose
        self._model = model if model is not None else self._load_model()

    @property
    def backend_name(self) -> str:
        return "llamacpp"

    def generate(self, request: LLMRequest) -> LLMResponse:
        query = request.query.strip()
        if not query:
            raise ValueError("request.query must be a non-empty string.")

        completion_kwargs = self._build_generation_kwargs(request)
        raw_response = self._generate_with_available_api(request, completion_kwargs)
        return self._parse_response(raw_response)

    def _build_generation_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": request.params.temperature,
            "max_tokens": request.params.max_tokens,
            "top_p": request.params.top_p,
            "stop": request.params.stop or self._stop,
        }

        if kwargs["max_tokens"] is None:
            kwargs["max_tokens"] = self._max_tokens
        if kwargs["top_p"] is None:
            kwargs["top_p"] = self._top_p
        if not kwargs["stop"]:
            kwargs.pop("stop")

        return kwargs

    def _generate_with_available_api(
        self, request: LLMRequest, completion_kwargs: dict[str, Any]
    ) -> Any:
        if hasattr(self._model, "create_chat_completion"):
            return self._model.create_chat_completion(
                messages=self._build_chat_messages(request),
                stream=False,
                **completion_kwargs,
            )

        if hasattr(self._model, "create_completion"):
            return self._model.create_completion(
                prompt=self._build_prompt(request),
                stream=False,
                **completion_kwargs,
            )

        raise ValueError(
            "Provided llama.cpp model object must expose either "
            "'create_chat_completion' or 'create_completion'."
        )

    def _build_chat_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        user_prompt = self._build_prompt(request)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _build_prompt(self, request: LLMRequest) -> str:
        if request.context_text:
            return (
                "Use the provided context to answer the question. "
                "If context is insufficient, say so.\n\n"
                f"Context:\n{request.context_text}\n\n"
                f"Question:\n{request.query.strip()}"
            )
        return request.query.strip()

    def _parse_response(self, response: Any) -> LLMResponse:
        if not isinstance(response, dict):
            return LLMResponse(text=str(response), raw=response)

        choices = response.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(first_choice, dict):
            first_choice = {}

        message = first_choice.get("message")
        text = ""
        if isinstance(message, dict):
            content = message.get("content")
            text = content if isinstance(content, str) else str(content or "")
        elif "text" in first_choice:
            candidate_text = first_choice.get("text")
            text = candidate_text if isinstance(candidate_text, str) else str(
                candidate_text or ""
            )

        finish_reason = first_choice.get("finish_reason")
        usage = self._parse_usage(response.get("usage"))

        model = response.get("model")
        model_name = model if isinstance(model, str) else None

        return LLMResponse(
            text=text,
            model=model_name,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage,
            raw=response,
        )

    def _parse_usage(self, usage: Any) -> TokenUsage | None:
        if not isinstance(usage, dict):
            return None

        raw_input_tokens = usage.get("prompt_tokens")
        raw_output_tokens = usage.get("completion_tokens")
        raw_total_tokens = usage.get("total_tokens")

        if not isinstance(raw_input_tokens, int):
            return None
        if not isinstance(raw_output_tokens, int):
            return None
        if not isinstance(raw_total_tokens, int):
            return None

        return TokenUsage(
            input_tokens=raw_input_tokens,
            output_tokens=raw_output_tokens,
            total_tokens=raw_total_tokens,
        )

    def _load_model(self) -> Any:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required to use LlamaCppBackend."
            ) from exc

        model_kwargs: dict[str, Any] = {
            "model_path": self._model_path,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "verbose": self._verbose,
        }
        if self._seed is not None:
            model_kwargs["seed"] = self._seed
        if self._chat_format is not None:
            model_kwargs["chat_format"] = self._chat_format

        return Llama(**model_kwargs)
