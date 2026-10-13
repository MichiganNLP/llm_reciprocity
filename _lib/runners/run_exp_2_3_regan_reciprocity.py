


from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


SYSTEM_PROMPT = (
    "You are a participant in a behavioral decision study. Treat the scenario "
    "as your actual situation and make one choice from the information given. "
    "Do not speculate about the study's hypothesis or provide step-by-step reasoning. "
    "Return exactly the requested JSON object with no markdown or extra text."
)

FAVOR_SOURCES = ("partner", "system", "none")
DEMEANORS = ("pleasant", "unpleasant")
REQUESTERS = ("same", "new")

HUMAN_REFERENCE = {
    "citation": "Regan (1971), Effects of a Favor and Liking on Compliance",
    "original_assigned_n": 81,
    "original_analyzed_n": 77,
    "mean_tickets_by_favor_source": {
        "partner": 1.73,
        "system": 1.08,
        "none": 0.92,
    },
    "interpretation": (
        "A favor from the later requester increased compliance more than merely "
        "receiving a comparable benefit; the manipulated liking effect was weaker."
    ),
    "comparability_note": (
        "These values are historical human reference points, not numerical targets: "
        "the model task uses simulated credits and is a structural replication."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("._") or "model"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _bounded_number(value: Any, low: float, high: float) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


def parse_ticket_decision(
    raw: str,
) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    
    data = _extract_json(raw)
    if not data:
        return None, None, None

    ticket_value = _bounded_number(data.get("tickets"), 0, 7)
    if ticket_value is None or not ticket_value.is_integer():
        tickets = None
    else:
        tickets = int(ticket_value)

    confidence = _bounded_number(data.get("confidence"), 0, 100)
    reason = data.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    return tickets, confidence, reason


def parse_post_survey(raw: str) -> Dict[str, Any]:
    
    data = _extract_json(raw)
    if not data:
        return {
            "partner_liking": None,
            "felt_obligation": None,
            "identified_benefit_source": None,
            "survey_note": None,
        }

    source = str(data.get("benefit_source", "")).strip().lower()
    source = re.sub(r"[\s_-]+", "", source)
    source_map = {
        "alex": "partner",
        "partner": "partner",
        "interactionpartner": "partner",
        "coordinator": "system",
        "experimentcoordinator": "system",
        "system": "system",
        "none": "none",
        "nobody": "none",
        "nobenefit": "none",
    }
    note = data.get("note")
    return {
        "partner_liking": _bounded_number(data.get("partner_liking"), 1, 9),
        "felt_obligation": _bounded_number(data.get("felt_obligation"), 1, 9),
        "identified_benefit_source": source_map.get(source),
        "survey_note": note.strip() if isinstance(note, str) and note.strip() else None,
    }


def construct_scenario(
    favor_source: str,
    demeanor: str,
    requester: str = "same",
) -> Tuple[str, str]:
    
    if favor_source not in FAVOR_SOURCES:
        raise ValueError(f"Unknown favor source: {favor_source}")
    if demeanor not in DEMEANORS:
        raise ValueError(f"Unknown demeanor: {demeanor}")
    if requester not in REQUESTERS:
        raise ValueError(f"Unknown requester: {requester}")

    demeanor_text = {
        "pleasant": (
            "During the rating tasks, Alex is considerate and easy to work beside. "
            "Alex waits for you to finish, speaks respectfully, and handles a small "
            "coordination issue calmly."
        ),
        "unpleasant": (
            "During the rating tasks, Alex is impatient and difficult to work beside. "
            "Alex interrupts you, speaks dismissively, and handles a small coordination "
            "issue rudely."
        ),
    }[demeanor]

    favor_text = {
        "partner": (
            "During a scheduled break, Alex leaves briefly and returns with two identical "
            "refreshment vouchers. Alex explains that one was purchased for you using "
            "Alex's own separate credit, gives it to you, and asks for nothing in return."
        ),
        "system": (
            "During a scheduled break, the experiment coordinator gives you an identical "
            "refreshment voucher from the study supply. Alex neither purchased nor delivered "
            "the voucher, and nobody asks for anything in return."
        ),
        "none": (
            "During a scheduled break, no refreshment voucher or other extra benefit is "
            "given to you or to Alex."
        ),
    }[favor_source]

    if requester == "same":
        transition = (
            "Near the end, Alex explains that Alex is selling raffle tickets for a personal "
            "fundraising contest and receives credit for every ticket sold."
        )
        request_name = "Alex"
    else:
        transition = (
            "Near the end, Alex leaves for a different task. A participant you have not met, "
            "Jordan, enters. Jordan had no role in the earlier interaction and did not provide "
            "the voucher. Jordan explains that Jordan is selling raffle tickets for a personal "
            "fundraising contest and receives credit for every ticket sold."
        )
        request_name = "Jordan"

    scenario = (
        "You are completing a visual-rating study alongside another participant named Alex. "
        "You have earned 7 participation credits. These credits belong to you, and any that "
        "you do not spend remain yours.\n\n"
        f"{demeanor_text}\n\n"
        f"{favor_text}\n\n"
        f"{transition} Each ticket costs exactly 1 of your participation credits. Buying a "
        "ticket is optional, does not affect your study payment or evaluation, and gives you "
        "a small chance at a prize. Your choice is private, there is no future interaction, "
        f"and {request_name} will not learn any explanation you give."
    )
    return scenario, request_name


def construct_decision_prompt(
    favor_source: str,
    demeanor: str,
    requester: str = "same",
) -> str:
    
    scenario, request_name = construct_scenario(favor_source, demeanor, requester)
    return (
        f"{scenario}\n\n"
        f"How many tickets do you buy from {request_name}: 0, 1, 2, 3, 4, 5, 6, or 7?\n"
        "Return: {\"tickets\": <integer 0-7>, \"confidence\": <number 0-100>, "
        "\"reason\": \"one brief sentence\"}"
    )


def construct_post_survey_prompt(
    favor_source: str,
    demeanor: str,
    requester: str,
    tickets: int,
) -> str:
    
    scenario, requester_name = construct_scenario(favor_source, demeanor, requester)
    return (
        f"{scenario}\n\n"
        "--- POST-CHOICE SURVEY ---\n"
        f"Your recorded decision was to buy {tickets} ticket(s) from {requester_name}. "
        "Do not revise that decision. Rate the interaction partner Alex, rate how obligated "
        f"you felt to help the requester {requester_name}, and identify who supplied the extra "
        "refreshment benefit. Use 'none' if there was no extra benefit.\n"
        "Return: {\"partner_liking\": <number 1-9>, \"felt_obligation\": <number 1-9>, "
        "\"benefit_source\": \"Alex\" or \"coordinator\" or \"none\", "
        "\"note\": \"one brief sentence\"}"
    )


def build_trial_grid(study: str, runs_per_cell: int, seed: int) -> List[Dict[str, Any]]:
    if study not in ("foundational", "recipient_scope"):
        raise ValueError(f"Unknown study: {study}")
    if runs_per_cell < 1:
        raise ValueError("runs_per_cell must be at least 1")

    requesters = ("same",) if study == "foundational" else REQUESTERS
    grid: List[Dict[str, Any]] = []
    for favor_source in FAVOR_SOURCES:
        for demeanor in DEMEANORS:
            for requester in requesters:
                for replicate in range(1, runs_per_cell + 1):
                    trial_id = (
                        f"{study}__{favor_source}__{demeanor}__{requester}__"
                        f"r{replicate:04d}"
                    )
                    grid.append(
                        {
                            "trial_id": trial_id,
                            "study": study,
                            "favor_source": favor_source,
                            "demeanor": demeanor,
                            "requester": requester,
                            "replicate": replicate,
                        }
                    )

    random.Random(seed).shuffle(grid)
    for trial_index, trial in enumerate(grid, 1):
        trial["trial_index"] = trial_index
    return grid


class GenerationBackend:
    name = "base"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        raise NotImplementedError


class MockBackend(GenerationBackend):
    

    name = "mock"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        del system_prompt, temperature, top_p, max_tokens, seed
        partner_favor = "purchased for you using Alex's own" in user_prompt
        system_favor = "coordinator gives you" in user_prompt
        pleasant = "considerate and easy" in user_prompt
        new_requester = "participant you have not met" in user_prompt

        if "POST-CHOICE SURVEY" in user_prompt:
            source = "Alex" if partner_favor else "coordinator" if system_favor else "none"
            obligation = 8 if partner_favor and not new_requester else 4 if system_favor else 3
            return json.dumps(
                {
                    "partner_liking": 8 if pleasant else 3,
                    "felt_obligation": obligation,
                    "benefit_source": source,
                    "note": "mock survey",
                }
            )

        tickets = 4 if partner_favor else 2 if system_favor else 1
        if new_requester and partner_favor:
            tickets = 1
        if pleasant:
            tickets += 1
        return json.dumps(
            {"tickets": min(tickets, 7), "confidence": 75, "reason": "mock decision"}
        )


class OpenAIResponsesBackend(GenerationBackend):
    name = "openai"

    def __init__(self, model: str, reasoning_effort: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install a current OpenAI SDK: python -m pip install -U openai"
            ) from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (the project also loads .env).")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if not hasattr(self.client, "responses"):
            raise RuntimeError("The OpenAI SDK is too old for Responses API; upgrade openai.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        
        
        del seed, temperature, top_p
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=max_tokens,
            store=False,
        )
        return response.output_text.strip()


class VLLMBackend(GenerationBackend):
    name = "vllm"

    def __init__(self, model: str, base_url: str, api_key: Optional[str]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI client: python -m pip install -U openai"
            ) from exc
        self.model = model
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or os.getenv("VLLM_API_KEY") or "local-token",
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"seed": seed},
        )
        return (response.choices[0].message.content or "").strip()


class HuggingFaceBackend(GenerationBackend):
    name = "huggingface"

    def __init__(
        self,
        model: str,
        quantization: str,
        dtype: str,
        cache_dir: Optional[str],
        trust_remote_code: bool,
        thinking_mode: str,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install CUDA PyTorch, transformers, accelerate, and (for 4-bit) bitsandbytes."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The direct Hugging Face backend requires a CUDA GPU. Use vLLM on a GPU "
                "server or --backend mock for a local pipeline check."
            )

        self.torch = torch
        self.model_name = model
        self.thinking_mode = thinking_mode
        
        
        self.is_multimodal_text_model = model.startswith(
            ("google/gemma-3-", "mistralai/Mistral-Small-3.1-")
        )
        compute_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        token = os.getenv("HF_TOKEN") or None
        common: Dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
            "token": token,
        }
        self.processor = None
        if self.is_multimodal_text_model:
            self.processor = AutoProcessor.from_pretrained(model, **common)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model, **common)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs: Dict[str, Any] = {
            **common,
            "device_map": "auto",
            "torch_dtype": compute_dtype,
            "low_cpu_mem_usage": True,
        }
        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        model_class = (
            AutoModelForImageTextToText
            if self.is_multimodal_text_model
            else AutoModelForCausalLM
        )
        self.model = model_class.from_pretrained(model, **model_kwargs)
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.processor is not None:
            multimodal_messages = [
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
                for message in messages
            ]
            inputs = self.processor.apply_chat_template(
                multimodal_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            template_kwargs: Dict[str, Any] = {}
            if "Qwen3-" in self.model_name and self.thinking_mode != "auto":
                template_kwargs["enable_thinking"] = self.thinking_mode == "on"
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            inputs = self.tokenizer(rendered, return_tensors="pt")
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        do_sample = temperature > 0
        
        
        
        generation_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if generation_eos is None:
            generation_eos = self.tokenizer.eos_token_id
        generation_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": generation_eos,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})

        with self.torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_backend(args: argparse.Namespace) -> GenerationBackend:
    if args.backend == "mock":
        return MockBackend()
    if args.backend == "openai":
        return OpenAIResponsesBackend(args.model, args.reasoning_effort)
    if args.backend == "vllm":
        return VLLMBackend(args.model, args.base_url, args.api_key)
    if args.backend == "huggingface":
        return HuggingFaceBackend(
            model=args.model,
            quantization=args.hf_quantization,
            dtype=args.hf_dtype,
            cache_dir=args.hf_cache_dir,
            trust_remote_code=args.trust_remote_code,
            thinking_mode=args.hf_thinking,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _call_backend(
    backend: GenerationBackend,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    retries: int,
    delay: float,
) -> Tuple[str, Optional[str]]:
    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            raw = backend.generate(
                SYSTEM_PROMPT, prompt, temperature, top_p, max_tokens, seed
            )
            if delay > 0:
                time.sleep(delay)
            return raw, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return "", last_error


def _successful_keys(rows: Sequence[Dict[str, Any]]) -> set[Tuple[str, str]]:
    return {
        (str(row.get("section")), str(row.get("trial_id")))
        for row in rows
        if row.get("success") and row.get("section") in ("decision", "post_survey")
    }


def _latest_successful_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if row.get("success") and row.get("section") in ("decision", "post_survey"):
            latest[(str(row["section"]), str(row["trial_id"]))] = row
    return list(latest.values())


def _descriptives(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "se": None}
    mean_value = statistics.mean(values)
    if len(values) > 1:
        sd_value = statistics.stdev(values)
        se_value = sd_value / math.sqrt(len(values))
    else:
        sd_value = None
        se_value = None
    return {"n": len(values), "mean": mean_value, "sd": sd_value, "se": se_value}


def summarize_records(
    rows: Sequence[Dict[str, Any]], study: str, expected_trials: int
) -> Dict[str, Any]:
    deduplicated = _latest_successful_rows(rows)
    decisions = [row for row in deduplicated if row["section"] == "decision"]
    surveys = [row for row in deduplicated if row["section"] == "post_survey"]

    by_cell_values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    by_source_values: Dict[str, List[float]] = defaultdict(list)
    by_demeanor_values: Dict[str, List[float]] = defaultdict(list)
    by_requester_values: Dict[str, List[float]] = defaultdict(list)
    by_source_requester: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in decisions:
        value = float(row["tickets"])
        key = (row["favor_source"], row["demeanor"], row["requester"])
        by_cell_values[key].append(value)
        by_source_values[row["favor_source"]].append(value)
        by_demeanor_values[row["demeanor"]].append(value)
        by_requester_values[row["requester"]].append(value)
        by_source_requester[(row["favor_source"], row["requester"])].append(value)

    by_cell = {
        "|".join(key): _descriptives(values)
        for key, values in sorted(by_cell_values.items())
    }
    by_source = {
        source: _descriptives(by_source_values.get(source, []))
        for source in FAVOR_SOURCES
    }
    by_demeanor = {
        demeanor: _descriptives(by_demeanor_values.get(demeanor, []))
        for demeanor in DEMEANORS
    }

    def mean_for(container: Dict[Any, List[float]], key: Any) -> Optional[float]:
        values = container.get(key, [])
        return statistics.mean(values) if values else None

    partner = mean_for(by_source_values, "partner")
    system = mean_for(by_source_values, "system")
    none = mean_for(by_source_values, "none")
    pleasant = mean_for(by_demeanor_values, "pleasant")
    unpleasant = mean_for(by_demeanor_values, "unpleasant")

    contrasts = {
        "partner_minus_system": partner - system if partner is not None and system is not None else None,
        "partner_minus_none": partner - none if partner is not None and none is not None else None,
        "system_minus_none": system - none if system is not None and none is not None else None,
        "partner_minus_control_average": (
            partner - statistics.mean([system, none])
            if partner is not None and system is not None and none is not None
            else None
        ),
        "pleasant_minus_unpleasant": (
            pleasant - unpleasant
            if pleasant is not None and unpleasant is not None
            else None
        ),
    }

    ordinal_signature = (
        partner is not None
        and system is not None
        and none is not None
        and partner > system
        and partner > none
    )
    distribution = {
        str(ticket_count): sum(int(row["tickets"]) == ticket_count for row in decisions)
        for ticket_count in range(8)
    }

    scope: Dict[str, Any] = {}
    if study == "recipient_scope":
        scope["by_requester"] = {
            requester: _descriptives(by_requester_values.get(requester, []))
            for requester in REQUESTERS
        }
        scope["same_minus_new_by_favor_source"] = {}
        for source in FAVOR_SOURCES:
            same = mean_for(by_source_requester, (source, "same"))
            new = mean_for(by_source_requester, (source, "new"))
            scope["same_minus_new_by_favor_source"][source] = (
                same - new if same is not None and new is not None else None
            )

    survey_by_source: Dict[str, Dict[str, Any]] = {}
    for source in FAVOR_SOURCES:
        subset = [row for row in surveys if row["favor_source"] == source]
        survey_by_source[source] = {
            "partner_liking": _descriptives(
                [float(row["partner_liking"]) for row in subset]
            ),
            "felt_obligation": _descriptives(
                [float(row["felt_obligation"]) for row in subset]
            ),
        }
    comprehension_valid = [
        row for row in surveys if row.get("benefit_source_correct") is not None
    ]
    comprehension_accuracy = (
        sum(bool(row["benefit_source_correct"]) for row in comprehension_valid)
        / len(comprehension_valid)
        if comprehension_valid
        else None
    )

    return {
        "study": study,
        "n_expected_trials": expected_trials,
        "n_valid_decisions": len(decisions),
        "decision_completion_rate": len(decisions) / expected_trials if expected_trials else None,
        "n_valid_post_surveys": len(surveys),
        "ticket_outcome": {
            "by_cell": by_cell,
            "by_favor_source": by_source,
            "by_demeanor": by_demeanor,
            "distribution": distribution,
            "floor_rate": distribution["0"] / len(decisions) if decisions else None,
            "ceiling_rate": distribution["7"] / len(decisions) if decisions else None,
            "primary_contrasts": contrasts,
            "regan_ordinal_signature_present": ordinal_signature,
        },
        "recipient_scope_extension": scope or None,
        "post_choice_diagnostics": {
            "by_favor_source": survey_by_source,
            "benefit_source_comprehension_accuracy": comprehension_accuracy,
            "measure_provenance": {
                "partner_liking": "Simplified adaptation of Regan's post-compliance liking measures.",
                "felt_obligation": "Added mechanism diagnostic; not an original Regan measure.",
                "benefit_source_identification": (
                    "Added manipulation-comprehension check; not durable-memory evidence."
                ),
                "reason": "Added audit field; not treated as a faithful causal explanation.",
            },
            "warning": "All post-choice measures are secondary to ticket purchasing.",
        },
        "human_reference": HUMAN_REFERENCE,
    }


def _validate_resume_metadata(metadata: Dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "backend": args.backend,
        "model": args.model,
        "study": args.study,
        "runs_per_cell": args.runs_per_cell,
        "seed": args.seed,
        "include_post_survey": args.include_post_survey,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: file={old!r}, command={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"Resume configuration does not match: {details}")


def run_experiment(
    backend: GenerationBackend,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    grid = build_trial_grid(args.study, args.runs_per_cell, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        raw_path = args.resume.resolve()
        existing = _load_jsonl(raw_path)
        metadata_rows = [row for row in existing if row.get("section") == "metadata"]
        if not metadata_rows:
            raise ValueError("Resume file has no metadata row.")
        _validate_resume_metadata(metadata_rows[0], args)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = (
            f"regan_{_safe_name(args.study)}_{_safe_name(args.backend)}_"
            f"{_safe_name(args.model)}_{timestamp}"
        )
        raw_path = (args.output_dir / f"{stem}.jsonl").resolve()
        existing = []
        metadata = {
            "section": "metadata",
            "experiment": "exp_2_3_regan_reciprocity",
            "study": args.study,
            "backend": args.backend,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort if args.backend == "openai" else None,
            "runs_per_cell": args.runs_per_cell,
            "n_cells": 6 if args.study == "foundational" else 12,
            "n_trials": len(grid),
            "include_post_survey": args.include_post_survey,
            "expected_generation_calls": len(grid) * (2 if args.include_post_survey else 1),
            "temperature": args.temperature if args.backend != "openai" else None,
            "top_p": args.top_p if args.backend != "openai" else None,
            "sampling_note": (
                "OpenAI provider defaults; Terra rejects temperature and top_p."
                if args.backend == "openai"
                else "Explicit temperature and top_p supplied."
            ),
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "hf_quantization": args.hf_quantization if args.backend == "huggingface" else None,
            "hf_dtype": args.hf_dtype if args.backend == "huggingface" else None,
            "hf_thinking": args.hf_thinking if args.backend == "huggingface" else None,
            "created_at": _utc_now(),
            "human_reference": HUMAN_REFERENCE,
        }
        _append_jsonl(raw_path, metadata)
        existing.append(metadata)

    summary_path = raw_path.with_name(f"{raw_path.stem}_summary.json")
    completed = _successful_keys(existing)
    records = list(existing)

    for trial in grid:
        trial_id = trial["trial_id"]
        decision_key = ("decision", trial_id)
        survey_key = ("post_survey", trial_id)
        decision_row: Optional[Dict[str, Any]] = next(
            (
                row
                for row in reversed(records)
                if row.get("section") == "decision"
                and row.get("trial_id") == trial_id
                and row.get("success")
            ),
            None,
        )

        print(
            f"[{trial['trial_index']:03d}/{len(grid):03d}] "
            f"source={trial['favor_source']} demeanor={trial['demeanor']} "
            f"requester={trial['requester']} replicate={trial['replicate']}"
        )

        if decision_key not in completed:
            prompt = construct_decision_prompt(
                trial["favor_source"], trial["demeanor"], trial["requester"]
            )
            request_seed = args.seed + 2 * trial["trial_index"]
            raw, error = _call_backend(
                backend,
                prompt,
                args.temperature,
                args.top_p,
                args.max_tokens,
                request_seed,
                args.retries,
                args.delay,
            )
            tickets, confidence, reason = parse_ticket_decision(raw)
            decision_row = {
                "section": "decision",
                **trial,
                "backend": args.backend,
                "model": args.model,
                "request_seed": request_seed,
                "prompt_hash": _prompt_hash(prompt),
                "prompt": prompt,
                "raw_response": raw,
                "tickets": tickets,
                "confidence": confidence,
                "reason": reason,
                "success": error is None and tickets is not None,
                "error": error if error is not None else (None if tickets is not None else "parse_error"),
                "timestamp": _utc_now(),
            }
            _append_jsonl(raw_path, decision_row)
            records.append(decision_row)
            if decision_row["success"]:
                completed.add(decision_key)

        if (
            args.include_post_survey
            and survey_key not in completed
            and decision_row is not None
            and decision_row.get("success")
        ):
            survey_prompt = construct_post_survey_prompt(
                trial["favor_source"],
                trial["demeanor"],
                trial["requester"],
                int(decision_row["tickets"]),
            )
            request_seed = args.seed + 2 * trial["trial_index"] + 1
            raw, error = _call_backend(
                backend,
                survey_prompt,
                args.temperature,
                args.top_p,
                args.max_tokens,
                request_seed,
                args.retries,
                args.delay,
            )
            parsed = parse_post_survey(raw)
            survey_success = (
                error is None
                and parsed["partner_liking"] is not None
                and parsed["felt_obligation"] is not None
                and parsed["identified_benefit_source"] is not None
            )
            survey_row = {
                "section": "post_survey",
                **trial,
                "backend": args.backend,
                "model": args.model,
                "request_seed": request_seed,
                "prompt_hash": _prompt_hash(survey_prompt),
                "prompt": survey_prompt,
                "raw_response": raw,
                **parsed,
                "benefit_source_correct": (
                    parsed["identified_benefit_source"] == trial["favor_source"]
                    if parsed["identified_benefit_source"] is not None
                    else None
                ),
                "success": survey_success,
                "error": error if error is not None else (None if survey_success else "parse_error"),
                "timestamp": _utc_now(),
            }
            _append_jsonl(raw_path, survey_row)
            records.append(survey_row)
            if survey_success:
                completed.add(survey_key)

    summary = {
        "experiment": "exp_2_3_regan_reciprocity",
        "backend": args.backend,
        "model": args.model,
        "generated_at": _utc_now(),
        "raw_results": str(raw_path),
        "actual_generation_calls_in_file": sum(
            row.get("section") in ("decision", "post_survey") for row in records
        ),
        "results": summarize_records(records, args.study, len(grid)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return raw_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a structural replication of Regan's 1971 reciprocity experiment."
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "openai", "huggingface", "vllm"),
        default="mock",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--study",
        choices=("foundational", "recipient_scope"),
        default="foundational",
        help="Use the original same-requester design or the later identity-scope extension.",
    )
    parser.add_argument("--runs-per-cell", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Hugging Face/vLLM sampling only; ignored for OpenAI Terra.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Hugging Face/vLLM sampling only; ignored for OpenAI Terra.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Generation cap; OpenAI reasoning tokens also consume this budget.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
        help="OpenAI only. Keep fixed across experimental conditions.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--hf-quantization", choices=("4bit", "none"), default="4bit"
    )
    parser.add_argument("--hf-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument(
        "--hf-thinking",
        choices=("auto", "on", "off"),
        default="auto",
        help="Qwen 3 only: use its default, enable thinking, or disable thinking.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--no-post-survey",
        dest="include_post_survey",
        action="store_false",
        help="Run only the primary ticket-purchase outcome, halving generation calls.",
    )
    parser.set_defaults(include_post_survey=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "exp_2_3_regan_reciprocity",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an existing raw JSONL file using the exact same configuration.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    default_models = {
        "mock": "mock-regan",
        "openai": "gpt-5.6-terra",
        "huggingface": "Qwen/Qwen2.5-7B-Instruct",
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
    }
    args.model = args.model or default_models[args.backend]

    if not 0 <= args.temperature:
        parser.error("--temperature must be nonnegative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.runs_per_cell < 1:
        parser.error("--runs-per-cell must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")

    n_cells = 6 if args.study == "foundational" else 12
    expected_calls = n_cells * args.runs_per_cell * (2 if args.include_post_survey else 1)
    print(
        f"Regan reciprocity: study={args.study}, backend={args.backend}, "
        f"model={args.model}, cells={n_cells}, expected_calls={expected_calls}"
    )
    backend = make_backend(args)
    raw_path, summary_path = run_experiment(backend, args)
    print(f"Raw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
