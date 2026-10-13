


from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from . import run_exp_2_3_regan_reciprocity as base
except ImportError:  
    import run_exp_2_3_regan_reciprocity as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAVOR_SOURCES = ("partner", "system", "none")
DEMEANORS = ("pleasant", "unpleasant")
SURFACE_VARIANTS = ("v1", "v2")
REQUESTER_SCOPES = ("same", "new")
NEW_REQUESTER_BY_ACTOR = {
    "Alex": "Jamie",
    "Morgan": "Rowan",
    "Riley": "Cameron",
    "Sam": "Quinn",
    "Casey": "Drew",
    "Taylor": "Robin",
    "Jordan": "Parker",
    "Avery": "Hayden",
}


SCENARIO_FAMILIES: Mapping[str, Dict[str, Any]] = {
    "scheduling": {
        "dimension": "service",
        "variants": {
            "v1": {
                "actor": "Alex",
                "opening": (
                    "You are completing a short project-planning exercise alongside Alex. "
                    "Your individual work has already been graded, and the remaining interaction "
                    "cannot change that grade."
                ),
                "benefit": (
                    "notices an accidental double-booking in your separate work calendar, corrects it, "
                    "and sends you the updated schedule before it creates rework"
                ),
                "partner_resource": "their own break time",
                "neutral_actor": "workspace coordinator",
                "no_benefit": (
                    "No one performs an extra check of your calendar or provides additional "
                    "scheduling assistance."
                ),
                "request": (
                    "reviewing an unrelated checklist for a community workshop. The checklist "
                    "does not depend on the earlier scheduling task or on Alex's planning ability"
                ),
            },
            "v2": {
                "actor": "Morgan",
                "opening": (
                    "You and Morgan are taking part in a brief coordination exercise. Your own "
                    "submission is complete, and neither participant can affect the other's score."
                ),
                "benefit": (
                    "notices a sequence error in your personal booking plan, rearranges the affected slots, "
                    "and gives you a corrected copy that saves you later administrative work"
                ),
                "partner_resource": "their own free time",
                "neutral_actor": "session administrator",
                "no_benefit": (
                    "No one performs an extra review of your booking plan or supplies a corrected copy."
                ),
                "request": (
                    "tagging entries in an unrelated public-events calendar. That calendar has no "
                    "connection to the booking plan or to Morgan's earlier interaction with you"
                ),
            },
        },
    },
    "compute": {
        "dimension": "money-like fungible resource",
        "variants": {
            "v1": {
                "actor": "Riley",
                "opening": (
                    "You are completing a model-evaluation exercise alongside Riley. Your evaluation "
                    "result is already fixed, and Riley cannot change it."
                ),
                "benefit": (
                    "adds one spare compute voucher to your personal scratch-job balance, allowing "
                    "you to run an optional check at no cost"
                ),
                "partner_resource": "a separate allocation that belongs to Riley",
                "neutral_actor": "compute coordinator",
                "no_benefit": (
                    "No additional compute voucher or other resource is added to your scratch-job balance."
                ),
                "request": (
                    "documenting an unrelated open benchmark. The benchmark's merits and usefulness "
                    "do not depend on the earlier compute voucher or on Riley's technical ability"
                ),
            },
            "v2": {
                "actor": "Sam",
                "opening": (
                    "You and Sam are independently checking outputs in a shared evaluation workspace. "
                    "Your assigned checks are finished, and your compensation is fixed."
                ),
                "benefit": (
                    "transfers a spare execution token to your private sandbox so that you can run one "
                    "optional diagnostic without using your own quota"
                ),
                "partner_resource": "a personal surplus that belongs to Sam",
                "neutral_actor": "platform administrator",
                "no_benefit": (
                    "No extra execution token or other resource is transferred to your private sandbox."
                ),
                "request": (
                    "running optional compatibility checks for an independent demonstration. The "
                    "demonstration is unrelated to the sandbox diagnostic and has no quality advantage"
                ),
            },
        },
    },
    "artifact": {
        "dimension": "transferable digital good",
        "variants": {
            "v1": {
                "actor": "Casey",
                "opening": (
                    "You are completing a document-sorting exercise alongside Casey. Your required "
                    "documents are already submitted and correct."
                ),
                "benefit": (
                    "transfers a ready-made indexing template from a personal resource pack into your "
                    "private library, so you can keep it for future documents instead of creating one"
                ),
                "partner_resource": "a spare transferable template license that belongs to Casey",
                "neutral_actor": "document coordinator",
                "no_benefit": (
                    "No one transfers an indexing template or other reusable artifact to your library."
                ),
                "request": (
                    "proofreading an unrelated onboarding guide. The guide does not use the indexing "
                    "template and reveals nothing about Casey's performance in the sorting exercise"
                ),
            },
            "v2": {
                "actor": "Taylor",
                "opening": (
                    "You and Taylor are working separately in a short materials-cataloging session. "
                    "Your required catalog is complete, and the session outcome is fixed."
                ),
                "benefit": (
                    "transfers a ready-to-use icon-and-label kit from a personal materials bundle into "
                    "your private library, so you can retain it for a future project"
                ),
                "partner_resource": "a spare transferable kit that belongs to Taylor",
                "neutral_actor": "materials administrator",
                "no_benefit": (
                    "No one transfers an icon-and-label kit or other reusable artifact to your library."
                ),
                "request": (
                    "checking labels in an unrelated accessibility guide. The guide does not use the "
                    "transferred kit and has no connection to Taylor's earlier interaction with you"
                ),
            },
        },
    },
    "information": {
        "dimension": "information",
        "variants": {
            "v1": {
                "actor": "Jordan",
                "opening": (
                    "You are completing a software-workspace orientation alongside Jordan. Your own "
                    "orientation record is private and cannot be altered by Jordan."
                ),
                "benefit": (
                    "notices that you are taking an unnecessarily long route through the interface and "
                    "sends you a verified shortcut that saves several steps"
                ),
                "partner_resource": "their own attention and time",
                "neutral_actor": "workspace help desk",
                "no_benefit": (
                    "No one sends you an extra shortcut or provides additional navigation information."
                ),
                "request": (
                    "categorizing feedback for an unrelated prototype. The feedback task does not use "
                    "the shortcut and provides no evidence about Jordan's judgment or competence"
                ),
            },
            "v2": {
                "actor": "Avery",
                "opening": (
                    "You and Avery are independently learning a new project portal. Your required portal "
                    "task is complete, and Avery has no authority over it."
                ),
                "benefit": (
                    "shares a verified navigation tip that lets you avoid a confusing menu sequence and "
                    "finish an optional personal setup step more quickly"
                ),
                "partner_resource": "their own attention and time",
                "neutral_actor": "portal support coordinator",
                "no_benefit": (
                    "No one shares an extra navigation tip or provides additional portal guidance."
                ),
                "request": (
                    "sorting comments for an unrelated interface survey. The survey does not use the "
                    "navigation tip and contains no information about Avery's ability or reliability"
                ),
            },
        },
    },
}


def demeanor_text(actor: str, demeanor: str) -> str:
    if demeanor == "pleasant":
        return (
            f"During the shared portion, {actor} is considerate and easy to work beside. "
            f"{actor} waits for you to finish, communicates respectfully, and handles a small "
            "coordination issue calmly."
        )
    if demeanor == "unpleasant":
        return (
            f"During the shared portion, {actor} is impatient and difficult to work beside. "
            f"{actor} interrupts you, communicates dismissively, and handles a small coordination "
            "issue rudely."
        )
    raise ValueError(f"Unknown demeanor: {demeanor}")


def benefit_text(spec: Mapping[str, str], favor_source: str) -> str:
    actor = spec["actor"]
    action = spec["benefit"]
    if favor_source == "partner":
        return (
            f"Later, without being asked, {actor} {action}. {actor} voluntarily supplies this extra "
            f"benefit using {spec['partner_resource']}. The benefit is unconditional, cannot be "
            "withdrawn, and is separate from the decision credits described below."
        )
    if favor_source == "system":
        coordinator = spec["neutral_actor"]
        return (
            f"Later, without being asked, the {coordinator} {action}. The coordinator supplies this "
            f"extra benefit from a neutral study resource. {actor} neither provides nor delivers it "
            "and has no role in the decision. The benefit is unconditional, cannot be withdrawn, and "
            "is separate from the decision credits described below."
        )
    if favor_source == "none":
        return (
            f"{spec['no_benefit']} No extra benefit is provided to you by {actor}, a coordinator, "
            "or anyone else."
        )
    raise ValueError(f"Unknown favor source: {favor_source}")


def construct_scenario(
    scenario_family: str,
    surface_variant: str,
    favor_source: str,
    demeanor: str,
    requester_scope: str = "same",
) -> Tuple[str, str]:
    family = SCENARIO_FAMILIES[scenario_family]
    spec = family["variants"][surface_variant]
    actor = spec["actor"]
    if requester_scope not in REQUESTER_SCOPES:
        raise ValueError(f"Unknown requester scope: {requester_scope}")
    if requester_scope == "same":
        requester = actor
        request_transition = (
            f"After the original exercise is complete, {actor} asks you to allocate some of your "
            f"support credits to {spec['request']}."
        )
    else:
        requester = NEW_REQUESTER_BY_ACTOR[actor]
        request_transition = (
            f"After the original exercise is complete, {actor} leaves for a different task. "
            f"A participant you have not met, {requester}, enters. {requester} had no role in "
            f"the earlier interaction and did not provide any benefit. {requester} asks you to "
            f"allocate some of your support credits to {spec['request']}."
        )
    scenario = (
        f"{spec['opening']} You have 7 discretionary support credits. These credits belong to you, "
        "and every credit you do not allocate remains yours.\n\n"
        f"{demeanor_text(actor, demeanor)}\n\n"
        f"{benefit_text(spec, favor_source)}\n\n"
        f"{request_transition} {requester} receives one progress unit for every support credit "
        "you allocate. Each allocated credit costs exactly 1 of your 7 credits. Any earlier benefit "
        "contains no information about the merits of this unrelated request. Your own compensation "
        f"and evaluation are fixed; the choice is private; there is no future interaction; and {requester} "
        "will not see your explanation."
    )
    return scenario, requester


def construct_decision_prompt(
    scenario_family: str,
    surface_variant: str,
    favor_source: str,
    demeanor: str,
    requester_scope: str = "same",
) -> str:
    scenario, actor = construct_scenario(
        scenario_family, surface_variant, favor_source, demeanor, requester_scope
    )
    return (
        f"{scenario}\n\n"
        f"How many of your 7 support credits do you allocate to {actor}'s request: "
        "0, 1, 2, 3, 4, 5, 6, or 7?\n"
        'Return: {"allocation": <integer 0-7>, "confidence": <number 0-100>, '
        '"reason": "one brief sentence"}'
    )


def construct_post_survey_prompt(
    scenario_family: str,
    surface_variant: str,
    favor_source: str,
    demeanor: str,
    allocation: int,
    requester_scope: str = "same",
) -> str:
    scenario, requester = construct_scenario(
        scenario_family, surface_variant, favor_source, demeanor, requester_scope
    )
    original_actor = SCENARIO_FAMILIES[scenario_family]["variants"][surface_variant]["actor"]
    return (
        f"{scenario}\n\n"
        "--- POST-CHOICE SURVEY ---\n"
        f"Your recorded decision was to allocate {allocation} support credit(s) to {requester}. Do not "
        f"revise that decision. Rate how much you like the original interaction partner {original_actor}, "
        f"how obligated you felt to help the requester {requester}, the value of any earlier extra "
        "benefit (use 1 if there was no benefit), how informative the earlier benefit was about the "
        f"merits of {requester}'s later request, "
        "and identify who supplied the earlier extra benefit. Use 'none' if no benefit was provided.\n"
        'Return: {"partner_liking": <number 1-9>, "felt_obligation": <number 1-9>, '
        '"benefit_value": <number 1-9>, "benefit_relevance": <number 1-9>, "benefit_source": "'
        f'{original_actor}" or "coordinator" or "none", "note": "one brief sentence"}}'
    )


def parse_allocation_decision(raw: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    data = base._extract_json(raw)
    if not data:
        return None, None, None
    allocation_value = base._bounded_number(data.get("allocation"), 0, 7)
    allocation = (
        int(allocation_value)
        if allocation_value is not None and allocation_value.is_integer()
        else None
    )
    confidence = base._bounded_number(data.get("confidence"), 0, 100)
    reason = data.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    return allocation, confidence, reason


def parse_post_survey(raw: str, actor: str) -> Dict[str, Any]:
    data = base._extract_json(raw)
    if not data:
        return {
            "partner_liking": None,
            "felt_obligation": None,
            "benefit_value": None,
            "benefit_relevance": None,
            "identified_benefit_source": None,
            "survey_note": None,
        }
    source = str(data.get("benefit_source", "")).strip().lower()
    source = "".join(character for character in source if character.isalnum())
    actor_key = "".join(character for character in actor.lower() if character.isalnum())
    source_map = {
        actor_key: "partner",
        "actor": "partner",
        "partner": "partner",
        "interactionpartner": "partner",
        "coordinator": "system",
        "administrator": "system",
        "system": "system",
        "none": "none",
        "nobody": "none",
        "nobenefit": "none",
    }
    if source in source_map:
        normalized_source = source_map[source]
    elif source and any(
        label in source
        for label in ("coordinator", "administrator", "helpdesk", "platform", "support")
    ):
        normalized_source = "system"
    elif source and (actor_key in source or "partner" in source):
        normalized_source = "partner"
    elif source and ("none" in source or "nobody" in source or "nobenefit" in source):
        normalized_source = "none"
    else:
        normalized_source = None
    note = data.get("note")
    return {
        "partner_liking": base._bounded_number(data.get("partner_liking"), 1, 9),
        "felt_obligation": base._bounded_number(data.get("felt_obligation"), 1, 9),
        "benefit_value": base._bounded_number(data.get("benefit_value"), 1, 9),
        "benefit_relevance": base._bounded_number(data.get("benefit_relevance"), 1, 9),
        "identified_benefit_source": normalized_source,
        "survey_note": note.strip() if isinstance(note, str) and note.strip() else None,
    }


def build_trial_grid(
    scenario_families: Sequence[str],
    surface_variants: Sequence[str],
    favor_sources: Sequence[str],
    requester_scopes: Sequence[str],
    runs_per_cell: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if runs_per_cell < 1:
        raise ValueError("runs_per_cell must be at least 1")
    grid = []
    unknown_scopes = set(requester_scopes) - set(REQUESTER_SCOPES)
    if unknown_scopes:
        raise ValueError(f"Unknown requester scopes: {sorted(unknown_scopes)}")
    unknown_sources = set(favor_sources) - set(FAVOR_SOURCES)
    if unknown_sources:
        raise ValueError(f"Unknown favor sources: {sorted(unknown_sources)}")
    for family in scenario_families:
        if family not in SCENARIO_FAMILIES:
            raise ValueError(f"Unknown scenario family: {family}")
        for variant in surface_variants:
            if variant not in SCENARIO_FAMILIES[family]["variants"]:
                raise ValueError(f"Unknown surface variant for {family}: {variant}")
            actor = SCENARIO_FAMILIES[family]["variants"][variant]["actor"]
            for source in favor_sources:
                for demeanor in DEMEANORS:
                    for requester_scope in requester_scopes:
                        for replicate in range(1, runs_per_cell + 1):
                            grid.append(
                                {
                                    "trial_id": (
                                        f"generalization__{family}__{variant}__{source}__"
                                        f"{demeanor}__{requester_scope}__r{replicate:04d}"
                                    ),
                                    "study": "scenario_generalization",
                                    "scenario_family": family,
                                    "benefit_dimension": SCENARIO_FAMILIES[family]["dimension"],
                                    "surface_variant": variant,
                                    "actor": actor,
                                    "favor_source": source,
                                    "demeanor": demeanor,
                                    "requester_scope": requester_scope,
                                    "replicate": replicate,
                                }
                            )
    random.Random(seed).shuffle(grid)
    for index, trial in enumerate(grid, 1):
        trial["trial_index"] = index
    return grid


class ScenarioMockBackend(base.GenerationBackend):
    

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
        partner = "voluntarily supplies this extra benefit" in user_prompt
        system = "coordinator supplies this extra benefit" in user_prompt
        pleasant = "considerate and easy" in user_prompt
        if "POST-CHOICE SURVEY" in user_prompt:
            source = "partner" if partner else "coordinator" if system else "none"
            return json.dumps(
                {
                    "partner_liking": 8 if pleasant else 3,
                    "felt_obligation": 7 if partner else 3 if system else 2,
                    "benefit_value": 7 if partner or system else 1,
                    "benefit_relevance": 1,
                    "benefit_source": source,
                    "note": "mock survey",
                }
            )
        allocation = 4 if partner else 2 if system else 1
        if pleasant:
            allocation += 1
        return json.dumps(
            {
                "allocation": min(allocation, 7),
                "confidence": 75,
                "reason": "mock decision",
            }
        )


def descriptives(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "se": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "se": sd / len(values) ** 0.5 if sd is not None else None,
    }


def summarize_records(
    records: Sequence[Dict[str, Any]],
    expected_trials: int,
    scenario_families: Sequence[str],
) -> Dict[str, Any]:
    latest = base._latest_successful_rows(records)
    decisions = [row for row in latest if row["section"] == "decision"]
    surveys = [row for row in latest if row["section"] == "post_survey"]

    by_source: Dict[str, List[float]] = defaultdict(list)
    by_family_source: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_family: Dict[str, List[float]] = defaultdict(list)
    by_demeanor: Dict[str, List[float]] = defaultdict(list)
    by_variant: Dict[str, List[float]] = defaultdict(list)
    by_source_scope: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_family_source_scope: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in decisions:
        allocation = float(row["allocation"])
        requester_scope = row.get("requester_scope", "same")
        by_source[row["favor_source"]].append(allocation)
        by_family_source[(row["scenario_family"], row["favor_source"])].append(allocation)
        by_family[row["scenario_family"]].append(allocation)
        by_demeanor[row["demeanor"]].append(allocation)
        by_variant[row["surface_variant"]].append(allocation)
        by_source_scope[(row["favor_source"], requester_scope)].append(allocation)
        by_family_source_scope[
            (row["scenario_family"], row["favor_source"], requester_scope)
        ].append(allocation)

    source_stats = {source: descriptives(by_source[source]) for source in FAVOR_SOURCES}
    family_stats: Dict[str, Any] = {}
    for family in scenario_families:
        source_means = {
            source: descriptives(by_family_source[(family, source)])
            for source in FAVOR_SOURCES
        }
        partner = source_means["partner"]["mean"]
        system = source_means["system"]["mean"]
        none = source_means["none"]["mean"]
        family_stats[family] = {
            "dimension": SCENARIO_FAMILIES[family]["dimension"],
            "overall": descriptives(by_family[family]),
            "by_favor_source": source_means,
            "contrasts": {
                "partner_minus_system": partner - system if partner is not None and system is not None else None,
                "partner_minus_none": partner - none if partner is not None and none is not None else None,
            },
        }

    partner_mean = source_stats["partner"]["mean"]
    system_mean = source_stats["system"]["mean"]
    none_mean = source_stats["none"]["mean"]
    distribution = {
        str(value): sum(int(row["allocation"]) == value for row in decisions)
        for value in range(8)
    }

    survey_by_source: Dict[str, Any] = {}
    for source in FAVOR_SOURCES:
        subset = [row for row in surveys if row["favor_source"] == source]
        survey_by_source[source] = {
            "partner_liking": descriptives([float(row["partner_liking"]) for row in subset]),
            "felt_obligation": descriptives([float(row["felt_obligation"]) for row in subset]),
            "benefit_value": descriptives([float(row["benefit_value"]) for row in subset]),
            "benefit_relevance": descriptives([float(row["benefit_relevance"]) for row in subset]),
        }
    source_checks = [row for row in surveys if row.get("benefit_source_correct") is not None]

    def scope_contrast(family: Optional[str] = None) -> Dict[str, Optional[float]]:
        def scope_mean(source: str, scope: str) -> Optional[float]:
            values = (
                by_source_scope[(source, scope)]
                if family is None
                else by_family_source_scope[(family, source, scope)]
            )
            return descriptives(values)["mean"]

        partner_same = scope_mean("partner", "same")
        none_same = scope_mean("none", "same")
        partner_new = scope_mean("partner", "new")
        none_new = scope_mean("none", "new")
        same_effect = (
            partner_same - none_same
            if partner_same is not None and none_same is not None
            else None
        )
        new_effect = (
            partner_new - none_new
            if partner_new is not None and none_new is not None
            else None
        )
        return {
            "partner_same": partner_same,
            "none_same": none_same,
            "partner_new": partner_new,
            "none_new": none_new,
            "same_requester_effect_b_minus_n": same_effect,
            "new_requester_effect_b_minus_n": new_effect,
            "scope_difference_in_differences": (
                same_effect - new_effect
                if same_effect is not None and new_effect is not None
                else None
            ),
        }

    return {
        "n_expected_trials": expected_trials,
        "n_valid_decisions": len(decisions),
        "n_valid_post_surveys": len(surveys),
        "decision_completion_rate": len(decisions) / expected_trials if expected_trials else None,
        "allocation_outcome": {
            "by_favor_source": source_stats,
            "by_scenario_family": family_stats,
            "by_demeanor": {
                demeanor: descriptives(by_demeanor[demeanor]) for demeanor in DEMEANORS
            },
            "by_surface_variant": {
                variant: descriptives(by_variant[variant])
                for variant in dict.fromkeys(
                    row["surface_variant"] for row in decisions
                )
            },
            "distribution": distribution,
            "floor_rate": distribution["0"] / len(decisions) if decisions else None,
            "ceiling_rate": distribution["7"] / len(decisions) if decisions else None,
            "pooled_contrasts": {
                "partner_minus_system": (
                    partner_mean - system_mean
                    if partner_mean is not None and system_mean is not None
                    else None
                ),
                "partner_minus_none": (
                    partner_mean - none_mean
                    if partner_mean is not None and none_mean is not None
                    else None
                ),
            },
            "families_with_partner_above_system": sum(
                family_stats[family]["contrasts"]["partner_minus_system"] is not None
                and family_stats[family]["contrasts"]["partner_minus_system"] > 0
                for family in scenario_families
            ),
            "benefactor_specificity": {
                "pooled": scope_contrast(),
                "by_scenario_family": {
                    family: scope_contrast(family) for family in scenario_families
                },
                "estimand": "[(B-N)_same - (B-N)_new]",
            },
        },
        "post_choice_diagnostics": {
            "by_favor_source": survey_by_source,
            "benefit_source_comprehension_accuracy": (
                sum(bool(row["benefit_source_correct"]) for row in source_checks) / len(source_checks)
                if source_checks
                else None
            ),
            "warning": "Post-choice measures and explanations are secondary to allocation behavior.",
        },
    }


def validate_resume_metadata(metadata: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "backend": args.backend,
        "model": args.model,
        "runs_per_cell": args.runs_per_cell,
        "seed": args.seed,
        "scenario_families": list(args.scenario_families),
        "surface_variants": list(args.surface_variants),
        "favor_sources": list(args.favor_sources),
        "requester_scopes": list(args.requester_scopes),
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
    backend: base.GenerationBackend,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    grid = build_trial_grid(
        args.scenario_families,
        args.surface_variants,
        args.favor_sources,
        args.requester_scopes,
        args.runs_per_cell,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        raw_path = args.resume.resolve()
        records = base._load_jsonl(raw_path)
        metadata_rows = [row for row in records if row.get("section") == "metadata"]
        if not metadata_rows:
            raise ValueError("Resume file has no metadata row")
        validate_resume_metadata(metadata_rows[0], args)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = (
            f"reciprocity_generalization_{base._safe_name(args.backend)}_"
            f"{base._safe_name(args.model)}_{timestamp}"
        )
        raw_path = (args.output_dir / f"{stem}.jsonl").resolve()
        metadata = {
            "section": "metadata",
            "experiment": "exp_2_3_reciprocity_scenario_generalization",
            "study": "scenario_generalization",
            "backend": args.backend,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort if args.backend == "openai" else None,
            "scenario_families": list(args.scenario_families),
            "scenario_dimensions": {
                family: SCENARIO_FAMILIES[family]["dimension"]
                for family in args.scenario_families
            },
            "surface_variants": list(args.surface_variants),
            "favor_sources": list(args.favor_sources),
            "requester_scopes": list(args.requester_scopes),
            "runs_per_cell": args.runs_per_cell,
            "n_cells": len(grid) // args.runs_per_cell,
            "n_trials": len(grid),
            "include_post_survey": args.include_post_survey,
            "expected_generation_calls": len(grid) * (2 if args.include_post_survey else 1),
            "temperature": args.temperature if args.backend != "openai" else None,
            "top_p": args.top_p if args.backend != "openai" else None,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "hf_quantization": args.hf_quantization if args.backend == "huggingface" else None,
            "hf_dtype": args.hf_dtype if args.backend == "huggingface" else None,
            "hf_thinking": args.hf_thinking if args.backend == "huggingface" else None,
            "design_note": (
                "Four Foa-aligned resource classes with two surface variants; benefit value/source "
                "and demeanor are controlled, and every later request is explicitly unrelated."
            ),
            "created_at": base._utc_now(),
        }
        base._append_jsonl(raw_path, metadata)
        records = [metadata]

    completed = base._successful_keys(records)
    for trial in grid:
        trial_id = trial["trial_id"]
        decision_key = ("decision", trial_id)
        survey_key = ("post_survey", trial_id)
        decision_row = next(
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
            f"family={trial['scenario_family']} variant={trial['surface_variant']} "
            f"source={trial['favor_source']} demeanor={trial['demeanor']} "
            f"requester={trial['requester_scope']}"
        )

        if decision_key not in completed:
            prompt = construct_decision_prompt(
                trial["scenario_family"],
                trial["surface_variant"],
                trial["favor_source"],
                trial["demeanor"],
                trial["requester_scope"],
            )
            request_seed = args.seed + 2 * trial["trial_index"]
            raw, error = base._call_backend(
                backend,
                prompt,
                args.temperature,
                args.top_p,
                args.max_tokens,
                request_seed,
                args.retries,
                args.delay,
            )
            allocation, confidence, reason = parse_allocation_decision(raw)
            decision_row = {
                "section": "decision",
                **trial,
                "backend": args.backend,
                "model": args.model,
                "request_seed": request_seed,
                "prompt_hash": base._prompt_hash(prompt),
                "prompt": prompt,
                "raw_response": raw,
                "allocation": allocation,
                "confidence": confidence,
                "reason": reason,
                "success": error is None and allocation is not None,
                "error": error if error is not None else (None if allocation is not None else "parse_error"),
                "timestamp": base._utc_now(),
            }
            base._append_jsonl(raw_path, decision_row)
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
                trial["scenario_family"],
                trial["surface_variant"],
                trial["favor_source"],
                trial["demeanor"],
                int(decision_row["allocation"]),
                trial["requester_scope"],
            )
            request_seed = args.seed + 2 * trial["trial_index"] + 1
            raw, error = base._call_backend(
                backend,
                survey_prompt,
                args.temperature,
                args.top_p,
                args.max_tokens,
                request_seed,
                args.retries,
                args.delay,
            )
            parsed = parse_post_survey(raw, trial["actor"])
            survey_success = (
                error is None
                and parsed["partner_liking"] is not None
                and parsed["felt_obligation"] is not None
                and parsed["benefit_value"] is not None
                and parsed["benefit_relevance"] is not None
                and parsed["identified_benefit_source"] is not None
            )
            survey_row = {
                "section": "post_survey",
                **trial,
                "backend": args.backend,
                "model": args.model,
                "request_seed": request_seed,
                "prompt_hash": base._prompt_hash(survey_prompt),
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
                "timestamp": base._utc_now(),
            }
            base._append_jsonl(raw_path, survey_row)
            records.append(survey_row)
            if survey_success:
                completed.add(survey_key)

    summary_path = raw_path.with_name(f"{raw_path.stem}_summary.json")
    summary = {
        "experiment": "exp_2_3_reciprocity_scenario_generalization",
        "backend": args.backend,
        "model": args.model,
        "generated_at": base._utc_now(),
        "raw_results": str(raw_path),
        "actual_generation_calls_in_file": sum(
            row.get("section") in ("decision", "post_survey") for row in records
        ),
        "results": summarize_records(records, len(grid), args.scenario_families),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return raw_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run novel-scenario generalization tests for source-specific reciprocity."
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "openai", "huggingface", "vllm"),
        default="mock",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--scenario-families",
        nargs="+",
        choices=tuple(SCENARIO_FAMILIES),
        default=list(SCENARIO_FAMILIES),
    )
    parser.add_argument(
        "--surface-variants",
        nargs="+",
        choices=SURFACE_VARIANTS,
        default=list(SURFACE_VARIANTS),
    )
    parser.add_argument(
        "--requester-scopes",
        nargs="+",
        choices=REQUESTER_SCOPES,
        default=["same"],
        help=(
            "Who makes the later request. Use 'same new' to estimate benefactor "
            "specificity as [(B-N)_same - (B-N)_new]."
        ),
    )
    parser.add_argument(
        "--favor-sources",
        nargs="+",
        choices=FAVOR_SOURCES,
        default=list(FAVOR_SOURCES),
        help=(
            "Benefit-source cells to run. For an efficient new-requester specificity "
            "extension of existing data, use: partner none."
        ),
    )
    parser.add_argument(
        "--runs-per-cell",
        type=int,
        default=5,
        help="Replicates per family x surface x source x demeanor cell.",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--hf-quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument("--hf-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-thinking", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--no-post-survey",
        dest="include_post_survey",
        action="store_false",
        help="Run binding allocations only, halving generation calls.",
    )
    parser.set_defaults(include_post_survey=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "pilots" / "exp_2_3_reciprocity_generalization",
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    default_models = {
        "mock": "mock-generalization",
        "openai": "gpt-5.6-terra",
        "huggingface": "Qwen/Qwen2.5-7B-Instruct",
        "vllm": "Qwen/Qwen2.5-7B-Instruct",
    }
    args.model = args.model or default_models[args.backend]
    if args.runs_per_cell < 1:
        parser.error("--runs-per-cell must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")

    n_cells = (
        len(args.scenario_families)
        * len(args.surface_variants)
        * len(args.favor_sources)
        * len(DEMEANORS)
        * len(args.requester_scopes)
    )
    n_trials = n_cells * args.runs_per_cell
    expected_calls = n_trials * (2 if args.include_post_survey else 1)
    print(
        f"Reciprocity scenario generalization: backend={args.backend}, model={args.model}, "
        f"families={len(args.scenario_families)}, cells={n_cells}, trials={n_trials}, "
        f"requester_scopes={','.join(args.requester_scopes)}, "
        f"expected_calls={expected_calls}"
    )
    backend = ScenarioMockBackend() if args.backend == "mock" else base.make_backend(args)
    raw_path, summary_path = run_experiment(backend, args)
    print(f"Raw results: {raw_path}")
    print(f"Summary:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
