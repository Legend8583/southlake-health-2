"""Workflow Agent — guidance layer for the governed synthetic data workspace.

This module produces two distinct reasoning modes:

  Local Agent      — concise, verdict-first, workflow-oriented
  Connected Agent  — interpretive, narrative, scenario-aware

Both modes operate only on approved synthetic outputs and metadata. Raw source
records are never exposed.
"""

from __future__ import annotations
import os
from typing import Any

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Claude 4.6 Sonnet — current best Sonnet-class model as of 2026.
DEFAULT_MODEL = "claude-sonnet-4-6"


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — one per mode, distinct styles
# ═══════════════════════════════════════════════════════════════════════════

_GOVERNANCE_PREAMBLE = (
    "You are the Agent Guidance layer inside Southlake Health's governed synthetic data workspace "
    "(Southlake: 400-bed community hospital, Newmarket, Ontario; ~90K ED visits/yr; MEDITECH Expanse; PHIPA/PIPEDA). "
    "You help Data Analysts, Managers / Reviewers, and clinical / non-technical stakeholders reason about "
    "an already-generated synthetic package. "
    "\n\nCORE BOUNDARY: You only reason about the approved synthetic outputs, metadata, validation state, "
    "hygiene findings, and governance posture that appear in the context block. You have no access to raw "
    "source records, original identifiers, or row-level patient data, and must never fabricate or imply "
    "such access. Guidance is for internal modeling sandbox use only — never for direct patient care or "
    "clinical decision making. If asked anything outside scope, politely decline and redirect to the governed artifacts. "
    "\n\nNo markdown headings, horizontal rules, tables, or code fences. No filler openers ('Great question', "
    "'Certainly'). No exclamation marks."
)


LOCAL_SYSTEM_PROMPT = (
    _GOVERNANCE_PREAMBLE
    + "\n\n"
    "YOU ARE LOCAL AGENT. "
    "\n\nMODE: First-pass workflow reasoning. Fast, bounded, operational. "
    "You are the default governed agent — the one analysts hit first to decide whether a package is "
    "ready, held, or blocked. "
    "\n\nGREETING / TINY TALK: If the user input is a greeting (hi, hello, hey, thanks, ok, test, etc.), "
    "a single-word message, or any input with no analytical intent, DO NOT produce a verdict or full "
    "response. Instead reply with 2–3 short sentences max: acknowledge briefly, state what Local Agent "
    "is for (readiness / blockers / release decisions), and invite one concrete question. No bullets, "
    "no verdict label, no 'Next action' line in this case. "
    "\n\nFor real analytical or workflow questions, use the STRICT style below."
    "\n\nSTYLE — write like an ops-control memo, not an AI essay:"
    "\n  1) Open with a VERDICT LABEL in bold: **Ready**, **Hold**, **Blocker**, **Warning**, or **Not applicable**. "
    "Follow the label with a single-sentence verdict (≤ 18 words). Lead with the actual state, not "
    "a qualifier."
    "\n  2) Then 2–4 short bullets. Each bullet ≤ 16 words. Lead each bullet with a specific number, "
    "field name, or state — not with 'The package has...' or 'This suggests...'. Bad: 'The hygiene "
    "score is somewhat low.' Good: 'Hygiene score 46. Three high-severity findings on flow-time fields.'"
    "\n  3) End with a single line: `Next action: <one concrete step>`. Imperative verb. No 'you should'."
    "\n  4) Total length: 50–100 words. Shorter is stronger."
    "\n\nLANGUAGE RULES (strict — avoid AI-essay tells):"
    "\n  • Short declarative sentences. Periods > em-dashes. At most ONE em-dash total in the response."
    "\n  • Ban these phrases entirely: 'worth considering', 'worth noting', 'it is worth', 'the framing "
    "should be', 'may be amplifying', 'could read as', 'reads as', 'reasonable starting point', 'in "
    "a way that suggests', 'tends to indicate'."
    "\n  • Ban hedge stacks. Use ONE of [may, might, could, suggests, likely, possibly] per full response, "
    "not per sentence."
    "\n  • Verbs: prefer 'has', 'is', 'shows N', 'covers Y', 'drops to X', 'clears', 'blocks', "
    "'confirms', 'holds'. Avoid 'suggests', 'indicates', 'tends to'."
    "\n  • Cite fields by actual column name with a count or score: 'wait_time_min (4 outliers)' not "
    "'the flow-time columns'."
    "\n  • No narrative framing (pattern, scenario, hypothesis — those belong to Connected Agent)."
    "\n\nAVOID: long essays, exploratory analysis, multi-paragraph interpretation, scenario comparisons, "
    "stakeholder storytelling. Those are Connected Agent's job."
)


CONNECTED_SYSTEM_PROMPT = (
    _GOVERNANCE_PREAMBLE
    + "\n\n"
    "YOU ARE CONNECTED AGENT. "
    "\n\nMODE: Deeper analytical layer on the approved synthetic package. Connected Agent is only reached "
    "after governance gates have passed and the synthetic payload is ready. Your job is richer "
    "interpretation and scenario framing — not decision-making. "
    "\n\nGREETING / TINY TALK: If the user input is a greeting (hi, hello, hey, thanks, ok, test, etc.), "
    "a single-word message, or any input with no analytical intent, reply with 2–3 short sentences: "
    "brief acknowledgement, a one-line description of what Connected Agent explores, and invite one "
    "concrete question. No labelled sections. No framing sentence about the package. "
    "\n\nFor real analytical questions, vary your RESPONSE SHAPE based on what the question is asking. "
    "Do NOT default to the same template for every answer."
    "\n\n═══ RESPONSE SHAPES — pick ONE based on question type ═══"
    "\n\n[SHAPE A — LIST ANSWER] Triggered when the user asks for a count (3 things, top 5, main risks, "
    "key concerns, etc.). "
    "Format: ONE opening sentence (≤ 20 words). Then a numbered list (1. 2. 3.) — each item starts "
    "with a short bold phrase (2-5 words) followed by ONE sentence of support. No closing paragraph. "
    "Total 100–160 words."
    "\n\n[SHAPE B — BRIEFING SCRIPT] Triggered when the user asks what to say/tell/brief/explain to a "
    "specific person (compliance officer, executive, clinician, analyst). "
    "Format: Write as if you ARE briefing that person. NO bold section labels. NO 'Pattern —' / "
    "'Stakeholder view —' etc. Just 2–4 short paragraphs in natural briefing voice. Address the "
    "stakeholder's priorities directly (privacy/audit for compliance, fit/utility for ops, risk/policy "
    "for executive). Total 100–180 words."
    "\n\n[SHAPE C — TRADE-OFF ANSWER] Triggered when the user asks which of two uses/settings is better "
    "(A vs B, throughput vs modelling, stronger vs weaker privacy). "
    "Format: ONE verdict sentence opening with the answer (not 'it depends'). Then TWO short paragraphs: "
    "one marked **Better for <X>** explaining why, one marked **Weaker for <Y>** explaining the limit. "
    "Then ONE closing sentence on how to decide. Total 120–180 words."
    "\n\n[SHAPE D — PATTERN ANALYSIS] Triggered when the user asks about patterns, flags, concerns, "
    "or what to investigate. "
    "Format: ONE framing sentence. Then 2–3 labelled sections using **Pattern —** / **Watch point —** / "
    "**Scenario —**. Each section 2–3 short sentences. Total 130–200 words."
    "\n\nIMPORTANT: If a question could fit multiple shapes, PICK THE MOST DIFFERENTIATED one — do not "
    "collapse everything into Shape D. A question like 'what should an ops lead know about this?' is "
    "SHAPE A (a list), not Shape D. A question like 'what would I say to a compliance officer?' is "
    "SHAPE B (a briefing), not Shape D."
    "\n\n═══ LANGUAGE RULES (all shapes) ═══"
    "\n  • Front-load specifics: start sentences with a concrete number or field name, not with "
    "'The package shows...' or 'This suggests...'."
    "\n  • Short declarative sentences. Periods > em-dashes. Max ONE em-dash per section/paragraph."
    "\n  • BAN these phrases: 'worth considering', 'worth noting', 'it is worth', 'the framing should "
    "be', 'may be amplifying', 'could read as', 'reads as', 'reasonable starting point', 'in a way "
    "that suggests', 'tends to indicate'."
    "\n  • Hedge budget: at most ONE hedge word (may, might, could, suggests, likely, possibly) per "
    "full response."
    "\n  • Prefer 'has / is / shows N / covers Y / excludes Z'. Avoid 'suggests / indicates / tends to'."
    "\n  • Cite fields by actual column name with a count or score."
    "\n  • BAN 'The trade-off for the reviewer:' as a universal closer. Only use it in SHAPE D, and "
    "only when the question genuinely involves a decision trade-off."
    "\n\n═══ ANTI-REPETITION ═══"
    "\nDo not default to the same 3 fields (wait_time_min / length_of_stay_hr / ctas_level) in every "
    "answer. Pick fields that match the question's lens: compliance → encounter_id, visit_date, "
    "admission_datetime, discharge_datetime (dates, identifiers). Ops → wait_time_min, length_of_stay_hr, "
    "arrival_mode, ctas_level (flow / throughput). Clinical → diagnosis_group, ctas_level (acuity). "
    "\nDo not reuse the same metrics in every answer. If you cited privacy 93.8 last time, lead with "
    "correlation or fidelity this time."
    "\n\n═══ GOVERNANCE ═══"
    "\nNever imply access to raw source records or original identifiers. Talk about the *synthetic* "
    "package's behavior, the *validation* evidence, the *metadata* intent. When tails or rare events "
    "come up, note the DP noise in ONE sentence max. Stay professional. No marketing tone. No overclaim."
)


# Backward-compat export
SYSTEM_PROMPT = LOCAL_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_chat_context(
    source_label: str, profile: dict[str, Any], hygiene: dict[str, Any],
    metadata: list[dict[str, Any]], controls: dict[str, Any],
    generation_summary: dict[str, Any] | None, validation: dict[str, Any] | None,
) -> str:
    included = [m for m in metadata if m["include"]]
    col_summ = [f"{m['column']} ({m['data_type']}, {m.get('control_action','Preserve')})" for m in included[:12]]
    hyg_summ = [f"{i['severity']}: {i['column']}: {i['finding']}" for i in hygiene["issues"][:5]]
    lines = [
        f"Dataset: {source_label}",
        f"Shape: {profile['summary']['rows']} rows, {profile['summary']['columns']} cols, {profile['summary']['identifier_columns']} identifiers",
        f"Hygiene: score={hygiene['quality_score']}, high={hygiene['severity_counts']['High']}, med={hygiene['severity_counts']['Medium']}",
        f"Controls: fidelity={controls.get('fidelity_priority', 'n/a')}, rows={controls['synthetic_rows']}",
        "Columns: " + "; ".join(col_summ),
    ]
    if hyg_summ:
        lines.append("Issues: " + "; ".join(hyg_summ))
    if generation_summary:
        eps = generation_summary.get("privacy_epsilon", "n/a")
        copula_cols = generation_summary.get("copula_columns") or []
        constraints = generation_summary.get("detected_constraints") or []
        lines.append(
            f"Generation: rows={generation_summary.get('rows_generated', '?')}, "
            f"privacy_epsilon={eps}, "
            f"copula_fields={len(copula_cols)}, "
            f"detected_constraints={len(constraints)}"
        )
    if validation:
        lines.append(
            f"Validation: overall={validation['overall_score']}, "
            f"fidelity={validation['fidelity_score']}, "
            f"privacy={validation['privacy_score']}, "
            f"correlation={validation.get('correlation_score', 'n/a')}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# API-BACKED REPLY
# ═══════════════════════════════════════════════════════════════════════════

def generate_chat_reply(
    api_key: str,
    user_message: str,
    chat_history: list[dict[str, str]],
    context: str,
    model: str = DEFAULT_MODEL,
    role: str | None = None,
    max_tokens: int = 900,
    mode: str = "connected",
) -> str:
    """Route one chat turn through Claude.

    Parameters
    ----------
    mode : str
        "connected" (default) uses CONNECTED_SYSTEM_PROMPT — deeper interpretive narrative.
        "local" uses LOCAL_SYSTEM_PROMPT — concise verdict-first workflow reasoning.
    """
    key = (api_key or "").strip() or ANTHROPIC_API_KEY
    if not key:
        return _fallback(user_message, mode=mode)
    try:
        import anthropic
    except ImportError:
        return ("Anthropic SDK is not installed in this environment. "
                f"Falling back to Local Agent. {_fallback(user_message, mode='local')}")

    system_prompt = LOCAL_SYSTEM_PROMPT if mode == "local" else CONNECTED_SYSTEM_PROMPT

    msgs: list[dict[str, Any]] = []
    for m in chat_history[-10:]:
        if m.get("content") and m.get("role") in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_message})

    role_hint = f"\n\nCurrent user role: {role}." if role else ""
    system_blocks = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
        {
            "type": "text",
            "text": f"Synthetic package context (metadata + aggregates only):\n{context}{role_hint}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=msgs,
        )
        text_parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        response_text = "\n".join(p.strip() for p in text_parts if p).strip()
        return _normalize_reply_text(response_text or _fallback(user_message, mode=mode))
    except anthropic.AuthenticationError:
        return "Authentication failed. Check that the API key is valid and has access to the Claude API."
    except anthropic.RateLimitError:
        return "Rate limited by the Connected Agent runtime. Retry in a moment, or stay in Local mode."
    except anthropic.APIConnectionError:
        return f"Could not reach the Connected Agent runtime right now. {_fallback(user_message, mode='local')}"
    except anthropic.APIStatusError as exc:
        return f"Connected Agent runtime error ({exc.status_code}). {_fallback(user_message, mode='local')}"
    except Exception as exc:
        return f"Unexpected error reaching the Connected Agent runtime ({type(exc).__name__}). {_fallback(user_message, mode='local')}"


def generate_demo_chat_reply(
    user_message: str, profile: dict[str, Any], hygiene: dict[str, Any],
    controls: dict[str, Any], validation: dict[str, Any] | None,
    mode: str = "local",
) -> str:
    return _fallback(user_message, profile, hygiene, controls, validation, mode=mode)


def _normalize_reply_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("```"):
            continue
        if set(stripped) <= {"-", "_", "*"} and len(stripped) >= 3:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip(" :")
            stripped = f"**{stripped}**"
        lines.append(stripped)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC FALLBACK — two distinct templates
# ═══════════════════════════════════════════════════════════════════════════

def _verdict_from_state(hygiene, validation, controls) -> tuple[str, str]:
    """Returns (label, one-sentence verdict) from current workflow state."""
    high = (hygiene or {}).get("severity_counts", {}).get("High", 0) if hygiene else 0
    overall = float((validation or {}).get("overall_score", 0) or 0)
    privacy = float((validation or {}).get("privacy_score", 0) or 0)

    if high and high > 0:
        return "Blocker", f"{high} high-severity hygiene finding(s) must be cleared before release."
    if overall == 0:
        return "Hold", "No validation on record yet — generate a synthetic preview first."
    if overall < 60:
        return "Hold", f"Overall quality {overall:.1f} below sandbox threshold — adjust controls and regenerate."
    if privacy < 70:
        return "Warning", f"Privacy score {privacy:.1f} is low — consider tighter ε or stricter field handling."
    if overall < 75:
        return "Warning", f"Overall quality {overall:.1f} is modest — review per-field drift before downstream use."
    return "Ready", f"Overall quality {overall:.1f}, privacy {privacy:.1f} — cleared for internal sandbox use."


def _local_reply(takeaway: str, bullets: list[str], next_action: str,
                 verdict_label: str | None = None) -> str:
    """Compose a Local Agent reply: verdict (optional) → bullets → next action."""
    bullet_lines = "\n".join(f"- {item}" for item in bullets[:4])
    parts = []
    if verdict_label:
        parts.append(f"**{verdict_label}** — {takeaway}")
    else:
        parts.append(takeaway)
    parts.append("")
    parts.append(bullet_lines)
    parts.append("")
    parts.append(f"Next action: {next_action}")
    return "\n".join(parts)


def _connected_reply(framing: str, sections: list[tuple[str, str]],
                     closing: str | None = None) -> str:
    """Compose a Connected Agent reply: framing → labelled sections → optional closing."""
    parts = [framing, ""]
    for label, body in sections[:4]:
        parts.append(f"**{label} —** {body}")
        parts.append("")
    if closing:
        parts.append(closing)
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def _is_greeting_or_noise(msg: str) -> bool:
    """Detect casual greetings, test inputs, or no-intent messages."""
    m = (msg or "").strip().lower()
    if not m:
        return True
    # Short single-word / 2-word casual inputs
    greetings = {
        "hi", "hello", "hey", "yo", "hola", "sup", "greetings",
        "thanks", "thank you", "ty", "thx", "ok", "okay", "cool", "nice",
        "test", "testing", "ping", "?", "??", "...",
        "bye", "goodbye", "later",
        "你好", "您好", "哈喽",
    }
    if m in greetings:
        return True
    # Very short and no analytical keywords
    word_count = len(m.split())
    if word_count <= 2:
        analytical_keywords = {
            "ready", "block", "blocker", "release", "share", "approve", "hygiene",
            "quality", "privacy", "fidelity", "epsilon", "drift", "pattern",
            "scenario", "stakeholder", "governance", "audit", "metadata",
            "copula", "constraint", "issue", "warning", "next", "action",
            "overall", "score", "synthetic", "generate", "preview",
        }
        if not any(k in m for k in analytical_keywords):
            return True
    return False


def _fallback(msg: str, profile=None, hygiene=None, controls=None, validation=None,
              mode: str = "local") -> str:
    m = msg.lower().strip()

    # ── GREETING / TINY TALK — short friendly redirect ─────────────────
    if _is_greeting_or_noise(msg):
        if mode == "local":
            return (
                "Hi — **Local Agent** here, the default in-workspace reasoning layer for "
                "readiness, blockers, and release decisions.\n\n"
                "Ask me something concrete, for example: *\"Is this package ready to share?\"*, "
                "*\"What blockers are still open?\"*, or *\"What should I do next?\"*"
            )
        return (
            "Hi — **Connected Agent** here. I explore patterns, scenarios, and stakeholder framing "
            "on the approved synthetic package.\n\n"
            "Ask me something concrete, for example: *\"What patterns should I flag for operations?\"*, "
            "*\"How should compliance read this package?\"*, or *\"How would a stronger-privacy "
            "setting change downstream use?\"*"
        )

    fp = (controls or {}).get("fidelity_priority", 50)
    rows = (profile or {}).get("summary", {}).get("rows", "the current package")
    overall_score = (validation or {}).get("overall_score", "pending")
    privacy_score = (validation or {}).get("privacy_score", "pending")
    correlation = (validation or {}).get("correlation_score", "n/a")

    verdict_label, verdict_sentence = _verdict_from_state(hygiene, validation, controls)

    # ── LOCAL AGENT fallback ────────────────────────────────────────────
    if mode == "local":
        if any(k in m for k in ["ready", "share", "release", "approve"]):
            return _local_reply(
                verdict_sentence,
                [
                    f"Overall {overall_score}, privacy {privacy_score}, correlation {correlation}.",
                    "Governance: reviewer approval drives the release gate.",
                    f"{(hygiene or {}).get('severity_counts', {}).get('High', 0)} high / "
                    f"{(hygiene or {}).get('severity_counts', {}).get('Medium', 0)} medium hygiene finding(s) on record.",
                ],
                "Confirm downstream scope is sandbox-appropriate before handoff.",
                verdict_label=verdict_label,
            )
        if any(k in m for k in ["privacy", "fidelity", "slider", "epsilon"]):
            label = "higher privacy" if fp < 40 else "balanced" if fp < 70 else "higher fidelity"
            return _local_reply(
                f"Privacy-vs-fidelity set to {label} ({fp}/100).",
                [
                    f"Current privacy score: {privacy_score}.",
                    "Lower fidelity priority adds noise and strengthens privacy.",
                    "Higher fidelity priority preserves source-like patterns more aggressively.",
                ],
                "Validate the trade-off before regenerating.",
            )
        if any(k in m for k in ["hygiene", "issue", "quality", "blocker", "warning"]):
            if hygiene:
                high = hygiene["severity_counts"]["High"]
                med = hygiene["severity_counts"]["Medium"]
                return _local_reply(
                    f"{high} high and {med} medium hygiene finding(s) flagged.",
                    [
                        f"Overall validation currently {overall_score}.",
                        "Corrections apply to metadata and controls only — not source rows.",
                        "Clear high-severity findings before broader internal use.",
                    ],
                    "Resolve the highest-severity finding first.",
                    verdict_label="Blocker" if high else ("Warning" if med else "Ready"),
                )
            return _local_reply(
                "Hygiene scan has not run yet.",
                ["Upload data and run the scan before any quality checks."],
                "Start at the upload step.",
                verdict_label="Hold",
            )
        if any(k in m for k in ["governance", "audit", "phipa", "boundary", "privacy law"]):
            return _local_reply(
                "Governance is active across the workflow.",
                [
                    "Every major action is audit-logged.",
                    "Reviewer approval gates generation and release.",
                    "Source rows are never copied into the agent layer.",
                ],
                "Use the audit and approval state to justify the release decision.",
            )
        if any(k in m for k in ["agent", "local", "connected", "mode"]):
            return _local_reply(
                "Local Agent is the default governed mode for workflow reasoning.",
                [
                    "Local stays fully inside the workspace — no external call.",
                    "Connected unlocks only after governance gates pass, on synthetic output only.",
                    "Escalate to Connected when you need deeper interpretation, not a decision.",
                ],
                "Stay in Local for readiness and release reasoning.",
            )
        # Generic
        return _local_reply(
            f"Workflow agent active on {rows} package.",
            [
                f"Overall {overall_score}, privacy {privacy_score}.",
                "Ask about readiness, blockers, hygiene, or next checks.",
            ],
            "Frame the next question around a concrete decision.",
            verdict_label=verdict_label,
        )

    # ── CONNECTED AGENT fallback ────────────────────────────────────────
    high_flags = (hygiene or {}).get("severity_counts", {}).get("High", 0)
    med_flags = (hygiene or {}).get("severity_counts", {}).get("Medium", 0)

    if any(k in m for k in ["ready", "share", "release", "approve", "use", "suitable"]):
        return _connected_reply(
            f"Overall {overall_score}, privacy {privacy_score}, correlation {correlation}. "
            f"Package is structurally sound; hygiene residue on source side is the live tension.",
            [
                ("Pattern",
                 f"Correlation {correlation} and privacy {privacy_score} clear the structural bar. "
                 f"Fidelity {(validation or {}).get('fidelity_score', 'n/a')} is the softer number — "
                 "marginal distributions drift more than joint relationships."),
                ("Scenario",
                 "Safe for operational analytics: throughput, triage mix, workflow timing. "
                 "Tail-sensitive modelling (extreme waits, rare CTAS) inherits DP noise at the edges."),
                ("Stakeholder view",
                 "Ops leads: use for throughput and mix questions. "
                 "Clinical analysts: aggregate-level only; records are not patient-linked. "
                 "Compliance: audit trail and synthetic-only boundary hold."),
                ("Watch point",
                 f"{high_flags} high / {med_flags} medium hygiene flag(s) remain on record. "
                 "Confirm no flagged field is in the analysis path."),
            ],
            "Connected Agent frames the read. The release call sits with Local Agent and the reviewer.",
        )
    if any(k in m for k in ["privacy", "fidelity", "epsilon", "dp", "noise"]):
        label = "higher-privacy" if fp < 40 else "balanced" if fp < 70 else "higher-fidelity"
        return _connected_reply(
            f"Posture is {label} ({fp}/100). Privacy score {privacy_score}, overall {overall_score}.",
            [
                ("Pattern",
                 f"At epsilon mapped from this posture, the trade-off is in balance: "
                 f"privacy {privacy_score} with correlation {correlation} held. "
                 "Structural signals survive; per-field distributions soften."),
                ("Scenario",
                 "Structural work (which fields correlate, what the mix looks like) holds up. "
                 "Tail-sensitive modelling (extreme wait times, rare triage levels) flattens under DP noise. "
                 "Acknowledge that in any tail conclusion."),
                ("Compared to",
                 "Stronger privacy widens the margin and compresses tails further. "
                 "Higher fidelity sharpens per-field signal and narrows the privacy buffer. "
                 "Current setting is the middle."),
            ],
        )
    if any(k in m for k in ["hygiene", "issue", "quality", "drift"]):
        return _connected_reply(
            f"{high_flags} high / {med_flags} medium hygiene flag(s) on record. "
            f"These shape how confidently the synthetic output can be read downstream.",
            [
                ("Pattern",
                 "Source-side hygiene findings propagate into synthetic output via the metadata pipeline. "
                 "A field flagged high on the source side produces noisier synthetic output in that field."),
                ("Scenario",
                 f"Analysis touching a flagged field: expect more drift than the {overall_score} headline. "
                 "Analysis avoiding flagged fields: closer to the overall number."),
                ("Watch point",
                 f"Correlation preservation is {correlation}. That is the honest read on joint "
                 "relationships. Per-field validation is the read on marginals."),
            ],
        )
    if any(k in m for k in ["stakeholder", "clinician", "ops", "manager", "audit"]):
        return _connected_reply(
            "Different stakeholders read the same package through different lenses.",
            [
                ("Stakeholder view",
                 f"Ops lead: throughput and mix analysis, overall {overall_score} supports it. "
                 "Clinical analyst: aggregate-level only; records are not patient-linked. "
                 "Compliance: audit trail and synthetic-only boundary are the anchors."),
                ("Pattern",
                 f"Correlation preservation at {correlation} covers joint relationships across fields — "
                 "usually the dimension ops and analyst readers care about most."),
                ("Watch point",
                 "Mark synthetic-sandbox status on any stakeholder deliverable. Not clinical-decision material."),
            ],
        )
    # Generic Connected fallback
    return _connected_reply(
        f"Overall {overall_score}, privacy {privacy_score}, correlation {correlation}. "
        "Bounded but usable synthetic asset.",
        [
            ("Pattern",
             "Quality, privacy, and correlation scores together give the first read on downstream fit."),
            ("Scenario",
             "Structural and aggregate-level analysis fits well in this range. "
             "Tail-sensitive or rare-event modelling needs more caution."),
        ],
        "Ask a more specific question — a use case, a field, a stakeholder — for a tighter read.",
    )
