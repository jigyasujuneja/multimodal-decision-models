#!/usr/bin/env python3
# Copyright 2026 Google LLC / Jigyasu Juneja
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scalable 7-Pillar, 14-Gate Binary (1/0) Evaluator & 3-Persona LLM Judge for TDDs and arXiv Papers.

Synthesizes Google Agent Marketplace and Google3 skills into a single domain-agnostic,
scalable evaluation harness supporting Markdown (`.md`), Typst (`.typ`), and LaTeX (`.tex`):
  1. `paper-writing-style` & `claudeisms` (`//learning/gemini/agents/clis/claudeisms/detector.go`
     + `references/scientific_paper_workflow.md` + `references/venue_checklists.md`)
  2. `design-doc-readability` (`//learning/gemini/agents/skills/software_design/design_doc_readability`
     + `references/taste_auditor.md` 14-Point Checklist + `references/clean_prose.md`)
  3. `writing-guidelines` (StellarDocs `//prototypes/projects/stellardocs/skills/writing_guidelines`
     Heading/Table Budgets & Plan-to-Doc Abstraction Firewall)
  4. `judgement-doc` (`go/judgment-is-the-job`: Feynman Whiteboard Defense, First-Principles
     Trade-Offs, 4 Marks of AI Slop, Contradiction Scan, Retrieve-or-Refuse Metric Provenance)
  5. `speak-like-tl` & `technical-writing` (`//learning/gemini/agents/skills/speak_like_tl`,
     `//learning/gemini/agents/skills/technical_writing`, `//learning/gemini/agents/skills/communication/doc_checklist`)
  6. `academic-paper-reviewer` & `document_review` (Math dimensional consistency, BibTeX
     citation resolution, anonymity/confidentiality, and 3-persona adversarial review)
  7. `golden-eval-judge-audit` (Calibrated LLM-as-a-Judge rubric where LLM leniency cannot
     override deterministic gate failures)

Every gate outputs a strict binary `1` (Meets Bar) or `0` (Fails Bar).
The overall document score is the logical AND (`min`) across all 14 gates (`14/14 -> 1`, else `0`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 1. PORT OF GOOGLE3 `claudeisms` DETECTOR (detector.go) + ZOMBIE NOUNS
# ==============================================================================

# Severity 3 (High-confidence LLM giveaway tics - strictly 0 allowed in narrative prose)
CLAUDEISMS_SEV3_PATTERNS: List[Tuple[str, str]] = [
    (r"\b[Dd]elve(?:s|d|ing)?\b", "vocab-delve: delve"),
    (r"\b(?:[Tt]apestr(?:y|ies)|rich\s+tapestry)\b", "vocab-tapestry: tapestry"),
    (r"\b[Pp]ivotal\b", "vocab-pivotal: pivotal"),
    (r"\b[Tt]estaments?\b", "vocab-testament: testament"),
    (r"\b[Rr]ealms?\b", "vocab-realm: realm"),
    (r"\b[Rr]evolutioni[sz](?:e|es|ed|ing)\b", "vocab-revolutionize: revolutionize"),
    (r"\b[Uu]nleash(?:es|ed|ing)?\b", "vocab-unleash: unleash"),
    (r"\b(?:ever-)?evolving\s+landscape\b", "vocab-landscape: evolving landscape"),
    (
        r"(?i)(?:I\s+hope\s+this\s+helps|Let\s+me\s+know\s+if|Of\s+course!|Certainly!|Great\s+question|You'?re\s+absolutely\s+right)",
        "vocab-chatbot-residue: conversational assistant filler",
    ),
    (
        r",\s*(?:thereby\s+|thus\s+)?(?:highlighting|underscoring|demonstrating|paving\s+the\s+way|fostering|showcasing|illuminating)\b",
        "participle-trailing: trailing significance participle",
    ),
    (
        r"\b[Nn]ot\s+only\b[^.!?\n]{1,90}\bbut\s+(?:also\b)?",
        "contrast-not-only-but: not only ... but also",
    ),
    (
        r"\b(?:is|are|was|were)\s+not\s+(?:just\s+|merely\s+|simply\s+|only\s+)?[^.!?\n]{1,60}\bbut\s+rather\b",
        "contrast-not-x-but-y: is not X, but rather Y",
    ),
    (
        r"\b(?:It\s+is\s+worth\s+noting|It'?s\s+worth\s+noting|It\s+bears\s+mentioning|Importantly,|Crucially,|Notably,)\b",
        "structure-throat-clearing: announced importance opener",
    ),
    (
        r"\b(?:In\s+conclusion,|To\s+summarize,|In\s+summary,|Taken\s+together,)\b",
        "structure-echo-conclusion: formulaic conclusion opener",
    ),
]

# Severity 2 (Inflated promotional vocabulary, copulative evasion, zombie nouns)
CLAUDEISMS_SEV2_PATTERNS: List[Tuple[str, str]] = [
    (
        r"\b(?:[Bb]eacon|[Cc]ornerstone|[Ss]ymphony|[Tt]ransformative|[Gg]roundbreaking|[Gg]ame-changing|[Cc]utting-edge|[Pp]aradigm\s+shift|[Ss]ynergy|[Hh]olistic|[Mm]eticulous|[Mm]ultifaceted|[Oo]verarching|[Pp]aramount|[Ii]ndispensable)\b",
        "vocab-puffery: inflated AI adjective/noun",
    ),
    (
        r"\b(?:[Uu]tiliz(?:e|es|ed|ing)|[Ll]everag(?:e|es|ed|ing)|[Ff]acilitat(?:e|es|ed|ing)|[Ff]oster(?:s|ed|ing)?|[Ee]mbark(?:s|ed|ing)?|[Ss]pearhead(?:s|ed|ing)?|[Bb]olster(?:s|ed|ing)?|[Gg]arner(?:s|ed|ing)?|[Bb]oast(?:s|ed|ing)?|[Hh]arness(?:es|ed|ing))\b",
        "vocab-inflated-verb: use plain 'use', 'enable', 'build', 'support'",
    ),
    (
        r"\b(?:serves\s+as\s+a|stands\s+as\s+a|acts\s+as\s+a|functions\s+as\s+a)\b",
        "structure-copula: inflated copula (use 'is')",
    ),
    (
        r",\s*ensuring\b",
        "participle-ensuring: trailing ', ensuring'",
    ),
    (
        r"\b(?:[Mm]oreover|[Ff]urthermore),\b",
        "structure-transition: mechanical sentence-initial connector (Moreover/Furthermore)",
    ),
    (
        r"\b(?:conduct\s+an\s+evaluation\s+of|make\s+a\s+determination|perform\s+the\s+ingestion\s+of|provide\s+an\s+explanation\s+for|facilitate\s+the\s+reduction\s+of|for\s+the\s+purpose\s+of|due\s+to\s+the\s+fact\s+that|at\s+this\s+point\s+in\s+time)\b",
        "clean-prose-zombie-noun: nominalized deadweight phrase (Steven Pinker / go/tw1)",
    ),
]

# Internal agent, level, or client leakage patterns (strictly 0 allowed)
META_AND_CLIENT_LEAKAGE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bJetski\b", "agent_leak: internal agent codename 'Jetski'"),
    (r"\bGDM-caliber\b|\bGDM\s+caliber\b", "meta_leak: 'GDM-caliber' self-label"),
    (r"\bL8\s+(?:GDM|Research|GenAI|Principal)\b", "meta_leak: internal level label 'L8 ...'"),
    (r"\bBoltzbit\b", "client_leak: Boltzbit"),
    (r"\bUnilever\b", "client_leak: Unilever"),
    (r"\bHUL\b", "client_leak: HUL"),
    (r"ul-fs-s-sandbx-16-prj", "client_leak: GCP sandbox project ID"),
    (r"\bProject\s+Aegis\b", "client_leak: internal codename Project Aegis"),
    (r"80%\s+of\s+the\s+way", "meta_prompt_leak: conversational prompt phrase '80% of the way'"),
]

# Stale model alias patterns & retired synthetic benchmark model names (strictly 0 allowed)
STALE_OR_SYNTHETIC_MODEL_PATTERNS: List[Tuple[str, str]] = [
    (r"gemini-2\.5-flash-lite", "stale_model: gemini-2.5-flash-lite (use gemini-3.5-flash-lite)"),
    (r"gemini-2\.5-flash\b", "stale_model: gemini-2.5-flash (use gemini-3.5-flash)"),
    (r"Claude-3\.7-Sonnet-Dirichlet-M5", "synthetic_model_leak: retired synthetic v0.5.0 model name"),
    (r"GPT-4o-Mini-Logprob-20Cap", "synthetic_model_leak: retired synthetic v0.5.0 model name"),
    (r"Microsoft-Decision-1-Style-9B", "synthetic_model_leak: retired synthetic v0.5.0 model name"),
]

# Stale MDM-Bench v0.5.0 synthetic numbers that were superseded by 100% real receipts
STALE_MDM_BENCH_METRIC_PATTERNS: List[Tuple[str, str]] = [
    (r"62\.50%\s*\(marginal\s+RAPS\)|rare-tail\s+coverage\s+from\s+`?62\.50%`?", "stale_metric: old synthetic Mondrian rare-tail baseline 62.50% (real is 58.75% -> 88.75%)"),
    (r"66\.25%\s*4-slot\s+coverage|from\s+`?66\.25%`?\s*\(uncorrected", "stale_metric: old synthetic 4-slot coverage 66.25% (real is 68.00% -> 90.25% / 90.50%)"),
    (r"flip\s+rate\s+from\s+`?16\.25%`?\s+to\s+`?6\.00%`?", "stale_metric: old synthetic PriDe flip rate 16.25% -> 6.00% (real is 10.25% -> 6.50%)"),
    (r"0\.1284\s*\\le\s*0\.14", "stale_metric: old synthetic LTT UCB 0.1284 (real is 0.0922 <= 0.14)"),
]


def detect_archetype(path: Path, raw_text: str) -> str:
    """Auto-detect document archetype from extension, filename, and content."""
    name = path.name.lower()
    if path.suffix == ".typ":
        return "arxiv_typst"
    if path.suffix == ".tex" or "\\documentclass" in raw_text:
        return "arxiv_tex"
    if "paper" in name or "preprint" in name:
        return "arxiv_md"
    if name == "result.md" or name.endswith("_report.md"):
        return "benchmark_report_md"
    if name == "readme.md":
        return "repo_readme_tdd"
    return "tdd_md"


def strip_code_and_math_for_prose(raw_text: str, suffix: str) -> str:
    """Mask fenced code, math, tables, and markup commands to isolate narrative prose."""
    text = raw_text
    if suffix == ".tex":
        text = re.sub(r"(?m)^%.*$", "", text)
        text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\\documentclass[\s\S]*?\\begin\{document\}", "", text)
        text = re.sub(
            r"\\begin\{(?:equation|align|Align|tabular|table|table\*|figure|figure\*|thebibliography|lstlisting|verbatim)\}[\s\S]*?\\end\{(?:equation|align|Align|tabular|table|table\*|figure|figure\*|thebibliography|lstlisting|verbatim)\}",
            "\n",
            text,
        )
        text = re.sub(r"\$[^$\n]+\$", " MATH_TOKEN ", text)
        text = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}", "\n\n", text)
        text = re.sub(r"\\(?:begin|end)\{(?:abstract|theorem|proof|enumerate|itemize)\}(?:\[[^\]]*\])?", "\n\n", text)
        text = re.sub(r"\\item\b", "\n\n", text)
        text = re.sub(r"\\(?:cite|citep|citet|ref|eqref|label|texttt|url|href)\{[^}]*\}", " CODE_TOKEN ", text)
        text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    elif suffix == ".typ":
        # Remove Typst comments //
        text = re.sub(r"(?m)^\s*//.*$", "", text)
        # Remove Typst #set, #show, #let preamble and #figure / #table blocks
        text = re.sub(r"(?m)^#(?:set|show|let|import)\b.*$", "", text)
        text = re.sub(r"(?s)#figure\([\s\S]*?\n\)", "\n", text)
        # Remove fenced code blocks
        text = re.sub(r"(?s)```.*?```", "\n", text)
        # Remove display math $ ... $ (multi-line or spaced) and inline math $...$
        text = re.sub(r"(?s)\$[^$]+\$", " MATH_TOKEN ", text)
        # Remove inline raw `...`
        text = re.sub(r"`[^`\n]+`", " CODE_TOKEN ", text)
        # Convert Typst headings (= Heading) to paragraph breaks
        text = re.sub(r"(?m)^=+\s+.*$", "\n\n", text)
    else:
        # Markdown
        text = re.sub(r"(?m)^```(?!`)[^\n]*\n[\s\S]*?^```(?!`)\s*$", "\n", text)
        text = re.sub(r"\$\$[\s\S]*?\$\$", "\n", text)
        text = re.sub(r"(?m)^\|.*\|$", "", text)
        text = re.sub(r"\$[^$\n]+\$", " MATH_TOKEN ", text)
        text = re.sub(r"`[^`\n]+`", " CODE_TOKEN ", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"(?m)^#{1,6}\s+.*$", "\n\n", text)
        text = re.sub(r"(?m)^\s*(?:[-*+]|\d+\.)\s+", "\n\n", text)
    return text


def find_line_matches(
    raw_text: str,
    pattern: str,
    suffix: str = ".md",
    ignore_code_and_quotes: bool = True,
    scan_inside_code_blocks: bool = False,
) -> List[Tuple[int, str]]:
    """Find line numbers and snippets matching a regex pattern, skipping code/math/table-header blocks."""
    matches: List[Tuple[int, str]] = []
    in_code = False
    in_env = False
    in_typ_figure = 0
    for idx, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```") and not stripped.startswith("````"):
            in_code = not in_code
            continue
        if in_code and not scan_inside_code_blocks:
            continue
        if suffix == ".tex":
            if not scan_inside_code_blocks and re.search(
                r"\\begin\{(?:equation|align|tabular|thebibliography|lstlisting|verbatim)\}", stripped
            ):
                in_env = True
                continue
            if not scan_inside_code_blocks and re.search(
                r"\\end\{(?:equation|align|tabular|thebibliography|lstlisting|verbatim)\}", stripped
            ):
                in_env = False
                continue
            if (in_env and not scan_inside_code_blocks) or stripped.startswith("%"):
                continue
        elif suffix == ".typ":
            if stripped.startswith("//"):
                continue
            if not scan_inside_code_blocks:
                if stripped.startswith("#figure(") or stripped.startswith("table("):
                    in_typ_figure += stripped.count("(") - stripped.count(")")
                    continue
                if in_typ_figure > 0:
                    in_typ_figure += stripped.count("(") - stripped.count(")")
                    if in_typ_figure < 0:
                        in_typ_figure = 0
                    continue

        scan_line = line
        if ignore_code_and_quotes:
            scan_line = re.sub(r"`[^`]+`", " ", scan_line)
            scan_line = re.sub(r"\$[^$\n]+\$", " ", scan_line)
            scan_line = re.sub(r"\\texttt\{[^}]+\}", " ", scan_line)
            # Skip quoted prompt-injection payload examples (e.g. '"IGNORE PREVIOUS..."')
            scan_line = re.sub(r'"[^"\n]*IGNORE PREVIOUS[^"\n]*"', " ", scan_line)
        m = re.search(pattern, scan_line)
        if m:
            matches.append((idx, m.group(0)))
    return matches


def _verify_markdown_links(path: Path, raw_text: str) -> List[str]:
    """Verify that relative file links and file:/// links resolve on disk."""
    broken: List[str] = []
    in_code = False
    for line in raw_text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        # Strip inline math $...$ and inline code `...` so math like [p_k](1 - E[p_k]) is not parsed as a URL
        scan_line = re.sub(r"\$[^$\n]+\$", " ", line)
        scan_line = re.sub(r"`[^`\n]+`", " ", scan_line)
        for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", scan_line):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#", "conversation://")):
                continue
            if target.startswith("file://"):
                file_part = target[len("file://") :].split("#")[0].replace("%20", " ")
                if not Path(file_part).exists():
                    broken.append(target)
            else:
                rel_part = target.split("#")[0]
                if not rel_part:
                    continue
                resolved = (path.parent / rel_part).resolve()
                if not resolved.exists():
                    broken.append(target)
    return broken


def _verify_bib_citations(path: Path, raw_text: str, suffix: str) -> Tuple[int, List[str]]:
    """Verify that all @citation keys in .typ or \\cite{} keys in .tex resolve in refs.bib."""
    bib_candidates = [
        path.parent / "refs.bib",
        path.parent / "references.bib",
        path.parent.parent / "refs.bib",
    ]
    bib_path = next((b for b in bib_candidates if b.is_file()), None)
    if bib_path is None:
        return 0, []

    bib_text = bib_path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))

    cited_keys: set[str] = set()
    if suffix == ".typ":
        # Strip inline raw `...` and math $...$ before scanning @key
        cleaned = re.sub(r"(?s)```.*?```", " ", raw_text)
        cleaned = re.sub(r"`[^`\n]+`", " ", cleaned)
        cleaned = re.sub(r"(?s)\$[^$]+\$", " ", cleaned)
        for m in re.finditer(r"(?<![\w`])@([a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_-]*)", cleaned):
            cited_keys.add(m.group(1))
    elif suffix == ".tex":
        for m in re.finditer(r"\\cite[pt]?\*?\{([^}]+)\}", raw_text):
            for k in m.group(1).split(","):
                if k.strip():
                    cited_keys.add(k.strip())

    missing = sorted(k for k in cited_keys if k not in bib_keys)
    return len(cited_keys), missing


def evaluate_document(path: Path, run_llm_judge: bool = False) -> Dict[str, Any]:
    """Run all 14 binary (1/0) gates across the 7 pillars on a single document."""
    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    archetype = detect_archetype(path, raw_text)
    prose_text = strip_code_and_math_for_prose(raw_text, suffix)

    gates: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------------------------
    # PILLAR 1: TASTE (G1, G2)
    # --------------------------------------------------------------------------
    # G1: Respectful Terminology ("existing" over "legacy") & Zero Meta/Agent/Client Leakage
    legacy_hits = find_line_matches(raw_text, r"\b[Ll]egacy\b", suffix=suffix)
    leak_hits: List[Tuple[int, str, str]] = []
    for pat, label in META_AND_CLIENT_LEAKAGE_PATTERNS:
        for ln, snippet in find_line_matches(
            raw_text, pat, suffix=suffix, ignore_code_and_quotes=False, scan_inside_code_blocks=True
        ):
            leak_hits.append((ln, snippet, label))

    g1_pass = 1 if (len(legacy_hits) == 0 and len(leak_hits) == 0) else 0
    gates["G1_taste_terminology_and_anonymity"] = {
        "pillar": "1. Taste",
        "name": "Respectful Terminology ('existing' vs 'legacy') & Zero Meta/Agent/Client Leaks",
        "score": g1_pass,
        "details": {
            "legacy_count": len(legacy_hits),
            "legacy_hits": legacy_hits[:5],
            "leak_count": len(leak_hits),
            "leaks": leak_hits[:5],
        },
    }

    # G2: Decisive Stance (go/tott/773) & Honest Negative Space / Limitations / Boundaries
    has_negative_space = bool(
        re.search(
            r"(?i)\b(?:non-goals?|limitations?|when\s+not\s+to\s+use|failure\s+modes?|scope\s+boundaries|rejected\s+alternatives|trade-?offs?|null\s+results?|falsif(?:ied|ies)|abstain|quarantine|under-?coverage|bottleneck)\b",
            raw_text,
        )
    )
    has_decisive_rationale = bool(
        re.search(
            r"(?i)\b(?:we\s+select|we\s+adopt|we\s+reject|we\s+decouple|we\s+introduce|we\s+propose|outperforms?|dominates?|pareto|ablation|compar(?:ed|ison|ing)\s+(?:to|with|against)|beats?|reduces?|eliminates?|restores?)\b",
            raw_text,
        )
    )
    g2_pass = 1 if (has_negative_space and has_decisive_rationale) else 0
    gates["G2_taste_decisive_stance_and_negative_space"] = {
        "pillar": "1. Taste",
        "name": "Decisive Stance (go/tott/773) & Explicit Negative Space / Limitations",
        "score": g2_pass,
        "details": {
            "has_negative_space_or_limitations": has_negative_space,
            "has_decisive_tradeoff_rationale": has_decisive_rationale,
        },
    }

    # --------------------------------------------------------------------------
    # PILLAR 2: WRITING & ANTI-SLOP (G3, G4)
    # --------------------------------------------------------------------------
    # G3: Zero LLM Stylistic Tells & Zombie Nouns (Google3 `claudeisms` detector.go port)
    sev3_hits: List[Tuple[int, str, str]] = []
    for pat, label in CLAUDEISMS_SEV3_PATTERNS:
        for ln, snippet in find_line_matches(raw_text, pat, suffix=suffix):
            sev3_hits.append((ln, snippet, label))

    sev2_hits: List[Tuple[int, str, str]] = []
    for pat, label in CLAUDEISMS_SEV2_PATTERNS:
        for ln, snippet in find_line_matches(raw_text, pat, suffix=suffix):
            sev2_hits.append((ln, snippet, label))

    g3_pass = 1 if (len(sev3_hits) == 0 and len(sev2_hits) == 0) else 0
    gates["G3_writing_zero_claudeisms_and_ai_slop"] = {
        "pillar": "2. Writing",
        "name": "Zero LLM Tells, Puffery & Zombie Nouns (Google3 `claudeisms` Port)",
        "score": g3_pass,
        "details": {
            "sev3_count": len(sev3_hits),
            "sev3_hits": sev3_hits[:10],
            "sev2_count": len(sev2_hits),
            "sev2_hits": sev2_hits[:10],
        },
    }

    # G4: Clean Punctuation & Active Engineering Register
    double_emdash_hits = [
        (ln, snip)
        for ln, snip in find_line_matches(
            raw_text,
            r"(?:—[^—\n]{3,80}—|(?<!\|)\s*---[^-|\n]{3,80}---)",
            suffix=suffix,
        )
        if not raw_text.splitlines()[ln - 1].strip().startswith("|")
    ]
    raw_arrow_hits: List[Tuple[int, str]] = []
    for ln, snippet in find_line_matches(raw_text, r"(?<=\s)->(?=\s)", suffix=suffix):
        line_str = raw_text.splitlines()[ln - 1].strip()
        if not line_str.startswith(("#", "|", "-", "*", "\\", "=", "//")):
            raw_arrow_hits.append((ln, snippet))

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", prose_text)
        if len(s.split()) >= 5
    ]
    passive_pat = re.compile(
        r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(?:\w+ed|built|done|made|given|shown|known|seen|found|taken|held|set|kept|drawn|sent|written|run|put|cut|read|lost|paid|met|said|told|brought|bought|caught|taught|thought|sought)\b",
        re.IGNORECASE,
    )
    passive_count = sum(1 for s in sentences if passive_pat.search(s))
    passive_pct = round(100.0 * passive_count / max(len(sentences), 1), 1)

    g4_pass = 1 if (len(double_emdash_hits) == 0 and len(raw_arrow_hits) == 0 and passive_pct <= 20.0) else 0
    gates["G4_writing_punctuation_and_active_voice"] = {
        "pillar": "2. Writing",
        "name": "Clean Punctuation (Zero Double Em-Dashes / Raw Prose Arrows) & Active Voice (<=20%)",
        "score": g4_pass,
        "details": {
            "double_emdash_count": len(double_emdash_hits),
            "double_emdash_hits": double_emdash_hits[:5],
            "raw_prose_arrow_count": len(raw_arrow_hits),
            "raw_prose_arrow_hits": raw_arrow_hits[:5],
            "passive_voice_pct": passive_pct,
        },
    }

    # --------------------------------------------------------------------------
    # PILLAR 3: FACTUALITY (G5, G6)
    # --------------------------------------------------------------------------
    # G5: Quantitative & Statistical Rigor (Exact N / sample counts, proper scoring / significance, units)
    has_sample_size = bool(
        re.search(
            r"\b[Nn]\s*=\s*[\d,]+|\b\d{2,5}[- ](?:item|query|trial|state|trajectory|case|instance|example)",
            raw_text,
        )
    )
    has_stat_or_proper_scoring = bool(
        re.search(
            r"\b(?:Brier|ECE|NLL|AURC|McNemar|Wilcoxon|bootstrap|CI\b|SE\b|p\s*[=<]\s*0\.\d+|p\s*=\s*\d+\.\d+\s*\\times\s*10|\bUCB\b|\\alpha\s*=)",
            raw_text,
            re.IGNORECASE,
        )
    )
    has_units = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|%|tokens|bytes|KB|MB|nats|bits|s\b)", raw_text)
    )
    g5_pass = 1 if (has_sample_size and has_stat_or_proper_scoring and has_units) else 0
    gates["G5_factuality_statistical_and_quantitative_rigor"] = {
        "pillar": "3. Factuality",
        "name": "Quantitative & Statistical Rigor (Explicit Sample Size N, Proper Scoring / Significance, Units)",
        "score": g5_pass,
        "details": {
            "has_sample_size": has_sample_size,
            "has_stat_or_proper_scoring": has_stat_or_proper_scoring,
            "has_physical_units": has_units,
        },
    }

    # G6: Cross-Document Model & Metric Parity (Zero stale/synthetic model drift or stale MDM-Bench numbers)
    stale_hits: List[Tuple[int, str, str]] = []
    for pat, label in STALE_OR_SYNTHETIC_MODEL_PATTERNS + STALE_MDM_BENCH_METRIC_PATTERNS:
        for ln, snippet in find_line_matches(
            raw_text, pat, suffix=suffix, ignore_code_and_quotes=False, scan_inside_code_blocks=True
        ):
            stale_hits.append((ln, snippet, label))

    g6_pass = 1 if len(stale_hits) == 0 else 0
    gates["G6_factuality_model_and_metric_consistency"] = {
        "pillar": "3. Factuality",
        "name": "Cross-Document Model & Metric Parity (Zero Stale/Synthetic Model or Metric Drift)",
        "score": g6_pass,
        "details": {
            "contradiction_or_stale_count": len(stale_hits),
            "contradiction_or_stale_hits": stale_hits[:10],
        },
    }

    # --------------------------------------------------------------------------
    # PILLAR 4: GROUNDEDNESS (G7, G8)
    # --------------------------------------------------------------------------
    # G7: Claim-to-Proof / Artifact Grounding, Broken Link Verification & Zero Placeholders
    placeholder_hits = find_line_matches(
        raw_text, r"\b(?:TODO|TBD|FIXME|XXX|PLACEHOLDER|INSERT\s+HERE)\b", suffix=suffix
    )
    broken_links = _verify_markdown_links(path, raw_text) if suffix == ".md" else []
    has_math_or_code_grounding = bool(
        re.search(
            r"(?:\\operatorname|\\sum|\\frac|sum_|frac\(|`[a-zA-Z0-9_]+\.(?:py|go|typ|json|jsonl)`|def\s+\w+|class\s+\w+)",
            raw_text,
        )
    )
    g7_pass = 1 if (len(placeholder_hits) == 0 and len(broken_links) == 0 and has_math_or_code_grounding) else 0
    gates["G7_groundedness_claim_to_proof_and_zero_placeholders"] = {
        "pillar": "4. Groundedness",
        "name": "Claim-to-Proof/Artifact Grounding, 100% Link Resolution & Zero Placeholders",
        "score": g7_pass,
        "details": {
            "placeholder_count": len(placeholder_hits),
            "placeholder_hits": placeholder_hits[:5],
            "broken_links_count": len(broken_links),
            "broken_links": broken_links[:5],
            "has_math_or_code_grounding": has_math_or_code_grounding,
        },
    }

    # G8: Citation & External Baseline Integrity (Zero unresolved citations + BibTeX resolution + named baselines)
    unresolved_citations = find_line_matches(
        raw_text, r"(?:\[\?\]|\?\?|\\cite\{\s*\})", suffix=suffix, ignore_code_and_quotes=False
    )
    cited_count, missing_bib_keys = _verify_bib_citations(path, raw_text, suffix)
    baseline_mentions = re.findall(
        r"\b(?:RAPS|APS|LAC|Platt|Temperature\s+Scaling|Vector\s+Scaling|Dirichlet|Jeffreys|Murphy|Brier|Vovk|Sadinle|Romano|Angelopoulos|Tibshirani|Guo|Zheng|PriDe|Bansal|Gibbs|Cand[eè]s|LinUCB|PUCT|McNemar|Hoeffding|Bentkus|Saerens|Tango|Youden|TF-IDF|FIFO|DecisionGemma|Gemini|Strands|Laya|Qwen|DeepSeek|Llama|Gemma)\b",
        raw_text,
        re.IGNORECASE,
    )
    unique_baselines = sorted({b.lower() for b in baseline_mentions})
    g8_pass = 1 if (
        len(unresolved_citations) == 0
        and len(missing_bib_keys) == 0
        and len(unique_baselines) >= 3
    ) else 0
    gates["G8_groundedness_citations_and_baselines"] = {
        "pillar": "4. Groundedness",
        "name": "Citation Integrity (100% BibTeX Resolution) & Named Baseline Comparisons (>=3)",
        "score": g8_pass,
        "details": {
            "unresolved_citation_count": len(unresolved_citations),
            "bib_citations_checked": cited_count,
            "missing_bib_keys": missing_bib_keys,
            "unique_baseline_or_method_count": len(unique_baselines),
            "sample_baselines_found": unique_baselines[:10],
        },
    }

    # --------------------------------------------------------------------------
    # PILLAR 5: STRUCTURE (G9, G10)
    # --------------------------------------------------------------------------
    # G9: Archetype Section Completeness & Intuitive Restatement / Operational Takeaway
    if suffix == ".tex":
        raw_headings = re.findall(r"\\(?:sub)*section\*?\{([^}]+)\}", raw_text)
    elif suffix == ".typ":
        raw_headings = [
            m.group(1).strip()
            for m in re.finditer(r"(?m)^=+\s+(.+)$", raw_text)
        ]
    else:
        raw_headings = [
            m.group(1).strip()
            for m in re.finditer(r"(?m)^#{1,3}\s+(.+)$", raw_text)
        ]

    min_headings = 3 if archetype == "benchmark_report_md" else 4
    has_operational_explanation = bool(
        re.search(
            r"(?i)\b(?:intuition|intuitive|in\s+plain|takeaway|finding|interpretation|mechanism|problem|solution|why\s+it\s+matters|verdict|guarantee|because|where\s+each\s+paradigm\s+wins)\b",
            raw_text,
        )
    )
    g9_pass = 1 if (len(raw_headings) >= min_headings and has_operational_explanation) else 0
    gates["G9_structure_archetype_and_intuitive_restatement"] = {
        "pillar": "5. Structure",
        "name": "Archetype Section Completeness & Intuitive / Operational Restatement",
        "score": g9_pass,
        "details": {
            "archetype": archetype,
            "headings_count": len(raw_headings),
            "min_headings_required": min_headings,
            "has_operational_or_intuitive_restatement": has_operational_explanation,
        },
    }

    # G10: Heading Brevity (<=12 words avg, zero Wh-question headings in formal specs) & Paragraph Pacing (<=85 words avg)
    avg_heading_words = round(
        sum(len(h.split()) for h in raw_headings) / max(len(raw_headings), 1), 1
    )
    question_headings = [
        h
        for h in raw_headings
        if re.match(r"(?i)^(?:\d+(?:\.\d+)*\s+)?(?:why|how|what|where|when|who)\b.*\?$", h.strip())
    ]
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", prose_text)
        if len(p.strip().split()) >= 12
    ]
    avg_para_words = round(
        sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1), 1
    )
    g10_pass = 1 if (
        avg_heading_words <= 12.0
        and len(question_headings) == 0
        and avg_para_words <= 85.0
    ) else 0
    gates["G10_structure_heading_and_paragraph_pacing"] = {
        "pillar": "5. Structure",
        "name": "Heading Brevity (<=12 words avg, Zero Wh-Question Headings) & Paragraph Pacing (<=85 words avg)",
        "score": g10_pass,
        "details": {
            "avg_heading_words": avg_heading_words,
            "question_headings": question_headings,
            "avg_paragraph_words": avg_para_words,
            "paragraph_count": len(paragraphs),
        },
    }

    # --------------------------------------------------------------------------
    # PILLAR 6: FORMATTING (G11, G12)
    # --------------------------------------------------------------------------
    # G11: Structured Comparison Tables (>=1), Balanced Math Delimiters & Compiled PDF (for .typ/.tex)
    if suffix == ".tex":
        table_count = len(re.findall(r"\\begin\{table\*?\}", raw_text))
        pdf_path = path.with_suffix(".pdf")
        has_compiled_pdf = pdf_path.exists() and pdf_path.stat().st_size > 50_000
    elif suffix == ".typ":
        table_count = len(re.findall(r"\btable\s*\(", raw_text))
        pdf_path = path.with_suffix(".pdf")
        has_compiled_pdf = pdf_path.exists() and pdf_path.stat().st_size > 50_000
    else:
        table_count = len(re.findall(r"(?m)^\|[-:| ]+\|$", raw_text))
        has_compiled_pdf = True

    unbalanced_math_lines: List[int] = []
    if suffix in (".md", ".tex"):
        in_code_blk = False
        in_display_math = False
        for idx, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_blk = not in_code_blk
                continue
            if in_code_blk or stripped.startswith("%"):
                continue
            if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 2:
                continue
            if stripped == "$$":
                in_display_math = not in_display_math
                continue
            if in_display_math:
                continue
            cleaned = re.sub(r"\\\$", "", line)
            cleaned = re.sub(r"\$\$", "", cleaned)
            cleaned = re.sub(r"`[^`]*`", "", cleaned)
            if cleaned.count("$") % 2 != 0:
                unbalanced_math_lines.append(idx)
    elif suffix == ".typ":
        # In Typst, check that total unescaped $ outside raw blocks is even
        cleaned_typ = re.sub(r"(?s)```.*?```", "", raw_text)
        cleaned_typ = re.sub(r"`[^`\n]+`", "", cleaned_typ)
        cleaned_typ = re.sub(r"\\\$", "", cleaned_typ)
        if cleaned_typ.count("$") % 2 != 0:
            unbalanced_math_lines.append(1)

    g11_pass = 1 if (table_count >= 1 and len(unbalanced_math_lines) == 0 and has_compiled_pdf) else 0
    gates["G11_formatting_tables_and_math_hygiene"] = {
        "pillar": "6. Formatting",
        "name": "Structured Comparison Tables (>=1), Balanced Math Delimiters & Compiled PDF",
        "score": g11_pass,
        "details": {
            "table_count": table_count,
            "unbalanced_math_lines": unbalanced_math_lines[:5],
            "has_compiled_pdf": has_compiled_pdf,
        },
    }

    # G12: Visual Scannability / Publication Figures & Zero Raw ASCII Box-Art Diagrams
    ascii_box_hits = find_line_matches(
        raw_text,
        r"(?:\+-[-+]{3,}\+|[┌┐└┘├┤┬┴┼─│]{3,})",
        suffix=suffix,
        ignore_code_and_quotes=False,
    )
    has_figures_or_high_scannability = bool(
        re.search(
            r"(?:\\includegraphics|#figure\(|image\(|!\[[^\]]*\]\([^)]+\.(?:png|svg|pdf)\)|```mermaid)",
            raw_text,
        )
    ) or (table_count >= 1 and len(re.findall(r"(?m)^(?:\s*[-*+]|\s*\d+\.|\\item|\|)", raw_text)) >= 8)

    g12_pass = 1 if (len(ascii_box_hits) == 0 and has_figures_or_high_scannability) else 0
    gates["G12_formatting_publication_visuals_and_zero_ascii_art"] = {
        "pillar": "6. Formatting",
        "name": "Publication Figures / Structured Scannability & Zero Raw ASCII Box Art",
        "score": g12_pass,
        "details": {
            "ascii_box_art_count": len(ascii_box_hits),
            "ascii_box_art_hits": ascii_box_hits[:5],
            "has_figures_or_structured_scannability": has_figures_or_high_scannability,
        },
    }

    # --------------------------------------------------------------------------
    # PILLAR 7: POLISH & 3-PERSONA LLM JUDGE (G13, G14)
    # --------------------------------------------------------------------------
    # G14: Goldfish Self-Containment Test & Plan-to-Doc Abstraction Firewall
    plan_leak_hits = find_line_matches(
        raw_text,
        r"(?i)\b(?:User\s+Request:|Step\s+\d+\.\d+:|Task\s+\d+:|In\s+this\s+PR\s+we|In\s+this\s+CL\s+we)\b",
        suffix=suffix,
    )
    # Cold-reader check: opening 2,500 chars state a clear title/heading and concrete technical scope
    has_self_contained_opening = len(raw_text.strip()) >= 1200 and bool(
        re.search(r"(?m)^(?:#|=|\\title|\\section)", raw_text[:1500])
    )
    g14_pass = 1 if (len(plan_leak_hits) == 0 and has_self_contained_opening) else 0
    gates["G14_polish_goldfish_self_containment_and_firewall"] = {
        "pillar": "7. Polish",
        "name": "Goldfish Self-Containment Test & Plan-to-Doc Abstraction Firewall",
        "score": g14_pass,
        "details": {
            "plan_leak_count": len(plan_leak_hits),
            "plan_leak_hits": plan_leak_hits[:5],
            "has_self_contained_opening": has_self_contained_opening,
        },
    }

    # G13: 3-Persona Adversarial Review Judge (Theory Specialist + Empirical Practitioner + Principal Architect)
    # Grounded in `paper-writing-style` Phase 5 + `golden-eval-judge-audit` calibration guard
    theory_persona_pass = (g7_pass == 1 and g11_pass == 1) and bool(
        re.search(
            r"(?i)\b(?:simplex|conformal|entropy|hesitation|dirichlet|brier|posterior|quantile|submodular|polytope|bandit|covariance|theorem|proposition|invariant)\b",
            raw_text,
        )
    )
    empirical_persona_pass = (g5_pass == 1 and g6_pass == 1 and g8_pass == 1)
    architect_persona_pass = (g1_pass == 1 and g2_pass == 1 and g3_pass == 1 and g4_pass == 1 and g9_pass == 1 and g10_pass == 1 and g14_pass == 1)

    llm_judge_verdict: Optional[Dict[str, Any]] = None
    if run_llm_judge:
        llm_judge_verdict = call_live_llm_judge(path.name, archetype, raw_text[:12000])

    deterministic_3persona_pass = 1 if (theory_persona_pass and empirical_persona_pass and architect_persona_pass) else 0
    if llm_judge_verdict and "overall_pass" in llm_judge_verdict:
        # Calibration guard (`golden-eval-judge-audit`): LLM judge cannot override a deterministic failure
        g13_pass = 1 if (deterministic_3persona_pass == 1 and llm_judge_verdict["overall_pass"] == 1) else 0
    else:
        g13_pass = deterministic_3persona_pass

    gates["G13_polish_three_persona_llm_judge"] = {
        "pillar": "7. Polish",
        "name": "3-Persona Adversarial Judge (Theory Specialist, Empirical Practitioner, Principal Architect)",
        "score": g13_pass,
        "details": {
            "theory_specialist_pass": int(theory_persona_pass),
            "empirical_practitioner_pass": int(empirical_persona_pass),
            "principal_architect_pass": int(architect_persona_pass),
            "live_llm_judge": llm_judge_verdict,
        },
    }

    total_passed = sum(g["score"] for g in gates.values())
    total_gates = len(gates)
    overall_binary_score = 1 if total_passed == total_gates else 0

    pillars_summary: Dict[str, int] = {}
    for g in gates.values():
        p = g["pillar"]
        pillars_summary[p] = pillars_summary.get(p, 1) & g["score"]

    return {
        "file": str(path),
        "filename": path.name,
        "archetype": archetype,
        "overall_binary_score": overall_binary_score,
        "gates_passed": total_passed,
        "gates_total": total_gates,
        "pillars_binary": pillars_summary,
        "gates": gates,
    }


def call_live_llm_judge(filename: str, archetype: str, excerpt: str) -> Dict[str, Any]:
    """Optional live LLM-as-a-Judge call via Vertex AI / Gemini API when credentials are present."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {
            "judge_mode": "calibrated_deterministic_3persona",
            "note": "Evaluated via deterministic 3-persona calibration-guarded rubric (set GEMINI_API_KEY for live API call).",
        }
    try:
        import urllib.request

        prompt = (
            "You are a strict binary (1/0) LLM Judge evaluating a technical design document or arXiv paper "
            f"('{filename}', archetype='{archetype}') across 3 adversarial personas:\n"
            "1. Theory Specialist: Are equations/mechanisms dimensionally sound, properly bounded, and paired with clear operational restatements?\n"
            "2. Empirical Practitioner: Are baselines named, sample sizes (N) and proper scoring / significance metrics explicit, and claims grounded in real data?\n"
            "3. Principal Architect: Is the prose free of AI slop ('delve', 'pivotal', 'tapestry', 'legacy'), decisive in trade-offs, and self-contained for a cold reader?\n\n"
            "Respond ONLY with valid JSON: "
            '{"theory_pass": 1, "empirical_pass": 1, "architect_pass": 1, "overall_pass": 1, "rationale": "one sentence summary"}\n\n'
            f"DOCUMENT EXCERPT:\n{excerpt[:6000]}"
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
        payload = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text_resp = body["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text_resp)
        if isinstance(parsed, dict) and "overall_pass" in parsed:
            parsed["judge_mode"] = "live_gemini_api"
            return parsed
    except Exception as exc:
        return {
            "judge_mode": "calibrated_deterministic_3persona",
            "note": f"Live API call skipped ({type(exc).__name__}); verified via deterministic 3-persona rubric.",
        }
    return {"judge_mode": "calibrated_deterministic_3persona"}


def format_markdown_report(results: List[Dict[str, Any]]) -> str:
    """Render a clean Markdown scorecard table across all evaluated documents."""
    lines: List[str] = []
    lines.append("# Scalable TDD & arXiv Research Paper Binary (`1/0`) Evaluation Report")
    lines.append("")
    lines.append(
        "| Document | Archetype | Taste (`G1`,`G2`) | Writing (`G3`,`G4`) | Factuality (`G5`,`G6`) | Groundedness (`G7`,`G8`) | Structure (`G9`,`G10`) | Formatting (`G11`,`G12`) | Polish (`G13`,`G14`) | Gates Passed | Overall (`1/0`) |"
    )
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        g = r["gates"]
        c_taste = f"{g['G1_taste_terminology_and_anonymity']['score']}/{g['G2_taste_decisive_stance_and_negative_space']['score']}"
        c_writ = f"{g['G3_writing_zero_claudeisms_and_ai_slop']['score']}/{g['G4_writing_punctuation_and_active_voice']['score']}"
        c_fact = f"{g['G5_factuality_statistical_and_quantitative_rigor']['score']}/{g['G6_factuality_model_and_metric_consistency']['score']}"
        c_grnd = f"{g['G7_groundedness_claim_to_proof_and_zero_placeholders']['score']}/{g['G8_groundedness_citations_and_baselines']['score']}"
        c_strc = f"{g['G9_structure_archetype_and_intuitive_restatement']['score']}/{g['G10_structure_heading_and_paragraph_pacing']['score']}"
        c_fmt = f"{g['G11_formatting_tables_and_math_hygiene']['score']}/{g['G12_formatting_publication_visuals_and_zero_ascii_art']['score']}"
        c_pol = f"{g['G13_polish_three_persona_llm_judge']['score']}/{g['G14_polish_goldfish_self_containment_and_firewall']['score']}"
        lines.append(
            f"| `{r['filename']}` | `{r['archetype']}` | **{c_taste}** | **{c_writ}** | **{c_fact}** | **{c_grnd}** | **{c_strc}** | **{c_fmt}** | **{c_pol}** | **{r['gates_passed']}/{r['gates_total']}** | **{r['overall_binary_score']}** |"
        )

    lines.append("")
    passed_docs = sum(1 for r in results if r["overall_binary_score"] == 1)
    lines.append(
        f"**Suite Summary**: `{passed_docs}/{len(results)}` documents achieved a strict **`1` (`14/14` gates passed)**."
    )
    lines.append("")
    for r in results:
        failed_gates = [(k, v) for k, v in r["gates"].items() if v["score"] == 0]
        if failed_gates:
            lines.append(f"### Gate Failures in `{r['filename']}` (`{r['file']}`)")
            for k, v in failed_gates:
                lines.append(f"- **`{k}` ({v['name']})**: `0` — `{json.dumps(v['details'])}`")
            lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TDDs and arXiv papers (.md, .typ, .tex) on 14 binary (1/0) gates across 7 pillars."
    )
    parser.add_argument("paths", nargs="+", help="Markdown (.md), Typst (.typ), or LaTeX (.tex) files to evaluate.")
    parser.add_argument("--llm-judge", action="store_true", help="Invoke live Gemini API 3-persona judge alongside deterministic gates.")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save JSON evaluation report.")
    parser.add_argument("--md-out", type=str, default="", help="Optional path to save Markdown evaluation report.")
    args = parser.parse_args(argv)

    results: List[Dict[str, Any]] = []
    for p_str in args.paths:
        p = Path(p_str)
        if not p.exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            return 2
        results.append(evaluate_document(p, run_llm_judge=args.llm_judge))

    md_report = format_markdown_report(results)
    print(md_report)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(md_report, encoding="utf-8")

    all_passed = all(r["overall_binary_score"] == 1 for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
