"""Braintrust-managed CA digital-twin system prompt (push with ``bt functions push``).

Keep the system message text identical to ``SYSTEM_PROMPT`` in
``src/ca_personas/personas.py`` and the fenced block in
``prompts/system_prompt.md`` when syncing local → Braintrust.

Workflow for incremental prompt improvement
-------------------------------------------
1. Push this file once (or after local edits)::

       pip install -e ".[braintrust]"
       export BRAINTRUST_API_KEY=...
       bt functions push prompts/braintrust_ca_system.py

2. Iterate in the Braintrust playground (slug: ``ca-digital-twin-system``).
3. Point vLLM at the registry::

       export BRAINTRUST_PROMPT_SLUG=ca-digital-twin-system
       # optional pin:
       # export BRAINTRUST_PROMPT_VERSION=<version>
       ./scripts/run_vllm.sh

4. Compare experiments in Braintrust (parse_ok, exact/band match, inverse MAE).
5. When a playground version wins, copy the system text back into
   ``personas.py`` + ``prompts/system_prompt.md`` so offline/mock paths stay
   in sync.
"""

from __future__ import annotations

import braintrust

from ca_personas.personas import SYSTEM_PROMPT

PROJECT_NAME = "psych755-ca-personas"
PROMPT_SLUG = "ca-digital-twin-system"

project = braintrust.projects.create(name=PROJECT_NAME)

ca_digital_twin_system = project.prompts.create(
    name="CA digital-twin system prompt",
    slug=PROMPT_SLUG,
    description=(
        "PRCA communication-apprehension digital-twin system prompt for "
        "tiered persona vLLM / LLM runs (JSON scores + bands)."
    ),
    tags=["ca-digital-twin", "prca", "vllm"],
    # Model field is required by Braintrust prompt objects; vLLM runs override
    # the actual weights via --model / MODEL= on the GPU host.
    model="meta-llama/Llama-3.1-8B-Instruct",
    params={
        "temperature": 0.3,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    },
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            # Mustache: persona narrative from export_prompts / build-personas.
            "content": "{{{persona}}}",
        },
    ],
    metadata={
        "version_label": "v3.1-enhanced",
        "repo": "psych755-jjb",
        "sync_with": "src/ca_personas/personas.py::SYSTEM_PROMPT",
    },
)
