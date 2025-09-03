# Resilience Controller Prototype for LLM Agents

## Summary
This repository presents a resilience-oriented controller prototype for LLM agents. It maintains a bounded internal state ("hormones" like stress and curiosity), composes those into safe control knobs, and applies simple coping layers (timeouts, resets, adaptive baselines). The goal is to make agent behavior more stable in toy simulations. This is a research preview.

---

## The Problem Addressed (prototype scope)
Agent loops can become unstable under noisy inputs and tool errors. This prototype explores whether bounded internal state and layered coping improve stability in a controlled, reproducible toy setting.

---

## Architecture (implemented here)
```
Inputs → Content Processor → Hormone State → Resilience Layers → Bounded Knob → Step/Reward
                                      ↑                                  │
                                      └────────── Value Head ◀───────────┘
```
- Hormone dynamics: exponential decay + impulses (clamped to [0,1])
- Knob mapping: weighted, clamped blend of hormone contributions
- Resilience layers: circuit breaker, timeout, and baseline adaptation
- Value head: lightweight GCN over a tiny hormone graph (optional)

---

## Evidence (toy benchmark)
- A deterministic toy simulation shows higher “in-range” control rates with resilience layers enabled.
- Reproduce via `python resilient_controller.py` (see `test_resilient_chaos()`).
- Numbers are illustrative and specific to this setup; they are not general reliability claims.

---

## Scope and Limitations
- In scope: bounded-state controller, coping modes, small value head, and content→hormone conversion.
- Not in scope: production memory systems, domain benchmarks, safety certification, or economic projections.

---

## Reproducibility
- Python 3.11; see `requirements.txt`.
- Run: `python resilient_controller.py` (CPU by default, `--plot` optional).
- Outputs: console summary; optional plot via `plot_resilience_story`.

---

## What’s Next
- Add minimal integration tests and CI for core invariants (bounds, no-NaN).
- Provide a small, saved log/plot from a seeded run.
- Explore sparse graphs and richer reward signals.

---

## Related Documentation

### Core Documentation
- **[README.md](../README.md)** - Project overview and quick start guide
- **[FOUNDATION.md](FOUNDATION.md)** - Mathematical foundations and formal proofs
- **[IMPACT.md](IMPACT.md)** - Research significance and broader applications
- **[whitepaper.md](whitepaper.md)** - Comprehensive technical white paper
- **[wiki_page.md](wiki_page.md)** - Complete project wiki and reference guide

### Project Documentation
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines and standards
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[SECURITY.md](SECURITY.md)** - Security policy and vulnerability reporting

---

*Authors: Resilient AI Research, July 2025*  
*License: Apache License 2.0*
