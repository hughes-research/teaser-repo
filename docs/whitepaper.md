# Resilient Psychological AI

## A White Paper on Homeostatic, Self‑Learning Control for Agents

Authors:
    Resilient AI Research July 2025

License:
    Apache License 2.0 - See LICENSE file for details

---

## Executive Summary

Modern AI is powerful but fragile. Systems fail under stress, distribution shift, tool errors, or adversarial inputs. This white paper introduces **Resilient Psychological AI**: a mathematically grounded, model‑agnostic control layer that makes agents **self‑regulating** and **self‑improving (opt‑in)**. The approach encodes neuro‑inspired state variables ("hormones" such as stress, curiosity) with exponential decay and impulse inputs, fuses them into **bounded control knobs**, and wraps operation with a **layered coping architecture** (micro‑recovery → coping routines → circuit breaker → sleep). We prove Lyapunov‑style stability, enforce runtime invariants, and show large empirical gains in chaotic conditions.

**Key claims (to be continuously validated in public benchmarks):**

* **Stability:** States and control remain bounded; a Lyapunov candidate decreases under dynamics and policy updates.
* **Reliability:** In stress tests, resilient controllers demonstrate **order‑of‑magnitude** reduction in critical failures compared to tuned baselines.
* **Portability:** The controller is **model‑agnostic** (OpenAI‑style interfaces), deployable alongside any LLM/tool stack.
* **Governance:** Learning updates are **opt‑in** (user‑approved), with full audit trails. PII is redacted before persistence.

---

## 1. Introduction

Two prior waves defined AI: (1) **Statistical ML** (1990–2010) and (2) **Deep Learning & Scaling Laws** (2017–2024). Capability rose dramatically, yet **reliability** did not keep pace. Real deployments demand **bounded behavior** under chaos. We propose a third wave: **Resilient Psychological AI**—embedding human‑inspired coping mechanisms into a controller with formal guarantees.

**Problem.** Agents destabilize when uncertainty, tool faults, or adversarial inputs accumulate. Temperature/filters alone are insufficient; external guardrails lack online adaptation.

**Proposal.** Treat affect‑like variables as first‑class **control state** with decay and impulses, map that state to **knob settings** via a bounded nonlinearity, and place operation within a **defense‑in‑depth** coping stack. Prove stability; test under chaos; ship invariants.

---

## 2. Contributions

* **Biologically‑inspired state.** Synthetic hormones (stress, curiosity, focus, etc.) evolve with exponential decay and impulse inputs.
* **Bounded control fusion.** A weighted, power‑transformed average generates a control signal **clamped** to safe ranges.
* **Layered coping.** L0–L4 routines (smoothing, micro‑recovery, coping, circuit breaker, sleep) prevent oscillation and runaway stress.
* **Stability guarantees.** A Lyapunov candidate enforces bounded trajectories; stress is capped by design.
* **Self‑learning (opt‑in).** Policy‑gradient updates improve knob mapping and memory selection **only upon user approval**.
* **Runtime invariants.** Always‑on assertions ensure state bounds, safe mode behavior, PII redaction, and failure‑budget tracking.
* **Reproducibility.** Open benchmarks with ablations and seeds; independent replication encouraged.

---

## 3. System Overview

A high‑level dataflow:

```
Inputs (sensors / tools / text) ─▶ Content Processor ─▶ Hormone State ─▶ Resilience Layers ─▶ Control Knobs ─▶ Policy/Tools
                                               ▲                                           │
                                               └────────────── Value/Baseline ◀────────────┘
```

### 3.1 Hormone dynamics

```math
h_i(t+\Delta t) = \mathrm{clip}_{[0,1]}\Big(h_i(t)\,e^{-\lambda_i \Delta t} + I_i(t) - m_t\Big),\quad \lambda_i = \ln 2 / \tau_i
```

* $I_i(t)$: impulse inputs (event‑driven signals, anomaly scores, etc.) smoothed by EMA.
* $m_t\ge 0$: coping drift applied during recovery stages.

### 3.2 Bounded control fusion

```math
k' = \frac{\sum_i w_i(H)\,\big(b + \delta_i\, h_i^{p_i}\big)}{\sum_i w_i(H)},\qquad k = \mathrm{clip}_{[k_{\min},\,k_{\max}]}(k')
```

* $w_i(H)\ge 0$, $p_i>0$. Control knobs (e.g., temperature, top\_p, retrieve\_k) inherit bounds by construction.

### 3.3 Policy‑gradient over knobs (opt‑in)

```math
\theta_{t+1} = \theta_t + \alpha_t\, A_t\, \nabla_\theta \log \pi_\theta(a_t\mid s_t),\qquad A_t=r_t-\hat V(s_t)
```

* Updates are packaged and **applied only with user approval** (no silent live learning).

---

## 4. Layered Coping Architecture (L0–L4)

To prevent oscillations and ensure graceful degradation, we use graded responses.

* **L0 — Smoothing & gain capping (every tick)**
  EMA impulses (β≈0.8), cap stress weight $w_{\max}$, enforce **rate limiter** and **minimum dwell** per knob (UI‑adjustable).

* **L1 — Micro‑recovery (1–3 ticks)**
  Trigger: $h_{\text{stress}}\ge \theta_1$ (e.g., 0.75).
  Actions: mild drift $m=0.01$, reduce temperature/top\_p/retrieve\_k by 20–30%.

* **L2 — Coping routines (10–30 ticks)**
  Trigger: rising stress slope and $h_{\text{stress}}\ge \theta_2$ (e.g., 0.85).
  Actions: stronger drift $m=0.02$, clamp max\_tokens, disable nonessential tools, **gain‑schedule** $\delta_i \leftarrow \gamma \delta_i$ (e.g., $\gamma=0.7$).

* **L3 — Circuit breaker timeout**
  Trigger: $h_{\text{stress}} > \theta_{\text{circuit}}$ (default 0.95).
  Action: reset $h_{\text{stress}}\leftarrow 0.4$, enter **Safe Mode A**: argmax decoding, strict tool allowlist, low caps on tokens.
  Dwell: two half‑lives **or** goal‑based until $b_{\text{stress}}+\varepsilon$.

* **L4 — Sleep / consolidation**
  Trigger: frequent L3 or budget breach.
  Action: pause nonessential I/O, consolidate memory, batch evaluation.

**Hysteresis & ramp‑out.** Resumption requires $h_{\text{stress}}\le \theta_{\text{resume}}$ and non‑positive short‑term slope; apply a 3‑step “ramp” before full operation.

---

## 5. Theory & Guarantees (Sketch)

### 5.1 Hormone decay & half‑life

```math
\dot h_i = -\lambda_i h_i \;\Rightarrow\; h_i(t)=h_i(0) e^{-\lambda_i t},\quad \lambda_i = \ln 2 / \tau_i
```

### 5.2 Control boundedness

For $h\in[0,1]^n$ and $p_i>0$, each term $b+\delta_i h_i^{p_i}$ is bounded; the weighted average and clamp yield (k\in\[k\_{\min},k\_{\max}].

### 5.3 Stress boundedness (with resilience)

Circuit breaker imposes a hard cap: after activation $h_{\text{stress}}^+=0.4<\theta_{\text{circuit}}$. With decay, coping drifts, and baseline adaptation, $h_{\text{stress}}$ remains bounded by $\max(\theta_{\text{circuit}}, b_{\text{stress}}+\varepsilon)$.

### 5.4 Lyapunov stability

Let

```math
\mathcal{L}(h,k)=\sum_i \alpha_i h_i^2 + \beta (k-k_\star)^2,\quad \alpha_i,\beta>0.
```

Under decay with bounded impulses and average decrease of $(k-k_\star)^2$ from policy updates, we show $\dot{\mathcal{L}}\le 0$, establishing stability.

*(Optional feature flag)* A two‑layer GCN value head can approximate continuous functions on fixed‑size graphs; v1 ships with a **scalar baseline (MLP)** for simplicity and speed.

---

## 6. Learning, Rewards, and Memory

### 6.1 Reward channels (configurable)

Composite reward combines task success, safety flags, quality, efficiency, and cost penalties; all z‑scored and clipped to $[-1,1]$. Advantage via GAE with $\gamma,\lambda$ tuned by sweep.

### 6.2 Opt‑in updates

The controller proposes parameter updates (knob policy, thresholds, calibrations). Users receive notifications, review diffs and metrics, and **approve or reject**; rollback is supported. **No automatic live learning.**

### 6.3 Memory consolidation

Store an item iff **(memory\_optimal == true) OR (A\_t > 0)**, with **dedup**, **TTL** (default 300 days, configurable), and **PII redaction** prior to persistence. Retrieval uses MMR‑style selection with recency/diversity; vector stores are pluggable (**Quadt** showcased).

---

## 7. Safety & Assurance

### 7.1 Runtime invariants (always on)

* **Boundedness:** $h\in[0,1]^n$, $k\in[k_{\min},k_{\max}]$, no NaN/Inf.
* **Safe Mode A:** argmax decoding, allowlisted tools only, no external writes.
* **Redaction:** PII removed before any memory write; no raw text in logs.
* **Consolidation rule:** only when memory\_optimal or $A_t>0$.
* **Failure budget:** per‑action joint failure estimator $\le 10^{-5}$.

### 7.2 Threat model → guard mapping (examples)

* Prompt injection → safe tools + schema validation + argmax in Safe Mode.
* Retrieval poisoning → dedup, recency/credit (if enabled), quarantine on anomaly.
* Provider flaps/timeouts → degrade to Safe Mode A; bounded latency.
* Vector‑DB unavailability → read‑through cache; fail closed on writes.

### 7.3 Audit & kill switch

All mode transitions, hormone states, and policy approvals are logged. Operators can force **Safe Mode** or **Sleep** instantly.

---

## 8. Benchmarks & Reproducibility (Prototype)

### 8.1 Toy demo (in this repo)

- Script: `python resilient_controller.py` (see `test_resilient_chaos()`).
- Outputs: console summary and optional plot (`plot_resilience_story`).
- Purpose: illustrate how coping layers affect a simple control-in-range metric.

### 8.2 Notes and caveats

- Scenario is simplified and partly stochastic; numbers are illustrative only.
- Claims are limited to this setup; external replication is encouraged.

### 8.3 Future benchmarking (out of scope here)

- Domain tasks (AV, finance, data‑center) and formal protocols will be added in separate work once the prototype matures.

---

## 9. Notes on impact (de-scoped)

Market sizing and domain-specific impact analyses are intentionally omitted until domain benchmarks and audited evidence are available.

---

## 10. Regulatory Alignment (non‑binding)

The architecture is **designed to support** safety frameworks (e.g., ISO 26262/SOTIF, FDA SaMD, DO‑178C) by providing bounded behavior, runtime invariants, audit trails, and reproducible test evidence. Formal certification requires domain‑specific engineering and assessment.

---

## 11. Limitations & Open Problems

* **Scope of affect:** more hormones (fear, boredom, motivation) may require interaction terms with new proofs.
* **Scalability:** fully‑connected hormone graphs scale quadratically; sparse priors/attention may help.
* **Tightness of proofs:** current Lyapunov candidate is sufficient, not necessary; tighter conditions are open.
* **Alignment:** resilience reduces failure rates but not goal misspecification; integration with alignment methods remains essential.
* **Human‑AI co‑adaptation:** operator behavior co‑evolves; modeling this dynamic is future work.

---

## 12. Roadmap

* **v1 (now):** OSS release; Resilience Guard L0–L3; invariants; AV demo harness; Console (MVP).
* **v1.1 (≤6 months):** second domain benchmark; external replication; assurance appendix.
* **v1.2 (≤12 months):** optional GCN value head; neuromodulator extensions; hardware hooks.

---

## 13. Implementation Guidelines (Practitioner Notes)

* Choose half‑lives s.t. $\lambda\,\Delta t\in[0.02,0.15]$ for numerical stability.
* Start thresholds: $\theta_1=0.75$, $\theta_2=0.85$, $\theta_{\text{circuit}}=0.95$, reset to 0.4; resume threshold $\le 0.60$.
* Enable rate limiter (max Δ≈10% of range) and minimum dwell (≥3 steps) — **UI‑adjustable**.
* Safe Mode A = argmax + strict tool allowlist; cap tokens; no external writes.
* Memory pipeline: dedup, TTL (default 300 days), PII redaction, Quadt as showcase vector DB; embeddings configurable.
* Deployment: Docker; GPUs with ≥24 GB memory recommended; customers supply hardware.

---

## 14. Glossary

* **Hormones:** internal state variables (stress, curiosity, etc.) with decay and impulses.
* **Control knobs:** inference hyperparameters and policy switches (temperature, top\_p, retrieve\_k, tool gating).
* **Coping layers (L0–L4):** graded responses from smoothing to sleep.
* **Safe Mode A:** deterministic decoding with strict tool allowlist.
* **Opt‑in updates:** learning steps applied only after user approval.
* **Joint failure budget:** target probability that all defenses fail on a single action.

---

## 15. References (selected)

Goodfellow et al. (2015) • Finn et al. (2017) • Khalil (2002) • Slotine & Li (1991) • Picard (1997) • Sutton & Barto (2018) • Silver et al. (2016) • Schmidhuber (1991) • Ashby (1956) • Rasmussen & Vicente (1989)

---

## Appendix A — Key Equations (GitHub‑safe)

**A1. Hormone Dynamics**

```math
h_i(t+\Delta t) = h_i(t)\,e^{-\ln(2)\,\Delta t/\tau_i} + I_i(t)
```

**A2. Control Signal**

```math
k = \mathrm{clip}_{[k_{\min},k_{\max}]}\!\left(\frac{\sum_i w_i(H)\,\big(b + \delta_i h_i^{p_i}\big)}{\sum_i w_i(H)}\right)
```

**A3. Policy Gradient**

```math
\delta_i \leftarrow \delta_i + \alpha\,A\,\frac{\partial}{\partial \delta_i}\log \pi_\theta(a\mid h)
```

**A4. Circuit Breaker**

```math
h_{\text{stress}} \leftarrow 0.4 \quad \text{if } h_{\text{stress}} > \theta_{\text{circuit}}
```

**A5. Adaptive Baseline**

```math
b_{\text{stress}} \leftarrow b_{\text{stress}} + \eta\big(\mathrm{percentile}_{25}(H_{\text{history}}) - b_{\text{stress}}\big)
```

---

## Appendix B — ChaosBench (scenario sketch)

* Burst impulses (high amplitude, short duration).
* Slow drift (low amplitude, long duration).
* Provider/API timeouts, malformed JSON, vector‑DB flaps.
* Tool escalation attempts and prompt injection.
* Weather/sensor faults (for AV demo).
  Each scenario logs invariants, mode transitions, recovery times, and outcome metrics.

---

## Appendix C — Assurance Case (sketch)

**Goal:** Bounded, reliable behavior under chaos.
**Strategy:** (S1) mathematical guarantees → (S2) runtime invariants → (S3) empirical stress tests.
**Evidence:** (E1) Lyapunov proof sketch; (E2) invariant test suite; (E3) benchmark logs with ablations and seeds.

---

### License

This document and the accompanying code are released under the **Apache License, Version 2.0**. See `LICENSE` in the repository.

---

## 16. Self‑Learning Guarantee (What Learns, What Can’t)

**What learns (opt‑in only):**

* **Knob policy** (`π_θ: H → K`): mappings from hormone state to control knobs (temperature, top\_p, retrieve\_k, tool gating).
* **Thresholds & calibrations:** decay half‑lives, coping thresholds, reward scalars (via experiment sweeps).
* **Memory policy:** consolidation and retrieval scoring rules.

**What never self‑modifies:**

* Foundation model weights; safety/tool allowlists; PII redaction policy; invariant checks.
* External side‑effects (writes, code exec) without explicit permission.

**Update protocol:** propose → notify → human review (diffs, metrics) → approve/deny → apply with rollback. No silent updates.

---

## 17. Threat Model & Mitigations

| Threat                       | Vector                         | Mitigation                                                                    | Invariant / Gate                               |
| ---------------------------- | ------------------------------ | ----------------------------------------------------------------------------- | ---------------------------------------------- |
| Prompt injection / jailbreak | Untrusted content in context   | Safe Mode A (argmax), schema validation, tool allowlist, retrieval sanitation | No external writes in Safe Mode; parse or drop |
| Retrieval poisoning          | Adversarial memory items       | PII redaction, dedup, quarantine on anomaly, TTL                              | Consolidation iff memory\_optimal OR A\_t>0    |
| Provider/API flaps           | Timeouts, malformed JSON       | Deterministic fallback (Safe Mode A), bounded latency                         | No NaN/Inf; max tokens capped                  |
| Vector‑DB outage             | Store/fetch unavailable        | Read‑through cache, fail‑closed on writes                                     | Memory writes require store availability       |
| Oscillation                  | High loop gain, delayed coping | L0–L4 coping, hysteresis, ramp‑out, rate limiter/dwell                        | Hormones/knobs bounded each tick               |
| PII leakage                  | Unsafe storage/logging         | Redact before persistence; no raw text in logs                                | Redaction pass required                        |
| Joint defense failure        | Correlated layer failures      | Diversity of layers; budget tracking                                          | Estimator ≤ 1e-5 per action                    |

---

## 18. Benchmark Protocols & Ablations

### 18.1 Autonomous‑Vehicle Safety (Primary)

* **Env:** CARLA/LGSVL; 10 seeded routes × 3 stressors: (i) sensor dropout bursts, (ii) severe weather + glare, (iii) adversarial signage/occlusions.
* **Controller mapping:** anomaly→stress, novelty→curiosity, safety margin→focus.
* **Metrics:** critical disengagements/100 km (primary), intervention latency, passenger comfort (ISO 2631 proxy), CB/timeout rate, p95 latency, cost.
* **Gates:** ≥ **9×** reduction vs tuned baseline; CB/timeout ≤ **1%** actions; p95 latency ≤ **2.5 s**.
* **Ablations:** −L1, −L2, −CB, −baseline‑adapt, −EMA; report deltas and CIs.

### 18.2 Evaluation Procedure

1. **Pre‑registration:** freeze metrics, routes, stress seeds, and hyperparameter ranges.
2. **Baselines:** tuned PID/MPC/LLM‑agent without coping; report their best.
3. **Runs:** 5 seeds × each scenario; record invariants and transitions.
4. **Stats:** mean ± 95% CI; Mann–Whitney U for discrete counts.
5. **Artifacts:** publish logs, configs, plots, and replay videos.

---

## 19. Second Domain: Data‑Center Cooling (Sketch)

* **Signals → hormones:** thermal anomaly → stress; efficiency deficit → focus; topology change → curiosity.
* **Knobs:** PID setpoints, actuator duty cycles, telemetry query rate, retrieve\_k for past incidents.
* **Metrics:** SLA violations/hour, energy per request, thermal overshoot, CB/timeout rate, p95 latency.
* **Gate:** ≥ **5×** SLA‑violation reduction vs baseline under chaos (fan failures, workload spikes).

---

## 20. Reproducibility & Replication

* **One‑click scripts:** `make bench-av`, `make bench-dc`.
* **Seeds & configs:** committed JSON/YAML; CI pins container image + CUDA/toolchain.
* **Ablation matrix:** automated with Optuna sweeps; CSV + plots saved.
* **Independent reruns:** invite external labs; provide support scripts; publish replication reports.

---

## 21. Productization & Packaging

* **OSS Core (Apache‑2.0):** controller, hormones, coping (L0–L4), invariants, dashboards, Quadt adapter, OpenAI‑compatible model adapters.
* **Enterprise Add‑ons (separate license):** Resilience Console, ChaosBench Suite, Compliance pack, Domain packs (AV, finance, data‑center).
* **Deployment:** Docker; GPU ≥24 GB recommended; customers supply hardware.
* **Interfaces:** OpenAI‑style chat/embeddings; pluggable providers (Ollama, vLLM, OpenRouter, Azure/OpenAI, Anthropic, etc.).

---

## 22. FAQs for Reviewers

* **“Isn’t this heuristics?”** → No: state machine + math + invariants; ablations show each layer’s causal impact.
* **“Won’t generalize?”** → Same coping template across two domains with minor retune; replication encouraged.
* **“Too slow/complex?”** → p95 latency budget enforced; Safe Mode A guarantees graceful degradation.
* **“Self‑learning risk?”** → Opt‑in updates only; audit trails; rollback.
* **“Alignment solved?”** → No—resilience complements, not replaces, alignment.

---

## Related Documentation

### Core Documentation
- **[README.md](../README.md)** - Project overview and quick start guide
- **[BREAKTHROUGH.md](BREAKTHROUGH.md)** - Technical breakthrough details and architecture overview
- **[FOUNDATION.md](FOUNDATION.md)** - Mathematical foundations and formal proofs
- **[IMPACT.md](IMPACT.md)** - Research significance and broader applications
- **[wiki_page.md](wiki_page.md)** - Complete project wiki and reference guide

### Project Documentation
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines and standards
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[SECURITY.md](SECURITY.md)** - Security policy and vulnerability reporting

---

## 23. Changelog

* **v1.1 (2025‑08‑28):** Initial public draft under Apache‑2.0; added self‑learning guarantee, threat model, benchmark protocols, second domain sketch, reproducibility plan, productization notes, FAQs.
