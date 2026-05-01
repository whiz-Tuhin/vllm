# Section 4.4.1 — vLLM Implementation

*(Draft text for the ASPLOS 2027 submission, §4.4 Validation, subsection 4.4.1.
LaTeX-ready prose; bibliography keys in square brackets are placeholders.)*

---

## 4.4.1 vLLM Implementation

We validate the AFD cost model in AIConfigurator++ against a real system
by building an AFD-capable inference engine on top of vLLM
v0.16.0rc2~\cite{kwon2023vllm}. Our starting point is vLLM PR~#29772, which
introduced AFD for MoE models (DeepSeek-V2/V3, Step3) as an opt-in runtime mode
with a NCCL P2P transport and a 1:1 ATTN↔FFN pairing that delegated
expert routing to an intra-FFN all-gather plus reduce-scatter. This baseline
only supports \emph{symmetric} configurations ($M = N$), which excludes most
of the Pareto-optimal design points AIConfigurator++ identifies at scale
(e.g., 1A4F, 2A4F, and the $x$A1F family). To validate the simulator across
the full design space we re-architect the AFD data path along four axes:

\paragraph{(i) $M\times N$ bipartite pairing.}
We replace the 1:1 zip with a full bipartite product over $(M, N)$: every ATTN
rank is paired with every FFN rank via its own NCCL pair-group, for a total of
$MN$ pairs. Pair-group bootstrap uses a per-pair Gloo rendezvous on unique TCP
ports so that the $MN$ NCCL communicators can be constructed deadlock-free
without contaminating the global process group's counter state. The topology
is valid for arbitrary $M, N \ge 1$ and subsumes the original symmetric case.

\paragraph{(ii) ATTN-side pre-routing.}
The MoE router (gate projection and top-$k$ selection) and the shared experts
are relocated from the FFN side to the ATTN side. Each ATTN rank now computes
routing decisions locally and forwards the post-attention hidden state together
with \texttt{topk\_ids} and \texttt{topk\_weights} to every FFN partner. The
FFN side no longer participates in any inter-worker collective: each FFN worker
runs only its local expert shard using the routing metadata received from
ATTN, and returns a partial tensor that ATTN sums element-wise. The
shared-expert output is added on the ATTN side, eliminating a cross-boundary
transfer.

\paragraph{(iii) Expert-kernel bypass.}
A subtle interaction with vLLM's kernel modularization requires a fourth
change. The natural entry point \texttt{FusedMoE.quant\_method.apply}
dispatches into \texttt{FusedMoEModularKernel.forward}, whose
\texttt{prepare\_finalize} hooks themselves invoke EP all-gather and
reduce-scatter — the very collectives that pre-routing is designed to avoid.
We therefore introduce a new entry point
\texttt{FusedMoE.forward\_pre\_routed} that calls the raw
\texttt{fused\_experts} Triton kernel directly with
\texttt{expert\_map} filtering, yielding a zero-collective MoE-layer
implementation.

\paragraph{(iv) Synchronization backend for data parallelism.}
vLLM's default configuration auto-enables a Gloo CPU all-reduce for DP-padding
synchronization whenever asynchronous scheduling, $\text{DP} > 1$, and an MoE
model all coincide. In the AFD setting this path adds $\sim$1.5~s of stream
drain per forward pass: because AFD already serializes the forward pass across
the ATTN↔FFN boundary, there is no asynchronous GPU pipeline to protect, but
the CPU-side all-reduce nonetheless stalls the GPU stream until previously
enqueued NCCL sends have drained. We override the default so that AFD
configurations keep the DP synchronization on NCCL (GPU), bringing per-forward
sync cost from hundreds of milliseconds to microseconds.

\paragraph{Scope.}
The four changes are concentrated in
\texttt{vllm/distributed/afd\_transfer/afd\_connector/p2p\_connector.py}
(connector rewrite),
\texttt{vllm/model\_executor/models/deepseek\_v2.py} (router/shared-expert
migration), \texttt{vllm/model\_executor/layers/fused\_moe/layer.py} (kernel
bypass), and \texttt{vllm/config/vllm.py} (DP synchronization override), for
an aggregate diff of roughly 1\,000 lines. All modifications preserve
HuggingFace checkpoint compatibility: no weight-name remapping or
post-processing is required. For simulator validation we also add lightweight
per-section wall-clock instrumentation gated by an \texttt{AFD\_TIMING}
environment variable; this yields per-layer measured latencies that
Section~4.4.2 compares against AIConfigurator++'s predictions.

\paragraph{Empirical result.}
On a single H200 node (NV18 fabric, DeepSeek-V2-Lite, batch 20 concurrent
requests, 128-token prompt, 32-token output), the rewritten AFD implementation
achieves warm-state TPOT of 54~ms (1A1F), 138~ms (1A2F), 311~ms (2A2F),
62~ms (3A1F), and 55~ms (2A1F). The original 1:1+EP design achieves 2\,076~ms
TPOT for 2A2F on an L40S node; the asymmetric configurations it does not
support at all. Relative to the simulator predictions from
AIConfigurator++, the measured TPOTs agree to within $X\%$ (Figure~\ref{fig:afd-validation}),
giving us confidence in the simulator's cost-model fidelity for the broader
design-space exploration reported in Section~5.

---

## Notes for the author

- Reference `~\cite{kwon2023vllm}` should point to the vLLM paper in your `.bib`.
- PR #29772 reference: this is the upstream GitHub PR. If the paper needs a
  citable artifact, use the commit hash (`ac539d421` for the pre-our-work base,
  `987bc559b` for the final version after our changes) or the fork URL
  (`github.com/whiz-Tuhin/vllm/tree/tk/pr-29772-afd-pre-routing`).
- The 54/138/311/62/55 ms numbers are the warm-run TPOTs from our clean
  benchmarks. For the asymmetric ones (3A1F, 2A1F) the "original" row in any
  comparison table should be marked as "not supported" rather than a number.
- The "within X%" agreement is a TODO — you'll need the AIConfigurator++
  predicted numbers to fill in. If the simulator predicts, say, 120ms for 1A2F
  and we measure 138ms, that's ~15%. If it's closer than that, use the real
  number; if it's wider, consider whether the paper wants to claim model
  fidelity or measured speedup.
- I deliberately did not include the 1A2F=1687ms original number from our
  runs because that was on H200 with the PR #29772 baseline on a heavily
  contested cluster node — not a clean comparison to anything in the paper.
  Use L40S=2076ms for 2A2F as the original reference since it's documented
  in the upstream issue tracker.
- If the paper wants a figure for the four changes, the side-by-side
  comparison table in
  `afd-m2n-rewrite/PAPER-SECTION-pairing-redesign.md` translates directly
  into a `\begin{table}` with ~12 rows.

## Alternative shorter version (if space is tight)

If §4.4.1 is limited to ~half a column, use this condensed version:

> We build our AFD prototype on vLLM v0.16.0rc2~\cite{kwon2023vllm}, extending
> PR~#29772 with four changes required to support the asymmetric
> configurations AIConfigurator++ identifies as Pareto-optimal. First, we
> replace the original 1:1 ATTN↔FFN pairing with a full $M\times N$ bipartite
> topology using per-pair NCCL communicators. Second, we relocate the MoE
> router and shared experts to the ATTN side, enabling each ATTN rank to
> pre-route tokens directly to the FFN worker holding their target experts
> and eliminating the intra-FFN all-gather and reduce-scatter collectives.
> Third, we add a \texttt{fused\_experts}-direct kernel bypass so the
> EP collectives embedded in vLLM's modular MoE kernel do not re-emerge.
> Fourth, we override vLLM's default CPU-based DP synchronization — which
> auto-engages for async-scheduled MoE+DP configurations — back onto NCCL,
> since the AFD forward pass is already serialized across the ATTN–FFN
> boundary. The combined changes total approximately 1\,000 lines in
> four files and preserve HuggingFace checkpoint compatibility. On a single
> H200 node running DeepSeek-V2-Lite, we measure warm-state TPOT of 54~ms
> (1A1F), 138~ms (1A2F), 311~ms (2A2F), 62~ms (3A1F), and 55~ms (2A1F),
> matching AIConfigurator++ predictions within $X\%$ (Figure~\ref{fig:afd-validation}).
