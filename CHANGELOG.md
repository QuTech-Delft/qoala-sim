CHANGELOG
=========

2026-09-08 (2.0.1)
-------------------

Attribution is given per entry as *(author, PR)*.

### Fixed

- **Local routines could run on a qubit belonging to an unrelated block.**
  `QpuScheduler.are_resources_available` treated any already-allocated virtual
  qubit as usable by a `LocalRoutineTask`. When a program reuses a qubit slot —
  which the compiler's block reordering introduces as soon as a measured qubit
  frees its slot for a later round — a routine could therefore be scheduled on
  a qubit holding another block's state, silently producing wrong results.

  The scheduler now walks the full task graph and only admits an
  already-allocated qubit if the block that allocated it is a transitive
  ancestor of the task; otherwise the task waits. This is what the
  `set_allocating_block` bookkeeping and `_get_ancestor_blocks` were added for
  in 2.0.0 — the check that consumes them was missing, leaving both unused.

  Where the full task graph is not available (a manually constructed
  scheduler, as opposed to one set up by the node scheduler), ancestry cannot
  be determined and the previous plain availability check still applies.

  > **This changes simulation output**, not just internals. Programs that reuse
  > qubit slots schedule differently and now report different — correct —
  > success rates, so results produced with 2.0.0 and 2.0.1 must not be mixed.
  > Measured on a streaming BQC client/server pair under noiseless hardware
  > parameters: the optimized variant went from 44% to 100% success at n=3 and
  > from 57% to 100% at n=5, while the unoptimized variant was 100% throughout
  > (it allocates a fresh slot per round, so it never hit the bug).
  *(@spoukke)*

2026-09-07 (2.0.0)
-------------------

Attribution is given per entry as *(author, PR)*. Contributors in this release:
@bvdvecht, @dieriver, @sampl0, @spoukke, @DavideFrr.

### Breaking changes

- **Block header annotations are now separated by `;` instead of `,`.** This is
  a source-language change: every existing `.iqoala` program with more than one
  block annotation must be updated.
  *(@spoukke, #50)*

  ```
  ^b1 {type = CL, deadlines = [b0: 100]}:   # 1.0.0
  ^b1 {type = CL; deadlines = [b0: 100]}:   # now
  ```

  The change was needed because the new annotations contain comma-separated
  lists (`dependencies = [b1, b2]`), so the top-level separator had to move.
  Commas are still used *inside* bracketed lists (`deadlines = [b0: 100, b1: 20]`,
  `predecessors = [b0, b1]`) and in the `META_START` section, which is
  unchanged.

  Blocks carrying only a type (`^b0 {type = CL}:`) are unaffected. A file using
  the old separator fails loudly, but with a misleading message —
  `QoalaParseError: Block type annotation must have exactly one '='.` — because
  the whole annotation string is treated as the type annotation.

- **Python 3.8 and 3.9 are no longer supported.** `python_requires` is now
  `>=3.10, <3.13`.
  *(@bvdvecht; @dieriver, #51)*
- **Dependency pins changed**, most notably `netqasm ==0.13.0a7` →
  `netqasm >=2.0, <3.0` (a major-version jump) and `numpy >=1.22, <1.23` →
  `numpy >=1.26, <2.0.0`. Code that imports `netqasm` directly alongside qoala
  may need updating.
  *(@dieriver, #51)*
- **`qoala.sim.scheduler` has been removed.** Its contents now live in the
  `qoala.sim.scheduling` package:
  *(@bvdvecht, #46)*

  | Old | New |
  | --- | --- |
  | `qoala.sim.scheduler.NodeScheduler` | `qoala.sim.scheduling.nodesched.NodeScheduler` |
  | `qoala.sim.scheduler.NodeSchedulerComponent` | `qoala.sim.scheduling.nodeschedcomp.NodeSchedulerComponent` |
  | `qoala.sim.scheduler.ProcessorScheduler` | `qoala.sim.scheduling.procsched.ProcessorScheduler` |
  | `qoala.sim.scheduler.ProcessorSchedulerComponent` | `qoala.sim.scheduling.procsched.ProcessorSchedulerComponent` |
  | `qoala.sim.scheduler.Status`, `SchedulerStatus` | `qoala.sim.scheduling.procsched` |
  | `qoala.sim.scheduler.CpuScheduler`, `CpuEdfScheduler`, `CpuFcfsScheduler` | `qoala.sim.scheduling.cpusched` |
  | `qoala.sim.scheduler.QpuScheduler` | `qoala.sim.scheduling.qpusched` |

- **Task builders moved out of `qoala.runtime.task`** into
  `qoala.runtime.taskbuilder`: `TaskGraphBuilder`, `TaskDurationEstimator`,
  `TaskGraphFromBlockBuilder` and `QoalaGraphFromProgramBuilder`. `TaskGraph`
  itself stays in `qoala.runtime.task`.
  *(@bvdvecht, #46)*
- **`NodeScheduler` is no longer directly instantiable as a working scheduler.**
  It has no `run()` method; use `OnlineNodeScheduler` or `StaticNodeScheduler`.
  `ProcNode` picks between them automatically based on `is_predictable`, so
  only code that constructed a `NodeScheduler` by hand is affected.
  *(@bvdvecht, #46)*
- **Time bins are now half-open.** An `ExplicitTimebin` includes its start time
  and *excludes* its end time, and the bin-end event fires at `end - 1`.
  Simulation timings involving network schedules shift by one tick, so tests or
  analyses asserting exact end-of-bin timestamps need updating.
  *(@bvdvecht, #43)*
- **Makefile `all-tests` changed meaning.** The coverage-based parallel run was
  renamed to the `tests` target *(@spoukke, #50)*, which left `verify` referring
  to a target that no longer existed; `all-tests` was then reintroduced as an
  alias for `unit-tests integration-tests` *(@dieriver, #51)*. Scripts calling
  `make all-tests` still succeed but no longer produce coverage.

### Added

- **Critical sections.** Blocks can be grouped into critical sections that are
  executed atomically. Programs declare them in the `META_START` header
  (`critical_sections: 1 -> AE`) and blocks join one via the
  `critical_section=<id>` block annotation. Three types are supported
  (`qoala/lang/program.py`): `A` (atomic allocation), `E` (atomic execution)
  and `AE` (both). The processor schedulers track the currently active section
  via `ActiveCriticalSection` and will not interleave tasks from other
  processes into it.
  *(@bvdvecht, #46)*
- **Block precedence annotations.** Blocks may carry explicit
  `predecessors`, `dependencies`, `prev_comm` and `prev_ent` annotations. When
  present, `QoalaGraphFromProgramBuilder` builds the task graph from these
  instead of deriving a linear chain, which lets a compiler emit reordered /
  pipelined block schedules. The four annotations must be given together or
  not at all; the parser rejects a partial set.
  *(@spoukke, #50)*
- **Conditional branch cancellation.** When a CL block takes a branch, the CPU
  scheduler now cancels the tasks of the not-taken blocks before the branch
  block is removed from the graph, so conditional blocks never become eligible
  roots. Cancellation follows predecessor/dependency edges only, so
  independent "floater" blocks (e.g. pipelined QC requests placed by a
  block-reordering pass) are left alone, and qubits allocated by a cancelled
  block are released. The memory manager tracks the allocating block per
  `(pid, virt_id)` to support this.
  *(@spoukke, #50)*
- **New classical host instructions:** `copy_cval` (`CopyCValueOp`),
  `sub_cval_c` (`SubCValueOp`) and `mult_cval` (`MultiplyCValueOp`).
  *(@spoukke, #50)*
- **Compiler-generated identifiers.** `%<number>` (e.g. `%1`, `%42`) is now a
  valid variable name, and routine names are validated against a rule set
  (identifier-shaped, not a Python keyword, no clash with host operation
  names) so compiler-mangled names such as `__epr_gen` parse.
  *(@spoukke, #50)*
- **Imperfect link configuration.** New `NetworkConfig.from_nodes_imperfect_links`
  plus topology helpers in `qoala/runtime/config.py`:
  `uniform_t1t2_qubits_uniform_imperfect_gates`,
  `..._imperfect_gates_limited_comm`,
  `..._uniform_single_gate_duration_and_noise`,
  `..._uniform_any_gate_duration_and_noise` and
  `..._any_gate_duration_and_noise_limited_comm`.
  *(@spoukke, #50)*
- **Empty deadline lists.** `deadlines=[]` is accepted by the parser.
  *(@spoukke, #50)*
- **Examples:** a 10-qubit BQC application and a 10-qubit BQC application
  executing a CNOT (`examples/bqc_cnot/`).
  *(@sampl0, #45)*
- **Example:** teleport using the new precedence annotations
  (`examples/teleport/example_teleport_with_precedence.py`).
  *(@spoukke, #50)*
- **Tests:** integration tests for a VBQC program with multipair requests, with
  and without a network schedule.
  *(@sampl0, #44)*

### Changed

- **Block header annotation separator changed from `,` to `;`** *(breaking)*,
  to make room for the comma-separated lists used by the new precedence
  annotations. See *Breaking changes* above.
  *(@spoukke, #50)*
- **Scheduler module reorganised** *(breaking)*. The monolithic
  `qoala/sim/scheduler.py` (~1500 lines) is replaced by the
  `qoala/sim/scheduling/` package: `procsched.py` (shared
  `ProcessorScheduler`), `cpusched.py` (`CpuScheduler`, `CpuEdfScheduler`,
  `CpuFcfsScheduler`), `qpusched.py`, `nodesched.py` and the new
  interface/component/message modules.
  *(@bvdvecht, #46)*
- **`NodeScheduler` split** *(breaking)* into `OnlineNodeScheduler` and
  `StaticNodeScheduler`, and its main loop simplified.
  *(@bvdvecht, #46)*
- **Task builders extracted** *(breaking)* from `qoala/runtime/task.py` into the
  new `qoala/runtime/taskbuilder.py`.
  *(@bvdvecht, #46)*
- **EntDist rewritten.** Message handling, delivery and time-bin end are now
  separate handlers (`_handle_messages`, `_handle_delivery`, `_handle_bin_end`),
  with `schedule_deliveries` and `get_all_joint_requests` replacing the previous
  inline logic. Entanglement can now be generated in the middle of a time bin.
  *(@bvdvecht, #43)*
- **Time bins are half-open** *(breaking)*. An `ExplicitTimebin` now includes
  its start time and *excludes* its end time; the bin-end event fires at
  `end - 1` and no entanglement requests are sent on the last tick of a bin.
  *(@bvdvecht, #43)*
- **Task-completion notifications are FIFO.** Processor schedulers report
  completion with a `TaskFinishedMsg` over the node-scheduler port instead of a
  signal, giving deterministic ordering.
  *(@spoukke, #50)*
- **Python support is now 3.10–3.12** *(breaking)*
  (`python_requires = >=3.10, <3.13`); 3.8 and 3.9 are dropped. Dependencies
  loosened accordingly: `numpy >=1.26,<2.0`, `netqasm >=2.0,<3.0`,
  `pydynaa >=0.3.0,<2.0.0`.
  *(@bvdvecht; @dieriver, #51)*
- **CI** runs unit tests, integration tests and examples across 3.10 / 3.11 /
  3.12, with lint and mypy as gating jobs. The publish workflow now only runs
  when a tag is pushed.
  *(@dieriver, #51)*
- The Makefile coverage run moved from the `all-tests` target to `tests`
  *(@spoukke, #50)*; `all-tests` is now an alias for
  `unit-tests integration-tests` *(@dieriver, #51)*. *(Breaking for scripts
  calling it — see Breaking changes above.)*
- The Makefile `verify` target works again and `clean` also removes `build/`,
  `dist/` and logs.
  *(@dieriver, #51)*
- The Makefile `mypy` target runs with `--check-untyped-defs`.
  *(@dieriver, #47)*

### Fixed

- **Scheduler stall when waiting on peer messages.** The CPU scheduler now
  yields only on message events from the peers it is actually waiting for,
  rather than on all peers, and each peer is listed only once.
  *(@dieriver, #47)*
- **Sibling else-branches not cancelled.** A block ending in a forward jump
  that skips its sibling else-branches left those branches schedulable, because
  they name the branch *source* — not the taken block — as their predecessor.
  The cancellation set is now seeded with the current block's predecessors.
  *(@spoukke)*
- **Duplicate multipair execution.** A multipair task could be executed
  multiple times within a single timestep; failure handling now frees only the
  qubit belonging to the failed request instead of aborting the whole batch.
  *(@sampl0, #42)*
- **Too many joint requests for multipair.** `EntDist` created excess
  `JointRequest`s when a node issued multiple requests with the same PID.
  *(@sampl0, #44)*
- **`IqoalaTuple` round-trip.** `IqoalaTuple.__str__` emitted `,` between tuple
  values while the parser expected `;`, so a printed tuple could not be parsed
  back. `__str__` now emits `;`. (Issue #39.)
  *(@DavideFrr, #40)*
- **`TypeError` in the CPU scheduler on Python 3.11+.** When a message is
  already buffered, `get_evexpr_for_msg_from` returns `None`, and the scheduler
  evaluated `None | ev_expr`. That yielded an `EventExpression` on 3.10 but
  raises `TypeError: unsupported operand type(s) for |: 'NoneType' and
  'pydynaa.core.EventExpression'` on 3.11 and 3.12. The scheduler now skips the
  `|` when there is no message event. Without this, three teleport tests fail on
  3.11/3.12.
  *(@dieriver, #51)*
- **Python-version-specific dictionary combination** in the BQC examples.
  *(@sampl0, #45)*
- Entanglement generation duration in the `three_nodes` example.
  *(@bvdvecht, #43)*
- **Flaky noisy tests and examples.** `test_noisy_qkd`, `test_noisy_singlenode`
  and their near-identical counterparts `example_noisy_qkd` and
  `example_noisy_singlenode` assert on sampled quantities — the QKD duration
  bound is documented as a 99% confidence interval — but seeded no RNG, so each
  failed roughly one run in a hundred. All four now pin `random` and
  `numpy.random`. Seeding netsquid alone is not enough: `ns.set_random_state`
  reseeds only netsquid's own generator, while the depolarise link samples
  entanglement attempts from numpy's global RNG.
  *(@spoukke)*

2024-09-13 (1.0.0)
-------------------
- First public release.
