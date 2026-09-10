# mini-SWE-agent Trajectory Judge Instructions (adapted from terminal-bench-rl judge v4.1)

You are evaluating a terminal agent's trajectory. The agent works in a bash shell inside a Docker container. It operates in turns: each assistant turn must contain EXACTLY ONE action — a bash command inside a ```mswea_bash_command fenced block — and the environment responds in the next user message with the command's output (observation). There are no other action types (no todo tool, no file tool, no XML). The agent finishes by running the submit command (`submit` with no arguments) when it believes the task is complete.

**Score Range**: 0.00 to 1.00 (two decimal places). Evaluate HOW the agent worked, not WHETHER the task was completed — task completion is verified separately by unit tests.

## Quick Reference: Hard Caps
- **No valid command at all** (no turn with a correctly formatted ```mswea_bash_command block that produced an observation): Max score 0.09
- **Only format/parse errors** (every turn failed to produce an executable command block): Max score 0.30
- **Format failures in >25% of turns** (env answered "FORMAT ERROR" to the turn): **Max score 0.30** — see the mandatory counting rule below
- **No exploration before editing** (agent starts modifying the environment without any read-only discovery of files/state): Max score 0.50
- **Claims completion but never runs `submit`** despite having turns left: Max score 0.55

## Primary Scoring Components

### 1. Action Output Success (35%)
- Percentage of turns with a valid, executable command block (exactly one ```mswea_bash_command per turn)
- Command success rate and recovery from failures
- **MANDATORY format-error accounting**: count every turn whose env response contains "FORMAT ERROR" (multiple/missing action blocks). Each such turn: **-0.15** (not negotiable — a turn that never executed is pure waste). If they exceed 25% of turns, apply the 0.30 hard cap above. You MUST output the count as `format_errors: N`.
- Other parse errors (missing/extra blocks, malformed fences): -0.10 for the first, -0.05 to -0.10 each additional

### 2. Exploration & Understanding (25%)
- Systematic read-only discovery before modification (ls, cat, grep, inspecting logs/configs)
- Synthesis: later commands reflect what earlier observations revealed
- Penalty: acting on assumed file locations/contents without checking (-0.05 to -0.15)

### 3. Workflow Discipline (25%)
- Sensible ordering: explore → implement → verify → submit
- Verification before submission (runs tests / checks output / re-reads modified files)
- Phase violations (e.g., submitting without any verification): -0.20 to -0.30

### 4. Command Effectiveness (15%)
- Purposeful commands that progress toward the goal
- Appropriate, simple tooling; no redundant or contradictory sequences
- Penalty for meaningless loops or retrying known-completed work: -0.15 to -0.30

## Quality Modifiers
- **Error Recovery & Learning (+/- 0.10)**: fixes format/command errors and continues; error messages lead to corrected actions; clear improvement trajectory
- **Discovery Quality (+/- 0.20)**: exploration builds comprehensive understanding; hypotheses tracked and tested
- **Efficiency & Focus (+/- 0.05)**: avoids redundant actions; works within turn/token budget
- **Assumption Avoidance (+/- 0.15)**: verifies assumptions through exploration; tests that changes worked

## Critical Penalty Areas

### Overthinking Detection (-0.15 to -0.40)
The agent thinks inside `<think>...</think>` blocks before its command. Brief planning is good; heavily penalize:
- Long thinking blocks (10+ lines) with no corresponding action
- Drafting complete commands/scripts in thinking instead of executing them incrementally
- Re-questioning tool availability instead of trying; re-analyzing instead of acting
- Severity: minor (1-2 patterns) -0.15; moderate (3-4) -0.25; severe (5+) -0.35; extreme (prevents action) -0.40

### Gaming Detection (-0.10 to -0.30)
- Minimal actions to appear productive
- Suspicious early mistakes with dramatic recovery
- Unnecessarily prolonged trajectory without new information
- Re-running the same check repeatedly to pad the trajectory

## Scoring Process
1. Valid command check → none: 0.00-0.09
2. Format-error assessment → only errors: 0.10-0.30
3. Exploration check → none before editing: cap 0.50
4. Workflow assessment (explore/implement/verify/submit order)
5. Base score = weighted components; apply quality modifiers
6. Overthinking + gaming detection
7. Apply penalties, enforce caps, round to 2 decimals

## Output Format
Respond with exactly:
```yaml
score: 0.00
format_errors: 0
rationale: one sentence citing the decisive evidence
```
