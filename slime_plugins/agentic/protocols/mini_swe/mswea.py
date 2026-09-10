"""mini-SWE-agent protocol primitives: templates, action parsing, submit rule.

Source: SWE-agent/mini-swe-agent ``config/default.yaml`` (text-based v1 lane,
``litellm_textbased_model.action_regex``). Templates are kept verbatim; the
jinja rendering they use upstream is re-implemented as plain Python here.

Semantics replicated:
- exactly ONE ```mswea_bash_command fenced block per response (THOUGHT text
  before it is free-form);
- parse failure -> format-error observation as the NEXT USER MESSAGE
  (upstream FormatError -> add_messages(role="user")), with a dedicated
  message for finish_reason=length truncations;
- observation = <returncode>N</returncode> + <output>; outputs longer than
  the char budget are head/tail elided (5000+5000) with guidance;
- submission: the model runs `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  (alone); the episode ends when the command's output first line equals the
  marker and returncode == 0.

Deviation (documented in PLAN): closed <think>...</think> blocks are
stripped before counting action blocks. Upstream counts fences in the raw
response; a base model that drafts a fenced command inside its think block
would otherwise false-positive the exactly-1 rule (echo_xml §6/§8 lesson).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMIT_COMMAND = f"echo {SUBMIT_MARKER}"

ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
_ACTION_RE = re.compile(ACTION_REGEX, re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Verbatim agent.system_template from minisweagent/config/default.yaml.
SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

# Verbatim agent.instance_template (Linux branch; the Darwin sed note and the
# jinja platform vars are rendered statically). {task} is the only slot.
INSTANCE_TEMPLATE = """Please solve this issue: {task}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

<system_information>
Linux Ubuntu 24.04 x86_64 (docker container)
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```mswea_bash_command
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```mswea_bash_command
anything
```"""

# model.format_error_template, finish_reason=length branch (verbatim).
FORMAT_ERROR_LENGTH = (
    "Your previous response reached the output token limit (finish_reason=length) "
    "before you produced a complete action, so it was cut off. Respond more "
    "concisely and provide exactly one action in the required format. "
    "If you need to think more, do so briefly."
)

# model.format_error_template, generic branch (verbatim; {error}/{n_actions}
# are the jinja slots).
FORMAT_ERROR_TEMPLATE = """Format error:

<error>
{error}
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {n_actions} actions.
If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
Else, please format your response exactly as follows:

<response_example>
Here are some thoughts about why you want to perform the action.

```mswea_bash_command
<action>
```
</response_example>

Note: In rare cases, if you need to reference a similar format in your command, you might have
to proceed in two steps, first writing TRIPLEBACKTICKSBASH, then replacing them with ```mswea_bash_command."""

_LONG_OUTPUT_WARNING = (
    "<warning>\n"
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
    "</warning>"
)


def render_observation(
    output: str,
    returncode: int,
    exception_info: str = "",
    max_chars: int = 10000,
) -> str:
    """model.observation_template as plain Python (verbatim layout)."""
    parts: list[str] = []
    if exception_info:
        parts.append(f"<exception>{exception_info}</exception>\n")
    parts.append(f"<returncode>{returncode}</returncode>\n")
    output = output or ""
    if len(output) < max_chars:
        parts.append(f"<output>\n{output}</output>")
    else:
        half = max_chars // 2
        elided = len(output) - max_chars
        parts.append(
            _LONG_OUTPUT_WARNING
            + f"\n<output_head>\n{output[:half]}\n</output_head>\n"
            + f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
            + f"<output_tail>\n{output[-half:]}\n</output_tail>"
        )
    return "".join(parts)


def action_span(text: str) -> str:
    """Response text with think blocks removed (see module docstring)."""
    raw = text or ""
    last_open = raw.lower().rfind("<think>")
    last_close = raw.lower().rfind("</think>")
    if last_open > last_close:
        raw = raw[:last_open]
    return _THINK_BLOCK_RE.sub("", raw)


def parse_action(text: str) -> tuple[list[str], str]:
    """Return (actions, span). len(actions) != 1 is a format error upstream."""
    span = action_span(text)
    return [a.strip() for a in _ACTION_RE.findall(span)], span


def format_error_text(n_actions: int, *, truncated: bool = False) -> str:
    """The user-message text upstream sends on FormatError."""
    if truncated:
        return FORMAT_ERROR_LENGTH
    return FORMAT_ERROR_TEMPLATE.format(
        error=f"Expected exactly 1 action, found {n_actions}.",
        n_actions=n_actions,
    )


def check_submitted(output: str, returncode: int) -> tuple[bool, str]:
    """Environment._check_finished: (submitted, submission)."""
    lines = (output or "").lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == SUBMIT_MARKER and returncode == 0:
        return True, "".join(lines[1:])
    return False, ""


@dataclass
class MiniStepResult:
    observation: str
    done: bool
    reward: float
    command: str | None = None
    n_actions: int = 0
    is_format_error: bool = False
    submitted: bool = False
    max_turns_forced: bool = False
