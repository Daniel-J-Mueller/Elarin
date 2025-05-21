# Terminal GUI Plan

This document outlines how to evolve the simple STDOUT logging into a usable terminal interface for monitoring and rating Elarin's motor cortex output.  The overall goal is to keep the interface text‑only so it can run in any terminal while still supporting interaction.

## Goals

1. Show motor cortex messages in a scrollable box on the left.
2. Include timestamps with date separators when the day changes.
3. Display the header `motor_cortex INFO` above the output list.
4. Provide colour coded rating buttons (-5 … 5) next to each message.
5. Allow selection of a message and submission of an alternate response for training.
6. Apply positive or negative reinforcement when a rating is chosen.

## Design

The interface will be implemented with Python's `curses` module so no extra dependencies are required.  It divides the terminal into two panes:

```
┌──────────────────────────────────────┐┌─────────────────────┐
│ motor_cortex INFO                    ││ Selected Output     │
│ [scrollable log with timestamps]     ││ [rating buttons]    │
│                                      ││                     │
│                                      ││ Text input for      │
│                                      ││ corrective prompt   │
└──────────────────────────────────────┘└─────────────────────┘
```

### Logging Handler

A custom `logging.Handler` will capture messages from the `motor_cortex` logger.  Each record is stored with its timestamp.  Dates are compared against the previous entry and a coloured divider is inserted when the day changes.

### Scrollable Output

The left pane uses a `curses` pad so older messages can be scrolled with the arrow keys or PgUp/PgDn.  A status bar at the bottom shows the current position.  New entries automatically scroll into view unless the user has scrolled back.

### Rating Bar

When a log entry is highlighted, a vertical bar of buttons from -5 to 5 appears.  Colours range from purple (negative) through blue to light cyan (positive).  Pressing a number triggers `MotorCortex.reinforce_output(rating, token_id)` which will be added as a new method.  Positive values strengthen the pathways that produced the token while negative values weaken them.

### Correction Input

The right pane contains a small text box.  Entering text here and pressing Enter sends the string to a new `teach(text, context)` helper which encodes it via Wernicke's area and trains the motor cortex adapters using the current context embedding.  This effectively tells Elarin what it should have said.

### Persistence

Ratings and corrections are appended to `persistent/cli_feedback.log` so future sessions can replay them.  This keeps the manual guidance history intact.

## Integration

* `brain.py` gains an optional `--tui` flag.  When enabled it starts the `TerminalGUI` instead of printing raw log lines.  The rest of the brain loop remains unchanged.
* `utils/logger.py` exposes `install_handler(handler)` so the GUI can hook into existing loggers without rewriting them.
* A new module `terminal_gui.py` implements the curses interface and the reinforcement helpers.
* The run script `run_brain.sh` will accept `--tui` and forward it to `brain.py`.

## Future Improvements

* Replace the basic curses interface with the richer `textual` library once extra dependencies are permitted.
* Add search/filter controls for long sessions.
* Visualise hormone levels and novelty metrics in additional panes.