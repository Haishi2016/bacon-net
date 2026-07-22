"""
Human-defined concept set and per-digit BACON logic for MNIST.

Everything here is meant to be edited by a human.  ``CONCEPTS`` is the list of
interpretable stroke/shape concepts the CNN must learn to detect (without any
concept-level supervision), and ``DIGIT_RULES`` is one human-authored BACON
formula per digit that combines those concepts with AND / OR / NOT.

The formulas use a tiny DSL understood by ``bacon_logic.parse_formula``:
    - concept names (must appear in CONCEPTS)
    - AND, OR, NOT  (case-insensitive)
    - parentheses for grouping

You can pass a different concept set / rule set from the command line via a
JSON file (see train.py --rules), but this is the default.
"""

# Interpretable stroke / shape concepts for handwritten digits.
CONCEPTS = [
    "loop_upper",        # closed loop in the upper half (0, 8, 9)
    "loop_lower",        # closed loop in the lower half (0, 6, 8)
    "vertical_line",     # a tall vertical stroke (1, 4, 7, 9)
    "horizontal_top",    # a bar across the top (5, 7)
    "horizontal_middle", # a bar / junction across the middle (3, 4, 8)
    "horizontal_bottom", # a bar across the bottom (2)
    "left_curve",        # a curve opening toward the left (2, 3, 5, 6)
    "right_curve",       # a curve opening toward the right (c-like) (reserved)
    "diagonal",          # a slanted stroke (7, 2)
]

# One human-defined BACON tree per digit.  Uses conjunction (AND),
# disjunction (OR) and negation (NOT, e.g. "not a loop").
DIGIT_RULES = {
    0: "loop_upper AND loop_lower AND NOT horizontal_middle AND NOT vertical_line",
    1: "vertical_line AND NOT loop_upper AND NOT loop_lower AND NOT horizontal_top",
    2: "left_curve AND horizontal_bottom AND diagonal AND NOT loop_lower",
    3: "left_curve AND horizontal_middle AND NOT loop_lower AND NOT vertical_line",
    4: "vertical_line AND horizontal_middle AND NOT loop_upper AND NOT loop_lower",
    5: "horizontal_top AND left_curve AND loop_lower AND NOT loop_upper",
    6: "loop_lower AND left_curve AND NOT loop_upper AND NOT horizontal_top",
    7: "horizontal_top AND (diagonal OR vertical_line) AND NOT loop_upper AND NOT loop_lower",
    8: "loop_upper AND loop_lower AND horizontal_middle",
    9: "loop_upper AND vertical_line AND NOT loop_lower",
}

# Graded-logic andness for the fixed trees.
AND_ANDNESS = 1.0   # 1.0 -> strong conjunction
OR_ANDNESS = 0.0    # 0.0 -> strong disjunction
