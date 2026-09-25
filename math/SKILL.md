---
name: z3-logic-puzzles
description: >
  Use this skill whenever the user gives a logic question or logic puzzle, riddle, brain teaser, or
  "who is lying" / "who owns what" style problem (e.g. mislabeled boxes, knights and
  knaves, zebra puzzles, river crossing, seating arrangements, truth-teller puzzles).
  Instead of reasoning about the puzzle in free text, use the z3 tools in
  `math/math_plus_mcp.py` to encode the puzzle as constraints and let the solver prove
  the answer. This avoids the common mistake of a plausible-sounding but wrong answer.
version: 1.0.0
---

# z3-logic-puzzles Skill

You are likely a smaller/weaker model. Do NOT try to solve logic puzzles purely by
"thinking out loud" — it is very easy to sound confident and still be wrong. Instead,
follow this skill mechanically: turn the puzzle into constraints, hand them to the
`math_plus_mcp` MCP server (tools starting with `z3_` or `check_`), read back the
solver's verdict, and only then explain the answer in plain English using that verdict.

**Golden rule:** if the solver says `unsat` / `contradicts` / `entailed`, that is the
ground truth. Never override the solver's answer with your own intuition.

---

## Step 0 — Classify the puzzle

| Puzzle shape | Example | Tool to use |
|---|---|---|
| "Find an assignment that fits all the clues" | Zebra puzzle, seating, who-owns-what | `z3_solve_constraints` |
| "Is this single guess forced, given one piece of evidence?" | Mislabeled boxes | `z3_run_script` with `itertools.permutations` |
| "Does knowing X prove Y?" | Knights & knaves ("is the guard lying?") | `check_entailment` / `z3_prove_theorem` |
| "Are these facts even possible together?" | Sanity-check a set of clues | `check_consistency` / `z3_solve_constraints` (look for `unsat`) |
| Puzzle needs quantifiers, custom categories/sorts, or a loop over every case | Anything with "for every person..." | `z3_run_script` |

If unsure, default to `z3_run_script` — it is the most general tool (full Z3 Python API
plus `itertools`, no import/file/network access).

---

## Step 1 — Extract the objects and unknowns

Before writing any Z3 code, write out in plain text:
1. What are the **entities** (boxes, people, houses, days...)?
2. What are the **possible states/labels** each entity can take?
3. What are the **clues**, as plain statements?
4. What is the puzzle **asking** for (a full assignment? a yes/no proof? a strategy)?

Do this extraction as a short bullet list before calling any tool. Skipping this step
is the #1 cause of wrong encodings.

---

## Step 2 — Encode with the right tool

### Pattern A: Small finite-domain search (`z3_solve_constraints` or `solve_equation`)

Use string constraints with `==`, `<=`, `>=`, `<`, `>`, `+`, `-`, `*`, `/`, `And`, `Or`,
`Not`, `Implies`. Good for numeric/boolean unknowns.

```
z3_solve_constraints([
  "x + y == 10",
  "x > 3",
  "y > 2"
])
```

### Pattern B: Enumerate every permutation/assignment and check a property
(`z3_run_script` + `itertools`)

This is the right tool for puzzles like **"every label is wrong"**, "no two adjacent",
"exactly one person lies", etc. `itertools` is preloaded — no import needed. The
general recipe:

1. Build the list of ground-truth possibilities with `itertools.permutations` (or
   `itertools.product` if choices repeat).
2. Loop over each candidate, and either:
   - filter it down with a Python `if` (pure enumeration, no Z3 needed), or
   - assert Z3 constraints per-candidate if the clues involve unknowns you still need
     Z3 to pin down.
3. Assign the final model/decision to a variable named `solver` — the tool always
   reads that name.

### Pattern C: Prove an entailment (`check_entailment` / `z3_prove_theorem`)

Use when the question is "does X follow from the clues?" — i.e. is there NO
counterexample.

**Important:** `check_entailment`/`check_consistency`/`z3_solve_constraints` only
support `Real`/`Int`/`Bool` variables — there is no string/enum sort in this DSL, so
`some_var == 'oranges'` will fail or behave unpredictably. Encode a named category as a
distinct integer (or one `Bool` per category) instead:

```
check_entailment(
  premises=["label_apples == 1", "label_oranges == 0",
            "label_mixed == 0 or label_mixed == 1"],
  claim="label_mixed != 2"
)
```
(here `0 = Oranges, 1 = Apples, 2 = Mixed` is a convention you pick and use
consistently across every premise and the claim.)

For anything beyond simple arithmetic/boolean claims — genuinely categorical puzzles
with several named roles, or permutations of roles — prefer `z3_run_script` instead:
either build a real model with `solver.add(Not(claim))` (`unsat` proves the claim), or,
for a puzzle better solved by brute-force enumeration than symbolic search, follow the
worked example below.

---

## Worked example: The Mislabeled Boxes

> Three boxes, each holding either only apples, only oranges, or a mix. Each box has a
> label ("Apples", "Oranges", "Mixed") and **every label is wrong**. You may draw one
> fruit from one box (without looking inside) to relabel all three correctly. Which box
> do you draw from, and how do you deduce the rest?

**Step 1 — extract:**
- Entities: 3 boxes, currently labeled "Apples", "Oranges", "Mixed".
- True contents: some permutation of {Apples-only, Oranges-only, Mixed} across the 3
  boxes, constrained so no box's true content matches its label.
- Clue: draw one fruit from ONE box; deduce all three true contents from that single
  observation.
- Ask: which box to draw from, and why that draw is always sufficient.

**Step 2 — encode and solve with `z3_run_script`.** The core insight to verify
computationally: draw from the box **labeled "Mixed"**. Since its label is wrong, it
cannot be the mixed box — so it must be all-apples or all-oranges, and a single fruit
reveals which. Verify this is the *only* box for which one draw always resolves
everything, by brute-forcing every valid true-content permutation and checking how
many fruit-types are consistent with the observation for each candidate draw-box:

**Whatever you compute in Python, the answer only reaches you through `solver`'s model
- `z3_run_script` returns `solver.check()`'s status plus `solver.model()`'s declared
variables, and NOTHING else. A `print(...)` call inside the script is invisible to
you: it writes to the server's own console, not to the tool's response. So the last
step must ASSERT every value you want back as a real Z3 declaration, never print it.**

```python
z3_run_script([
    "import itertools",  # already preloaded, shown for clarity
    "labels = ['Apples', 'Oranges', 'Mixed']",
    "contents = ['Apples', 'Oranges', 'Mixed']",
    "valid_worlds = [perm for perm in itertools.permutations(contents)",
    "                if all(perm[i] != labels[i] for i in range(3))]",
    "",
    "# For each box we could draw from, check: does the fruit we see always",
    "# uniquely determine which permutation is the true world?",
    "def resolves(draw_index):",
    "    for fruit_seen in ('Apples', 'Oranges'):",
    "        # A single fruit drawn from a box whose true content is 'Mixed' could be",
    "        # either type, but since content is fixed, model: Apples-box only yields",
    "        # apples, Oranges-box only yields oranges, Mixed-box yields either -",
    "        # so require the observed fruit type to narrow to exactly one world.",
    "        consistent = [w for w in valid_worlds if",
    "                      (w[draw_index] == 'Apples' and fruit_seen == 'Apples') or",
    "                      (w[draw_index] == 'Oranges' and fruit_seen == 'Oranges') or",
    "                      (w[draw_index] == 'Mixed')]",
    "        if len(consistent) != 1:",
    "            return False",
    "    return True",
    "",
    "resolving_indexes = [i for i in range(3) if resolves(i)]",
    "",
    "# Report the answer through real Z3 declarations, not print(): the caller only",
    "# ever sees solver.model(), so this is the one and only way the result gets back.",
    "solver = Solver()",
    "draw_index = Int('draw_index')",
    "resolving_count = Int('resolving_count')",  # sanity check: should be exactly 1
    "valid_world_count = Int('valid_world_count')",
    "solver.add(draw_index == resolving_indexes[0])",
    "solver.add(resolving_count == len(resolving_indexes))",
    "solver.add(valid_world_count == len(valid_worlds))",
])
```

Reading the result: the returned `model` should come back as
`{"draw_index": "2", "resolving_count": "1", "valid_world_count": "2"}` (index 2 =
`labels[2]` = `"Mixed"`) — confirming there are exactly 2 valid worlds, exactly 1 box
resolves everything with a single draw, and it is the box **labeled "Mixed"**. If
`resolving_count` ever comes back other than 1, something about the encoding is wrong
- re-check it rather than trusting the `draw_index` value.

Then reason forward in plain English:

- Draw a fruit from the box **labeled "Mixed"**.
  - If you draw an **apple**, that box is actually **all-Apples** (it can't be Mixed,
    since its label is wrong). Then the box labeled "Apples" can't truly be Apples
    (wrong label) and can't be Mixed (already assigned) → it must be **Oranges**. The
    box labeled "Oranges" is therefore the true **Mixed** box (only option left, and
    consistent with its label also being wrong).
  - If you draw an **orange**, by symmetry: labeled-"Mixed" box is truly
    **all-Oranges**, labeled-"Apples" box is truly **Mixed**, labeled-"Oranges" box is
    truly **Apples**.

Always state the answer using the solver's actual output values, not memorized/assumed
values — re-run the script rather than trusting recalled numbers if the puzzle wording
changes even slightly (e.g. 4 boxes, different fruits).

---

## Other common riddle templates

**Knights and knaves** (truth-tellers always true, liars always false):
```python
z3_run_script([
    "A, B = Bools('A B')",  # A = "A is a knight" (true statement), etc.
    "solver = Solver()",
    "# A says: 'B is a liar'  -> A's statement true iff A is a knight",
    "solver.add(A == Not(B))",
    "# B says: 'A and I are both knights' -> true iff B is a knight",
    "solver.add(B == And(A, B))",
])
```
Check `sat`/`unsat` and read the model for `A`, `B`.

**Zebra-style puzzles** (N houses/entities, several attribute categories): use
`Int` variables constrained to a range (e.g. house positions 1..5) with `Distinct(...)`
for "each in its own house", combined via `z3_solve_constraints` or `z3_run_script`.

**"Is it possible at all?"** sanity checks: throw the clues at
`check_consistency` or `z3_solve_constraints`; `contradicts`/`unsat` means the puzzle's
premises (as you encoded them) conflict — re-read the clues, you likely mis-encoded one.

---

## Step 3 — Report back

1. State the tool call(s) you made and the raw verdict (`sat`/`unsat`/`entailed`/model).
2. Translate that verdict into the plain-English answer the user asked for.
3. If the solver contradicts your first intuition, trust the solver — go back and
   check your encoding for a mistake before concluding the puzzle is unsolvable.
