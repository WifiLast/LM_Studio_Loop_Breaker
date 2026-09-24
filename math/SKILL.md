---
name: z3-logic-puzzles
description: >
  Use this skill whenever the user gives a logic puzzle, riddle, brain teaser, or
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

```
check_entailment(
  premises=["label_apples_box == 'oranges'",
            "label_oranges_box == 'apples'",
            "label_mixed_box == 'apples' or label_mixed_box == 'oranges'"],
  claim="..."
)
```
In practice, for anything beyond simple arithmetic/boolean claims (custom categories,
permutations of roles), prefer `z3_run_script` — build the model, add
`solver.add(Not(claim))`, and check: `unsat` means the claim is proved.

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

```python
z3_run_script([
    "import itertools",  # already preloaded, shown for clarity
    "labels = ['Apples', 'Oranges', 'Mixed']",
    "contents = ['Apples', 'Oranges', 'Mixed']",
    "valid_worlds = []",
    "for perm in itertools.permutations(contents):",
    "    # perm[i] = true content of the box labeled labels[i]",
    "    if all(perm[i] != labels[i] for i in range(3)):",
    "        valid_worlds.append(perm)",
    "",
    "# For each box we could draw from, check: does the fruit we see always",
    "# uniquely determine which permutation is the true world?",
    "results = {}",
    "for draw_index in range(3):",
    "    ok = True",
    "    for fruit_seen in ['Apples', 'Oranges']:",
    "        matches = [w for w in valid_worlds if",
    "                   (fruit_seen == 'Apples' and w[draw_index] in ('Apples', 'Mixed') and w[draw_index] != 'Oranges') or",
    "                   (fruit_seen == 'Oranges' and w[draw_index] in ('Oranges', 'Mixed') and w[draw_index] != 'Apples')]",
    "        # A single fruit drawn from a box whose true content is 'Mixed' could be",
    "        # either type, but since content is fixed, model: Apples-box only yields",
    "        # apples, Oranges-box only yields oranges, Mixed-box yields either -",
    "        # so require the observed fruit type to narrow to exactly one world.",
    "        consistent = [w for w in valid_worlds if",
    "                      (w[draw_index] == 'Apples' and fruit_seen == 'Apples') or",
    "                      (w[draw_index] == 'Oranges' and fruit_seen == 'Oranges') or",
    "                      (w[draw_index] == 'Mixed')]",
    "        if len(consistent) != 1:",
    "            ok = False",
    "    results[labels[draw_index]] = ok",
    "",
    "solver = Solver()",
    "best = [k for k, v in results.items() if v]",
    "solver.add(Bool('placeholder') == Bool('placeholder'))",  # keeps a trivial sat model
    "print('valid_worlds:', valid_worlds)",
    "print('box_that_always_resolves:', best)",
])
```

Reading the result: `best` should come back as `['Mixed']` — confirming you must draw
from the box labeled **"Mixed"**. Then reason forward in plain English using the
`valid_worlds` printed:

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
