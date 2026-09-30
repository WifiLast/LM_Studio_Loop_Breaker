§18 Why spaCy-only extraction fails

spaCy delivers syntax, not propositions. Dependency parsing yields one tree per sentence. Rule-based subject–verb–object extraction on that tree breaks at eight structural points:

Multiple statements per sentence. Coordination, relative clauses and appositions pack several facts into one sentence, but only one triple comes out.
Missing context. Pronouns, bridging references ("the voltage" without naming the device) and subjects inherited from list headings leave the subject empty or wrong.
Meaning outside the verb. Nominalizations, light verbs ("perform a restart"), compounds ("supply voltage") and possessives carry the predicate somewhere other than the main verb.
Logical operators outside the triple. Negation scope, modality, conditionals, quantifiers and comparisons are not captured by a plain triple at all.
Quantities. Ranges, tolerances, the German decimal comma, and compound units are mis-tokenized or lost.
LLM output format. Markdown lists, tables, code blocks and inline identifiers (GVL.xStart, 192.168.1.10, 3.5.19.0) break the parser.
Domain vocabulary. The pretrained NER knows no CODESYS/WAGO entities.
No validation. Every extracted triple is accepted, including wrong ones.

Consequence: spaCy remains the tokenization, sentence and parse substrate. Extraction itself becomes a layered pipeline with several independent extractors and validation at the end.

§19 Target pipeline for extraction
Layer	Task	Main component
L0	Structure normalization of LLM output	Markdown parser, own rules
L1	Coreference resolution	fastcoref / constrained LLM rewrite
L2	Decomposition into atomic propositions	rule-based clause splitter + propositionizer
L3	Entities and identifiers	GLiNER + regex/EntityRuler
L4	Quantities, units, intervals	own quantity grammar + pint
L5	Relation extraction as an ensemble	rules + schema-guided zero-shot RE + constrained LLM
L6	Logical operators	negation, modality, condition, quantifier, comparison, time
L7	Canonicalization	entity linking, predicate mapping
L8	Validation	type check against ontology + NLI round trip

L0–L4 each produce annotations on the original text. L5 generates candidates. L6–L8 enrich and filter. The output is the IR from [§4], extended in §29.

§20 L0: Structure normalization
Parse the Markdown into an AST with markdown-it-py, so each block type can be handled on its own.
Code blocks are excluded from NLP extraction.
They are marked CODE.
Optionally, a language-specific parser (ST/IEC 61131-3) handles them later.
Inline code (`GVL.xStart`) is replaced by a placeholder token ⟨ID_07⟩.
The original string goes into a mapping table.
This prevents the tokenizer from splitting on . or _.
Tables are converted directly into facts: the first column is the entity, the header row gives the predicates, and each cell is a value. Tables are the most precise fact source in LLM output and bypass NLP entirely.
Lists inherit their context:
The line introducing the list ("The PFC200 offers:") supplies subject and predicate.
Each list item becomes a complete sentence ("The PFC200 offers 2 Ethernet ports.").
Nested lists pass the context down recursively.
Headings set a document context entity, used as a fallback subject for bridging in §21.
Parentheses ((default: 500 ms), (e.g. Modbus)) are split into separate propositions that point back to their parent sentence.
Character offsets into the original text are kept through every transformation. This is a prerequisite for span replacement in [§7].
§21 L1: Coreference and bridging

The goal is that every proposition has an explicit subject, without a single pronoun left over.

Two options for the coreference engine:

A) fastcoref (FCoref / LingMessCoref) for English.
For German: coreferee, but check compatibility with the current spaCy version first.
Deterministic, fast, and no hallucination risk.
B) Constrained LLM rewrite. Instruction: "Replace pronouns with their referents, change nothing else." Then a token diff against the original checks that only pronoun positions changed; otherwise the output is discarded.
Handles both languages and bridging.
Needs the diff guard because the model can rewrite text.

Bridging ("The module has 8 channels. The voltage is 24 V."), handled by an ontology-guided rule:

The attribute voltage has a domain in the ontology (for example Device and Module).
Assign the attribute to the most recently mentioned entity whose class matches that domain.
Otherwise, assign it to the document context entity from §20.
Confidence is lower than for explicit references.

References: https://github.com/shon-otmazgin/fastcoref · https://github.com/richardpaulhudson/coreferee

§22 L2: Decomposition into atomic propositions

The goal is one statement per unit. This has the biggest single effect on extraction quality.

Rule-based splitter on the dependency parse (high precision)

Construction	Rule
Coordinated objects/subjects (conj)	Duplicate the proposition once per conjunct
Relative clause (relcl)	Separate proposition; the antecedent becomes the subject
Apposition (appos)	Separate is_a / alias proposition
Adverbial clause (advcl with mark)	Separate proposition linked by a condition or cause edge (§26)
Gapping ("Input 1 uses 24 V, input 2 12 V")	Copy the verb from the first conjunct into the elided one

Collective vs. distributive readings

Markers such as "together", "combined", "in total", "jointly" block duplication.
"A and B together draw 5 A" becomes one proposition about the group, not two.

Complement for sentences the rules do not split (two options)

A) Propositionizer model, a seq2seq model trained on proposition decomposition (Dense X Retrieval). It is dedicated and cheap, but trained on English Wikipedia style.
B) Small instruct model with grammar-constrained decoding that outputs a JSON array of atomic sentences.
Guard: every content word of every proposition must appear in the source sentence (lemma match). This blocks invented content.

References: https://arxiv.org/abs/2312.06648 · https://github.com/dottxt-ai/outlines

§23 L3: Entities and identifiers
Regex/EntityRuler layer (deterministic, highest priority):
IPv4/IPv6 addresses, ports, MAC addresses
version numbers
IEC data types (BOOL, INT, REAL, TIME)
variable paths (GVL.x*, Application.PLC_PRG.*)
article numbers (750-8212)
protocols (PROFINET, EtherCAT, OPC UA, Modbus TCP/RTU)
units
GLiNER for zero-shot NER with your own labels: device, module, software, protocol, signal, parameter, firmware, component, location.
gliner_multi covers German and English.
Labels come directly from the ontology classes [§5], so NER and ontology share one vocabulary.
Conflict rule: the regex span wins over the GLiNER span, and the longest span wins among equal sources.
Compound resolution ("PFC200 supply voltage", German "Versorgungsspannung"): check a compound head against the attribute lexicon of the ontology. If it matches, turn it into attribute(entity, predicate), e.g. supply_voltage(PFC200).

Reference: https://github.com/urchade/GLiNER

§24 L4: Quantities, units, intervals

Use a separate quantity grammar (lark or a regex cascade) instead of relying on the NER. Every quantity becomes an interval [lo, hi] with a unit, which maps directly to Z3 lo <= x, x <= hi.

Form	Example	Normalization
Point value	24 V	[24, 24] V
Range	10–20 °C, 10 to 20 °C, between 10 and 20	[10, 20]
Tolerance	24 V ±10 %	[21.6, 26.4]
Bound	max. 1 A, up to 1 A, ≤ 1 A	[−∞, 1]
Bound	at least 4, min. 4, ≥ 4	[4, ∞]
Approximation	about 5 ms, ~5 ms, approx. 5 ms	[5·(1−t), 5·(1+t)], with t as a valve
Relative	20 % higher than B	x = 1.2·B (becomes a comparison, §26)
Percentage points	3 percentage points more	additive, not multiplicative
Number formats	3,5 bar (DE), 1.000 vs 1,000, 1e-3, "two", ½	locale detection per answer language
Compound units	l/min, kWh, mA, µs	pint, converted to SI

Ambiguity rule: 1.000 in a German answer is 1000, and in an English answer 1.0. When the context is unclear, keep both readings, set confidence to low, and send the value to NLI validation (§28).

Reference: https://pint.readthedocs.io

§25 L5: Relation extraction as an ensemble

There are three independent sources. No single source is reliable enough on its own.

Rule extractor on the dependency parse.
Handles passive normalization (nsubjpass/agent), copula with prepositions ("is located in" becomes located_in), existentials ("there are 4 inputs" becomes has_count), possessives, light verbs via a lexicon ("perform a restart" becomes restart), and nominalizations via a lexicon ("activation of X" becomes activate(X)).
High precision, limited coverage.
Model extractor, two options:
A) GLiREL, schema-guided zero-shot RE. Relation labels come from the ontology predicate list. It is local, fast, and cannot hallucinate text because it only classifies pairs of entities already found by L3.
B) Constrained LLM extraction. The JSON schema contains enums of the ontology predicates and entity IDs from L3, so the model can only choose from what exists. It has the highest coverage for complex sentences, and requires the L8 guard.
Table/list extractor from §20, which is deterministic.

Merging candidates

Build a key (subject_id, predicate, object_id|value, polarity) for each candidate.
Confidence is the weighted sum of the agreeing sources:
table: 1.0
rule: 0.8
GLiREL: 0.6
LLM: 0.5
Candidates from exactly one model-based source must pass L8 before they count.
Disagreements between sources go to the annotation queue (§32).

References: https://github.com/jackboyla/GLiREL · https://github.com/SapienzaNLP/relik

§26 L6: Logical operators

Each proposition gets operator fields that map directly onto the Z3 encoding [§6].

Negation

Cue lexicon (EN/DE): not, no, never, without, neither…nor, kein, nicht, nie, ohne, weder…noch.
Scope comes from the dependency subtree of the cue.
Special cases:
"not only…but also" is not a negation.
"no longer" is a negation plus a temporal marker.
Double negation ("not impossible") becomes positive plus a hedge.
"does not support A and B" is ambiguous between ¬A∧¬B and ¬(A∧B). Keep both readings, mark it ambiguous, and do not assert it as a hard fact.

Modality, stored in the field modality:

Class	Cues	Z3 treatment
Assertion	indicative, no modal verb	hard claim
Possibility	can, may, possible, kann	not asserted; only checked for compatibility
Obligation	must, required, shall, muss	requirement constraint
Recommendation	should, recommended, soll	no constraint
Hedge	usually, typically, generally, meist	soft, low weight

Conditionals

Form	Encoding
if / when / wenn / falls	Implies(C, P)
unless / außer wenn	Implies(Not(C), P)
only if / nur wenn	Implies(P, C), reversed direction
if and only if / genau dann wenn	C == P
Counterfactual (would have / hätte)	not a fact claim, discarded

Causality: because, since, therefore, due to, deshalb, weil.

Both sides are asserted, plus Implies(cause, effect).
"since" / "da" can be temporal. If the clause contains a time expression, treat it as temporal.

Quantifiers

Form	Encoding
all / every / each / jede	ForAll over the class
none / kein	ForAll negated
some / einige	Exists
at least n / exactly n	cardinality via a sum over the Bool indicators
most / meist	not decidable, marked as hedge

Genericity

"PLCs use cyclic tasks" is a class statement, not an instance statement.
Signal: bare plural subject without a determiner, in the present tense.
Encoding: ForAll over instances of the class, with lower confidence.

Comparison and superlatives

Comparative: "faster than" becomes attr(A) > attr(B).
Ratio: "twice as" becomes attr(A) == 2*attr(B).
Superlative: "the fastest" becomes ForAll x≠A: attr(x) < attr(A), restricted to the entities in the context.

Time and version scope

Phrases such as "since firmware 04", "from CODESYS 3.5 SP19", "no longer" go into the field valid_scope.
Z3 assertions are then conditional on the version: Implies(fw >= 4, P).

Attribution and examples

"according to the manual" / "laut WAGO" is stored in the field attribution. The claim remains a claim.
"e.g." / "z.B." makes the list non-exhaustive, so it becomes Exists, never a complete enumeration.

Non-claims are filtered out: questions, imperatives and procedural instructions ("Open the device tree…") get the type PROCEDURE and are not verified.

Reference: https://github.com/jenojp/negspacy

§27 L7: Canonicalization

Entity linking

Exact alias match against the KB.
rapidfuzz match above a threshold.
Embedding similarity (multilingual sentence transformer), always filtered by the class from L3.
If nothing matches, the entity becomes NEW. It stays unverifiable, but remains available for internal consistency checks.

Predicate mapping, two options:

A) Synonym lexicon plus embedding similarity of the predicate phrase against the ontology predicate descriptions. Accept if above a threshold; otherwise UNMAPPED.
B) Zero-shot NLI classification. The hypothesis is "This sentence states the {predicate} of {subject}." Candidates are only the predicates whose domain and range fit the entity types.
More precise for rare phrasings.
Slower, since there is one NLI call per candidate.

Inverse relations and normalization

part_of(Y, X) and contains(X, Y) are mapped to one canonical direction, defined in the ontology.
Units are converted to SI, values stored as intervals (§24).
§28 L8: Validation
Type check.
Subject class must be in the predicate's domain; object class or value type in its range.
Values must be within physical bounds.
Violation means the candidate is discarded, not corrected.
NLI round-trip verification.
A template per predicate turns the triple back into a sentence ("{s} has a supply voltage of {v} {u}.").
An NLI cross-encoder then checks: premise is the source proposition from L2, hypothesis is the generated sentence.
entailment ≥ threshold means accept. contradiction means an extraction error (for example, polarity lost), and the triple is discarded.
This filters out hallucinated LLM triples and lost negations.
Span check. Every value and entity of the triple must lie within the character span of the source proposition.
Abstention. Below the overall confidence threshold the claim becomes UNEXTRACTABLE. The sentence stays unchanged and is marked unverified [§7].

References: https://huggingface.co/cross-encoder/nli-deberta-v3-base · https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

§29 Extended IR

These fields are added to [§4]:

Field	Content
proposition_id	the atomic unit from L2
parent_sentence_id	the source sentence
evidence_span	offsets into the original text
extractors[]	sources with individual confidence
confidence	aggregated value
value_interval	lo, hi, SI unit
polarity	pos / neg / ambiguous
modality	class from §26
hedge	bool
quantifier	none / all / exists / card(n, op)
generic	bool
condition_refs[]	proposition IDs plus condition type: if / unless / only_if / iff / cause
valid_scope	version or time scope
attribution	source named in the text
nli_score	round-trip result
status	OK / AMBIGUOUS / UNMAPPED / UNEXTRACTABLE / PROCEDURE
§30 Edge-case catalog (test categories)
#	Category	Example	Layer
1	Coordinated objects	"supports Modbus TCP and OPC UA"	L2
2	Coordinated subjects	"PFC100 and PFC200 run Linux"	L2
3	Collective reading	"A and B together draw 5 A"	L2
4	Gapping	"Input 1 uses 24 V, input 2 12 V"	L2
5	Relative clause	"The controller, which runs CODESYS 3.5, …"	L2
6	Apposition	"PFC200, a Linux-based PLC, …"	L2
7	Passive	"The value is written by task 2"	L5
8	Copula with preposition	"The fuse is located in cabinet 3"	L5
9	Existential	"There are 8 inputs on module X"	L5
10	Possessive/compound	"the PFC200's supply voltage", "Versorgungsspannung"	L3, L5
11	Nominalization	"Activation of the pump occurs after …"	L5
12	Light verb	"perform a restart"	L5
13	Pronoun	"It supports EtherCAT"	L1
14	Bridging	"The module … The voltage is 24 V"	L1
15	List inheritance	"Features:\n- 2 Ethernet ports"	L0
16	Table	parameter table	L0
17	Negation scope	"does not support A and B"	L6
18	Double negation	"not impossible"	L6
19	Pseudo-negation	"not only … but also"	L6
20	Modality	"can", "must", "should"	L6
21	Hedge	"usually", "typically"	L6
22	Conditional	"if / unless / only if / iff"	L6
23	Counterfactual	"would have failed"	L6
24	Causal / temporal "since"	"since the fuse tripped"	L6
25	Quantifier	"all modules", "at least 2 ports"	L6
26	Generic	"PLCs use cyclic tasks"	L6
27	Comparative / ratio	"twice as fast as B"	L6
28	Superlative	"the fastest module"	L6
29	Range / tolerance / bound	"24 V ±10 %", "max. 1 A"	L4
30	Number format	"3,5 bar", "1.000"	L4
31	Relative percentage	"20 % higher", "3 percentage points"	L4, L6
32	Version scope	"since firmware 04"	L6
33	Attribution	"according to the manual"	L6
34	Example list	"e.g. Modbus"	L6
35	Parenthesis / default	"(default: 500 ms)"	L0
36	Definition	"X means Y", "X, i.e. Y"	L5
37	Identifiers	GVL.xStart, 192.168.1.10, 3.5.19.0	L0, L3
38	Code block	ST code in the answer	L0
39	Instruction	"Open the device tree"	L6 → PROCEDURE
40	Language mix	a German sentence containing English terms	L3, L4
§31 German-specific handling
Parser: de_dep_news_trf for the parse. The German label set uses TIGER-based dependency labels (sb, oa, ng, cj, rc), so every rule from §22 and §25 needs its own German mapping table.
Compound splitting (Versorgungsspannung → Versorgung + Spannung): compound-split or CharSplit, then an attribute lexicon lookup on the head.
Separable verbs ("schaltet … ab" becomes abschalten): lemma reconstruction from the svp label.
V2 word order and verb-final subordinate clauses: clause boundaries come from cp/rc, not from word order.
Language detection per answer (lingua) controls the spaCy model, the number format (§24) and the cue lexicons (§26).
§32 Evaluation and improvement loop
Gold corpus. 400–600 propositions taken from real answers of the weak model in the CODESYS/WAGO domain.
At least 10 examples per category from §30.
Annotation in Argilla or Label Studio, directly in the IR format (§29).
Metrics per layer and per category:
triple precision/recall/F1, both exact and relaxed (value within the interval)
polarity accuracy
modality accuracy
condition structure accuracy
UNEXTRACTABLE rate
Regression suite: pytest, parametrized over the §30 categories. A drop in a category's F1 fails CI.
Active learning: extractor disagreements (§25) and NLI contradictions (§28) automatically land in the annotation queue.
Domain adaptation once 1,500+ annotations exist, two options:
A) Fine-tune GLiNER and GLiREL on the domain data. The model stays small and local, and gains the most on domain vocabulary.
B) Distillation. Label a large amount of unannotated text with constrained LLM extraction plus L8 validation, then train a spaCy rel component or GLiREL on the result. More data, but with a risk of propagating the LLM's errors.
§33 Implementation phases
Phase	Content	Exit criterion
P1	Ontology v1 (classes, predicates, types, bounds) + gold corpus v1	≥ 10 examples per §30 category
P2	L0 structure normalization + L3 regex layer + L4 quantity grammar	categories 15, 16, 29–31, 35, 37, 38 at F1 ≥ 0.9
P3	L2 rule-based splitter (EN + DE)	categories 1–6 at F1 ≥ 0.85
P4	L6 operators (negation, modality, conditions, quantifiers)	polarity accuracy ≥ 0.95
P5	L3 GLiNER + L5 rule extractor + L7 canonicalization	overall triple precision ≥ 0.85
P6	L8 type check + NLI round trip	false triples reduced ≥ 50 % against P5
P7	L5 model extractor (GLiREL or constrained LLM) + ensemble	recall increase without precision loss
P8	L1 coreference + bridging	categories 13–14 at F1 ≥ 0.8
P9	Active learning loop + domain adaptation (§32)	continuous

Reasoning for the order:

Deterministic layers come first (P2–P4). They are measurable, cause no regressions, and fix the most frequent errors in LLM output: lists, units, negation.
Validation (P6) comes before the model-based extractors (P7), so their hallucinations are filtered from the start.
Coreference comes late (P8). Its gain depends on the answer style, and inlet instructions for explicit subjects [§8] already reduce pronouns at the source.
§34 Remaining limits
Ambiguous negation scope and collective vs. distributive readings remain undecidable in some cases. They correctly end as AMBIGUOUS, not as a false fact.
Bridging without an ontology domain match stays unreliable.
A closed predicate vocabulary means everything outside the ontology stays UNMAPPED. Coverage grows only as the ontology grows.
Latency scales with the number of NLI and LLM calls. Budget control: run NLI only for claims from a single model-based source (§25), and run the constrained LLM only on propositions left over after the rule and GLiREL passes.