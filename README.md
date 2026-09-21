# MORENA tools

Fine-tuning a 208M and a 503M parameter model to call real APIs from Pidgin, Yoruba, Igbo and
Hausa, and running the result on a laptop CPU.

Live demo: **https://vambo--morena-pay-pay-web.modal.run**

The base models are [MORENA](https://huggingface.co/collections/vamboai/morena), an African
language model trained from scratch. This repository is the tool-calling work on top of them: the
data generator, the corpus builder, the SLURM jobs, the eval harness and the demo.

## The problem

Measured against a menu built from the real Paystack OpenAPI spec, the released models picked the
correct tool **9% of the time**, and 0% of their calls stayed inside the arguments the schema
declares.

The cause was in the corpus, not the model. The previous builder assembled every tool-calling
example's menu from one global pool of about fifteen names scanned out of the corpus itself. The
model never had to read a schema, because the answer was always recoverable from the name. Two
further failures came from the same place: asked to change one field of a pending call it re-emitted
the whole call and corrupted an untouched amount six times out of six, and it called tools on
requests that needed none.

## The data

Generated on Cloudflare Workers AI, one unique menu per record:

| type | teaches | records |
|---|---|---|
| `toolcall_api` | read an unfamiliar schema and call it | 4,983 |
| `toolrefuse` | answer in prose, call nothing | 4,777 |
| `toolpatch` | emit only the fields that change | 4,209 |
| `tooldisambig` | ask one question, then call | 3,438 |
| `toolsession` | 4 to 7 turns, switching language mid-conversation | 3,939 |

24,000 distinct tool menus across five naming conventions (`paystack`, `verbnoun`, `camel`,
`dotted`, `terse`) with an argument-synonym table, so nothing but the format is shared between two
examples. Nigerian languages are weighted double. The sessions contain 20,826 turns, 13,060 of
which call a tool in conversational context, and 4,870 language switches.

## Three things that cost a training run each

**The decision token has to be inside the trained span.** The first build masked the loss so that
`<reserved_1>\n` (open prose) or `<reserved_1><reserved_3>` (open a call) sat in the *untrained*
prefix. The model was handed that choice on every example and got no gradient on making it.
Abstention on requests needing no tool went 12.5% to 0.0%, and 4,777 refusal records bought
nothing. Moving one token fixed it: 0% to 100%.

**It was over-asking, not over-refusing.** A later run made free-choice accuracy worse, and the
obvious reading was that refusals outweighed calls. Reading the actual completions showed every
decline was a *clarifying question*, several of them asking for a field the request already
supplied. `tooldisambig` at x3 was a quarter of the corpus teaching "when unsure, ask".

**Repetition penalty and JSON do not mix.** A `repeat_penalty` added to stop a degenerate
identifier is a penalty on braces and quotes, and it pushed the model into an invalid byte
immediately after emitting a correct patch.

## Results

### Tool calling, held-out Paystack menu

No tool name in this menu appears in the training data, and the names are not guessable:
`transferrecipient_create`, `bank_resolveAccountNumber`.

| | base | nano-tools (209M) | mini-tools (503M) |
|---|---|---|---|
| Picks the right tool | 0 to 9% | 90.9% | **100%** |
| Decides to call unprompted | 18 to 27% | 90.9% | **100%** |
| Uses only declared arguments | 0% | **100%** | **100%** |
| Edits one field, not all | 0% | **100%** | **100%** |
| Stays quiet when no tool fits | 0 to 13% | **75%** | 25% |
| Follows a mid-conversation language switch | 0% | **100%** | **100%** |
| Carries a thread across six turns | 0% | **100%** | **100%** |
| Answers in the user's language | 0 to 50% | 75% | **100%** |

### Seedance 2.5, 120 held-out briefs

African-language video briefs turned into instructions postable to Seedance unedited. Scored
structurally, never against a reference wording, because a shot has no single right answer.

| | nano-tools | mini-tools |
|---|---|---|
| Chose `seedance_generate` over four other endpoints | 100% | 100% |
| Prompt carries the camera clause | 100% | 100% |
| Camera terms Seedance understands | 100% | 100% |
| Aspect ratio matches the purpose | 99.2% | 100% |
| Duration inside 4 to 30 seconds | 100% | 100% |
| Replied in the user's language | 96.7% | 99.2% |
| **Every check passing on one brief** | **87.5%** | **90.0%** |

Neither wins outright. mini leads everywhere except abstention, where it over-calls badly enough
to answer "Good morning, how are you today?" with a balance lookup, so the demo serves nano behind
the payments page and mini behind the video page.

## Four measurement bugs, each of which nearly became a false finding

Worth reading if you are building an eval rather than a model. Every one was caught by reading the
failing rows instead of the summary line.

1. **The eval hardcoded checkpoint step numbers.** When a run finished at a different step it
   silently SKIPPED both fine-tunes and printed a clean table of base models, which is
   indistinguishable from a result. It now resolves the newest checkpoint and prints `!! MISSING`.
2. **The language detector had word lists for five languages; the corpus has eight.** Every
   correct Zulu, Shona and Swahili answer scored as a failure and read as a model defect.
3. **Among the five it did cover**, "A n se fidio" failed while "Mo n se fidio" passed, because
   the list held "mo" but not the single letter "a". Both are perfect Yoruba. The replacement
   checks orthography first: Yoruba's under-dots, Igbo's dotted vowels, Hausa's hooked letters.
4. **The language-switch suite scored the switch on a turn that warrants a tool call.** A JSON
   object has no language, so it read the English field names and reported 25% for a model that
   was behaving correctly.

Reported wrongly at first: 24% language adherence, 14% postable. Actual: 96.7% and 87.5%.

## Layout## Layout

```
data-engine/   generators, validator, and the join that reattaches each record to its menu
seedance/      camera-grammar miner: 520,378 screenplay sentences in, 34 canonical terms out
train/         corpus builder (build_tools_v18.py) and the SLURM jobs
eval/          probe_tools_v2.py: base vs fine-tuned on a held-out menu
demo/          MORENA Pay (index.html) and MORENA Studio (studio.html), plus the Modal app
docs/          the write-up
```

## Running it

```bash
python data-engine/seed_tools_v2.py 20000       # single and two-turn types
python data-engine/seed_sessions_v3.py 4000     # long, language-switching sessions
python data-engine/join_tools_v2.py --seeds <jobs> --out staged <outputs>
python data-engine/validate_tools_v2.py staged
python train/build_tools_v18.py --raw <staged> --out-dir <shards>

sbatch train/slurm/tools_sft_build.sbatch       # tokenise, CPU
sbatch train/slurm/tools_sft_nano.sbatch        # 1x A100, ~25 min
sbatch train/slurm/tools_sft_mini.sbatch        # 1x A100, ~33 min
sbatch train/slurm/tools_eval.sbatch
```

The demo runs against any llama.cpp server:

```bash
llama-server -m <model>.gguf --path demo --no-jinja -c 4096
```

`--no-jinja` matters: the app drives the raw completion endpoint, and the chat-template parser
rejects an otherwise valid response when the model trails a stray byte.

## Licence

Apache 2.0, same as the base models.
