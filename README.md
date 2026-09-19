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

## Layout

```
data-engine/   generators, validator, and the join that reattaches each record to its menu
train/         corpus builder (build_tools_v18.py) and the SLURM jobs
eval/          probe_tools_v2.py: base vs fine-tuned on a held-out menu
demo/          MORENA Pay, the single-file app, and the Modal deployment
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
