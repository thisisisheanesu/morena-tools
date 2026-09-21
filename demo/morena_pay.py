"""MORENA Pay: the Nigerian fintech tool-calling demo, as a public web app.

    modal deploy pay/morena_pay.py

One container serves both the page and the model, so the browser talks to a single origin and
there is no CORS to arrange and no mixed-content rule to fight. The endpoints are the two that
llama.cpp's server exposes and the page already speaks, so the same index.html runs unchanged
against a local `./morena --ui` or against this.

The weights sit on a Modal volume, pushed straight from the laptop with `modal volume put`. The
hub is not involved: the token on Leonardo is read-only so the cluster cannot publish, and the ask
was a shareable demo rather than a weights release, so nothing here is public except the page.

CPU, not GPU, and that is the argument rather than a saving. A 503M model answering in about a
second on two cores is the whole point: this is a thing a Nigerian fintech could run on a cheap
box, not a thing that needs an A100.
"""
import json
import os

import modal

APP = "morena-pay"
# BOTH models, because neither wins outright and the two demos want opposite things.
#
# mini is better at nearly everything: right tool 100% against 90.9%, free-choice calling 100%
# against 90.9%, stays in the user's language 100% against 75%, and 90.0% POSTABLE on held-out
# Seedance briefs against 87.5%. But it OVER-CALLS: asked "Good morning, how are you today?" it
# calls balance_fetch, and asked who won the match it invents a tool that is not in the menu.
# Abstention is 25% against nano's 75%.
#
# Pay is the demo people prod with small talk, so nano sits behind it. Studio is a one-shot brief
# where over-calling cannot happen (there is only one sensible tool) and language fidelity is the
# whole point, so mini sits behind that. Two 4-bit files are 470MB together and load side by side.
MODELS = {"nano": "nano-tools-Q4_K_M.gguf", "mini": "mini-tools-Q4_K_M.gguf"}
DEFAULT_MODEL = os.environ.get("MORENA_PAY_GGUF", "nano")
MODEL_DIR = "/models"
HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake")
    .pip_install("llama-cpp-python==0.3.16", "fastapi[standard]==0.115.6")
    # Resolved absolute paths, and the files live beside this module rather than up-and-across.
    # With "pay/../local/ui/index.html" Modal kept serving a cached layer: three deploys in a row
    # reported success while the URL still returned the previous build, and the response carried
    # last-modified 1970, which is a baked layer rather than a fresh mount. pay/ui is a copy of
    # local/ui, synced by deploy.sh, so the same page is served locally and here.
    .add_local_dir(os.path.abspath(os.path.join(HERE, "ui")), "/ui")
)
vol = modal.Volume.from_name("morena-pay-models", create_if_missing=True)
app = modal.App(APP)


@app.cls(image=image, volumes={MODEL_DIR: vol},
         cpu=4, memory=4096, scaledown_window=300, timeout=600, max_containers=4)
@modal.concurrent(max_inputs=4)
class Pay:
    @modal.enter()
    def load(self):
        from llama_cpp import Llama
        self.error = None
        self.llms = {}
        try:
            vol.reload()
            for key, fn in MODELS.items():
                path = os.path.join(MODEL_DIR, fn)
                if not (os.path.exists(path) and os.path.getsize(path) > 10_000_000):
                    raise FileNotFoundError(
                        f"{path} missing. Push it with: modal volume put morena-pay-models {fn}")
                # n_threads matches the 4 cores requested; llama.cpp otherwise guesses from the
                # host, which on a shared machine means oversubscribing and getting slower.
                self.llms[key] = Llama(model_path=path, n_ctx=4096, n_threads=4,
                                       n_batch=512, verbose=False)
        except Exception as e:
            self.llms = {}
            self.error = f"{type(e).__name__}: {e}"

    def pick(self, name):
        """Resolve a requested model name, falling back to the default rather than erroring: a
        demo page asking for a model that is not loaded should still answer."""
        return self.llms.get(name) or self.llms.get(DEFAULT_MODEL)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, JSONResponse

        api = FastAPI(title="MORENA Pay")

        @api.get("/")
        def index():
            return FileResponse("/ui/index.html")

        @api.get("/console.html")
        def console():
            return FileResponse("/ui/console.html")

        @api.get("/studio")
        @api.get("/studio.html")
        def studio():
            # The Seedance demo. Same model, same endpoint, a different menu: the point of a
            # schema-reading model is that the second use case costs a page, not a fine-tune.
            return FileResponse("/ui/studio.html")

        @api.get("/props")
        def props(model: str = ""):
            # The page reads model_path to show which model it is talking to, and treats a
            # non-200 as "model offline".
            if not self.llms:
                return JSONResponse({"error": self.error}, status_code=503)
            key = model if model in self.llms else DEFAULT_MODEL
            return {"model_path": MODELS[key], "model": key, "n_ctx": 4096,
                    "available": sorted(self.llms)}

        @api.post("/completion")
        async def completion(req: Request):
            if not self.llms:
                return JSONResponse({"error": self.error}, status_code=503)
            b = await req.json()
            llm = self.pick(b.get("model") or DEFAULT_MODEL)
            prompt = b.get("prompt") or ""
            if not prompt:
                return JSONResponse({"error": "no prompt"}, status_code=400)
            # A public URL is an open text box. The page never needs more than a short call plus a
            # sentence, so the ceilings are set where the demo lives rather than where the model
            # could go.
            n = min(int(b.get("n_predict") or 200), 320)
            prompt = prompt[-14000:]
            out = llm(
                prompt,
                max_tokens=n,
                temperature=float(b.get("temperature") or 0.0),
                top_p=float(b.get("top_p") or 1.0),
                stop=b.get("stop") or ["<reserved_0>", "<reserved_5>", "<|tool_result|>"],
                echo=False,
            )
            return {"content": out["choices"][0]["text"]}

        return api
