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
# mini, not nano. On the held-out Paystack menu the two tie on picking the right tool (90.9%) and
# on argument hygiene (95.5%), but mini emits a minimal patch 100% of the time against nano's
# 66.7%, and asks before guessing 100% against 83.3%. Editing a pending transfer is the part of
# this demo people actually try to break, so the better one goes in front of them.
GGUF = os.environ.get("MORENA_PAY_GGUF", "mini-tools-Q4_K_M.gguf")
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
        try:
            vol.reload()
            path = os.path.join(MODEL_DIR, GGUF)
            if not (os.path.exists(path) and os.path.getsize(path) > 10_000_000):
                raise FileNotFoundError(
                    f"{path} missing. Push it with: modal volume put morena-pay-models {GGUF}")
            # n_threads matches the 4 cores requested; llama.cpp otherwise guesses from the host,
            # which on a shared machine means oversubscribing and getting slower.
            self.llm = Llama(model_path=path, n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
        except Exception as e:
            self.llm = None
            self.error = f"{type(e).__name__}: {e}"

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

        @api.get("/props")
        def props():
            # The page reads model_path to show which model it is talking to, and treats a
            # non-200 as "model offline".
            if self.llm is None:
                return JSONResponse({"error": self.error}, status_code=503)
            return {"model_path": GGUF, "n_ctx": 4096}

        @api.post("/completion")
        async def completion(req: Request):
            if self.llm is None:
                return JSONResponse({"error": self.error}, status_code=503)
            b = await req.json()
            prompt = b.get("prompt") or ""
            if not prompt:
                return JSONResponse({"error": "no prompt"}, status_code=400)
            # A public URL is an open text box. The page never needs more than a short call plus a
            # sentence, so the ceilings are set where the demo lives rather than where the model
            # could go.
            n = min(int(b.get("n_predict") or 200), 320)
            prompt = prompt[-14000:]
            out = self.llm(
                prompt,
                max_tokens=n,
                temperature=float(b.get("temperature") or 0.0),
                top_p=float(b.get("top_p") or 1.0),
                stop=b.get("stop") or ["<reserved_0>", "<reserved_5>", "<|tool_result|>"],
                echo=False,
            )
            return {"content": out["choices"][0]["text"]}

        return api
