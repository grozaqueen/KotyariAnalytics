from fastapi import FastAPI
from pydantic import BaseModel
from service.context_builder import build_context_for_query
from service.prompt_templates import make_prompt
from service.generator import generate

app = FastAPI(title="Post Generator")

class GenReq(BaseModel):
    query: str

@app.post("/generate")
def gen(req: GenReq):
    ctx = build_context_for_query(req.query)
    if not ctx.get("chosen"):
        return {"ok": False, "error": "no_cluster"}
    msg = make_prompt(ctx)
    out = generate(msg)
    return {"ok": True, "result": out, "chosen": ctx["chosen"]}
