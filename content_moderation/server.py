from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Kotyari Content Moderation")

class ModerationRequest(BaseModel):
    text: str

class ModerationResponse(BaseModel):
    ok: bool
    labels: list[str] = []
    reasons: list[str] = []

def run_moderation(text: str) -> ModerationResponse:
    # TODO: запуск
    return ModerationResponse(ok=True, labels=[], reasons=[])

@app.post("/moderate", response_model=ModerationResponse)
def moderate(req: ModerationRequest):
    return run_moderation(req.text)
