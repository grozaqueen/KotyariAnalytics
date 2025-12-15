import os
import asyncio
import concurrent.futures as cf
import re
import subprocess, shlex
import grpc
from posts import posts_pb2 as pb
from posts import posts_pb2_grpc as pb_grpc

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
SQLALCHEMY_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

GENERATOR_CMD = os.getenv("GENERATOR_CMD")

try:
    from app.generator_service.main import generate_post  # noqa
except Exception:
    generate_post = None

def build_prompt(user_prompt: str, profile_prompt: str, bot_prompt: str) -> str:
    parts = []
    if bot_prompt: parts.append(f"[BOT]\n{bot_prompt}")
    if profile_prompt: parts.append(f"[PROFILE]\n{profile_prompt}")
    if user_prompt: parts.append(f"[USER]\n{user_prompt}")
    return "\n\n".join(parts).strip()

def generate_post_via_cli(query: str) -> str | None:
    if not GENERATOR_CMD:
        return None
    cmd = f"{GENERATOR_CMD} {shlex.quote(query)}"
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"[generator_cli] failed (code={p.returncode}): {p.stderr[:400]}")
        return None
    out = (p.stdout or "").strip()
    return out or None

def pick_title(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()

    paragraphs: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                paragraphs.append(current)
                current = []

    if current:
        paragraphs.append(current)

    if not paragraphs:
        return ""

    first_paragraph = " ".join(paragraphs[0])
    return first_paragraph.replace("#", "")

def drop_first_paragraph(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    result_lines: list[str] = []

    in_first_paragraph = True
    seen_nonempty_in_first = False

    for line in lines:
        if in_first_paragraph:
            if line.strip():
                seen_nonempty_in_first = True
                continue
            else:
                if seen_nonempty_in_first:
                    in_first_paragraph = False
                continue
        else:
            result_lines.append(line)

    body = "\n".join(result_lines).lstrip("\n")

    body = re.sub(r"\n{2,}", "\n", body)

    body = body.replace("\n", " ")

    body = re.sub(r"\s+", " ", body).strip()

    return body.replace("#", "")

def generate_single(req: pb.GetPostRequest) -> pb.GetPostResponse:
    q = build_prompt(req.user_prompt, req.profile_prompt, req.bot_prompt)
    if not q:
        return pb.GetPostResponse(post_title="", post_text="")

    text: str | None = None

    if generate_post is not None:
        try:
            text = generate_post(q)
        except Exception as e:
            print(f"[generator_fn] error: {e}")

    if text is None:
        text = generate_post_via_cli(q)

    if text is None:
        text = f"(stub) Generated for:\n{q}"

    return pb.GetPostResponse(post_title=pick_title(text), post_text=drop_first_paragraph(text))

class PostsService(pb_grpc.PostsServiceServicer):
    async def GetPost(self, request, context):
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, generate_single, request)
        return res

    async def GetPostsBatch(self, request, context):
        reqs = list(request.posts_request)
        if not reqs:
            return pb.GetPostsResponse(posts_response=[])
        loop = asyncio.get_event_loop()
        with cf.ThreadPoolExecutor(max_workers=min(8, max(1, len(reqs)))) as ex:
            futs = [loop.run_in_executor(ex, generate_single, r) for r in reqs]
            out = await asyncio.gather(*futs)
        return pb.GetPostsResponse(posts_response=out)

async def serve():
    server = grpc.aio.server(options=[
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ("grpc.max_send_message_length",    64 * 1024 * 1024),
    ])
    pb_grpc.add_PostsServiceServicer_to_server(PostsService(), server)
    port = int(os.getenv("GRPC_PORT", "50051"))
    server.add_insecure_port(f"[::]:{port}")
    print(f"[analytics gRPC] :{port}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
