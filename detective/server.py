"""REST API and static front end.

The server runs on the investigator's own machine, so its main exposure is a web page
in the same browser reaching localhost. Trusted hosts block DNS rebinding, and the
custom header on every write forces a CORS preflight, which this server never approves.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .authoring import MAX_ARCHIVE_BYTES, CaseDraft, CaseError
from .claude_backend import MODEL
from .groq_backend import groq_model
from .workspace import OFFLINE, Backends, Conflict, NotFound, WorkspaceManager

STATIC = Path(__file__).resolve().parent / "web"
WRITE_HEADER = "x-detective-client"


class RunRequest(BaseModel):
    backend: str = OFFLINE
    auto_leads: bool = True


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def create_app(data_dir: Path | None = None, cases_dir: Path | None = None, backends: Backends | None = None) -> FastAPI:
    data = data_dir or Path(os.environ.get("DETECTIVE_DATA", "data"))
    kwargs = {"cases_dir": cases_dir} if cases_dir else {}
    manager = WorkspaceManager(data, backends=backends, **kwargs)
    app = FastAPI(title="AI Detective", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.manager = manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def require_write_header(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS") and WRITE_HEADER not in request.headers:
            return JSONResponse({"detail": "Missing client header."}, status_code=403)
        return await call_next(request)

    for error_type, status in ((NotFound, 404), (Conflict, 409), (CaseError, 422)):
        app.add_exception_handler(error_type, _handler(status))

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "claude_configured": manager.backends.claude_available(),
            "model": MODEL,
            "groq_configured": manager.backends.groq_available(),
            "groq_model": groq_model(),
        }

    @app.get("/api/cases")
    def list_cases():
        return [w.summary() for w in manager.workspaces.values()]

    @app.post("/api/cases", status_code=201)
    def create_case(draft: CaseDraft):
        return manager.create(draft).summary()

    @app.post("/api/cases/import", status_code=201)
    async def import_case(request: Request):
        length = int(request.headers.get("content-length") or 0)
        if length > MAX_ARCHIVE_BYTES:
            raise CaseError("The archive is larger than 10 MB.")
        return manager.import_zip(await request.body()).summary()

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        return manager.get(case_id).overview()

    @app.delete("/api/cases/{case_id}", status_code=204)
    def delete_case(case_id: str):
        manager.delete(case_id)

    @app.get("/api/cases/{case_id}/state")
    def get_state(case_id: str):
        return manager.get(case_id).state()

    @app.post("/api/cases/{case_id}/runs", status_code=202)
    def start_run(case_id: str, request: RunRequest):
        workspace = manager.get(case_id)
        workspace.start_run(request.backend, request.auto_leads)
        return {"status": workspace.status}

    @app.post("/api/cases/{case_id}/reset", status_code=204)
    def reset(case_id: str):
        manager.get(case_id).reset()

    @app.get("/api/cases/{case_id}/documents")
    def documents(case_id: str):
        return manager.get(case_id).documents()

    @app.get("/api/cases/{case_id}/documents/{doc_id}")
    def document(case_id: str, doc_id: str):
        return manager.get(case_id).document(doc_id)

    @app.get("/api/cases/{case_id}/search")
    def search(case_id: str, q: str = Query(min_length=1, max_length=300)):
        return manager.get(case_id).search(q)

    @app.post("/api/cases/{case_id}/leads/{lead_id}/uncover", status_code=202)
    def uncover(case_id: str, lead_id: str):
        workspace = manager.get(case_id)
        workspace.uncover(lead_id)
        return {"status": workspace.status}

    @app.post("/api/cases/{case_id}/questions")
    def ask(case_id: str, question: Question):
        return manager.get(case_id).ask(question.question).model_dump(mode="json")

    @app.get("/api/cases/{case_id}/report.md", response_class=PlainTextResponse)
    def report(case_id: str):
        headers = {"Content-Disposition": f'attachment; filename="{case_id}_report.md"'}
        return PlainTextResponse(manager.get(case_id).report(), media_type="text/markdown", headers=headers)

    @app.get("/api/cases/{case_id}/evaluation")
    def evaluation(case_id: str):
        return manager.get(case_id).evaluate()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _handler(status: int):
    async def handle(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=status)

    return handle


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
