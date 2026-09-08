"""Application entry point.

``create_app`` assembles the FastAPI application: settings, logging, middleware, routes, error
handlers, and the lifespan that owns shared resources. The module-level ``app`` is what uvicorn
serves::

    uvicorn app.main:app --reload

Everything expensive is created once in the lifespan and reached through ``app.state``.
Nothing is rebuilt per request. Responses are serialised straight from Pydantic models, which
FastAPI 0.141 does natively and faster than a custom JSON response class.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import router
from app.config import Settings, get_settings
from app.lifespan import lifespan
from app.observability.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Pass ``settings`` to override the environment, as tests do."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_as_json)

    app = FastAPI(
        title="AI Engineer Assessment",
        version=__version__,
        description=(
            "Ask a natural-language question. The service decides whether the answer lives in "
            "the text corpus, the Superhero API, or both, and returns the answer with sources."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()
