"""Application entry point.

``create_app`` assembles the FastAPI application: settings, logging, middleware, routes, error
handlers, rate limiting, and the lifespan that owns shared resources. The module-level ``app`` is
what uvicorn serves::

    uvicorn app.main:app --reload

Everything expensive is created once in the lifespan and reached through ``app.state``. Nothing
is rebuilt per request. Responses are serialised straight from Pydantic models, which FastAPI
0.141 does natively and faster than a custom JSON response class.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app import __version__
from app.api.ask import build_ask_router
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.ratelimit import build_limiter, rate_limit_exceeded
from app.api.routes import router as health_router
from app.config import Settings, get_settings
from app.lifespan import lifespan
from app.observability.logging import configure_logging
from app.observability.metrics import configure_metrics
from app.observability.tracing import configure_tracing


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

    limiter = build_limiter(settings)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded)

    app.include_router(health_router)
    app.include_router(build_ask_router(settings, limiter))
    register_exception_handlers(app)

    # Observability last: metrics wraps the app in middleware, tracing instruments the ASGI stack.
    app.state.ask_metrics = configure_metrics(app, settings)
    app.state.tracer_provider = configure_tracing(app, settings)
    return app


app = create_app()
