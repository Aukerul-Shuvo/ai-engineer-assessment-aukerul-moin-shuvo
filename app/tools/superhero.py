"""Superhero API client and the two tool functions built on it.

The API has quirks the rest of the system must never see:

* every request 302-redirects to a ``www`` host, so the HTTP client must follow redirects
* errors arrive as HTTP 200 with ``{"response": "error", "error": "..."}``
* power stats are strings and can be the literal ``"null"``; unknown text fields are ``"-"``
* a name search can match several characters; "batman" returns three

``SuperheroClient`` hides all of that behind ``search`` and ``get``, which return typed records.
It retries transient failures, caches results, and trips a circuit breaker when the upstream is
down. The access token sits in the URL path, so it is scrubbed from every error message and
never logged.

``search_superheroes`` and ``get_superhero`` are the model-facing surface: compact outputs and
failures returned as data, so an agent can react by trying an alias instead of crashing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
import structlog
from cachetools import TTLCache
from pydantic import BaseModel, ConfigDict
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.api.errors import (
    AppError,
    UpstreamError,
    UpstreamRateLimitedError,
    UpstreamUnavailableError,
)
from app.tools.circuit_breaker import CircuitBreaker

log = structlog.get_logger(__name__)

_NOT_FOUND_MESSAGES = {"character with given name not found", "invalid id"}
_ACCESS_DENIED = "access denied"
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
_TOKEN_PLACEHOLDER = "{token}"  # noqa: S105 - not a secret, the literal text that replaces one


# --------------------------------------------------------------------------- records
class PowerStats(BaseModel):
    """The six numeric stats. ``None`` where the API says ``"null"``."""

    model_config = ConfigDict(frozen=True)

    intelligence: int | None = None
    strength: int | None = None
    speed: int | None = None
    durability: int | None = None
    power: int | None = None
    combat: int | None = None


class HeroSummary(BaseModel):
    """What a search returns per match: enough to disambiguate and to compare stats."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    full_name: str | None
    publisher: str | None
    alignment: str | None
    powerstats: PowerStats


class Hero(HeroSummary):
    """The full normalized record for one character."""

    aliases: list[str]
    alter_egos: str | None
    place_of_birth: str | None
    first_appearance: str | None
    gender: str | None
    race: str | None
    height: list[str]
    weight: list[str]
    occupation: str | None
    base: str | None
    group_affiliation: str | None
    relatives: str | None
    image_url: str | None


# --------------------------------------------------------------------------- tool results
class HeroSearchResult(BaseModel):
    """Model-facing search outcome. Zero matches is a normal result, not an error."""

    query: str
    count: int
    results: list[HeroSummary]
    note: str | None = None


class ToolError(BaseModel):
    """Model-facing failure. Returned, not raised, so the agent can decide what to do."""

    error: str
    hint: str | None = None


# --------------------------------------------------------------------------- normalizing
_EMPTY_TEXT = {"", "-", "null", "no alter egos found."}


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _EMPTY_TEXT else text


def _stat(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _texts(value: object) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [t for t in (_text(item) for item in items) if t is not None]


def _summary_from_payload(record: Mapping[str, Any]) -> HeroSummary:
    bio = record.get("biography") or {}
    stats = record.get("powerstats") or {}
    return HeroSummary(
        id=str(record["id"]),
        name=str(record["name"]),
        full_name=_text(bio.get("full-name")),
        publisher=_text(bio.get("publisher")),
        alignment=_text(bio.get("alignment")),
        powerstats=PowerStats.model_validate(
            {key: _stat(stats.get(key)) for key in PowerStats.model_fields}
        ),
    )


def _hero_from_payload(record: Mapping[str, Any]) -> Hero:
    summary = _summary_from_payload(record)
    bio = record.get("biography") or {}
    look = record.get("appearance") or {}
    work = record.get("work") or {}
    links = record.get("connections") or {}
    image = record.get("image") or {}
    return Hero.model_validate(
        {
            **summary.model_dump(),
            "aliases": _texts(bio.get("aliases")),
            "alter_egos": _text(bio.get("alter-egos")),
            "place_of_birth": _text(bio.get("place-of-birth")),
            "first_appearance": _text(bio.get("first-appearance")),
            "gender": _text(look.get("gender")),
            "race": _text(look.get("race")),
            "height": _texts(look.get("height")),
            "weight": _texts(look.get("weight")),
            "occupation": _text(work.get("occupation")),
            "base": _text(work.get("base")),
            "group_affiliation": _text(links.get("group-affiliation")),
            "relatives": _text(links.get("relatives")),
            "image_url": _text(image.get("url")),
        }
    )


# --------------------------------------------------------------------------- client
class _TransientStatusError(Exception):
    """Internal: a 429 or 5xx that tenacity should retry."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class SuperheroClient:
    """Typed, cached, retrying access to the Superhero API."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str,
        token: str,
        cache_ttl_s: int = 600,
        cache_size: int = 512,
        max_retries: int = 2,
        retry_wait_max_s: float = 5.0,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=cache_size, ttl=cache_ttl_s)
        self._max_attempts = max_retries + 1
        self._retry_wait_max_s = retry_wait_max_s
        self._breaker = breaker or CircuitBreaker()

    async def search(self, name: str) -> list[Hero]:
        """Every character matching ``name``. Empty list when the API knows none."""
        key = f"search:{name.strip().lower()}"
        cached: list[Hero] | None = self._cache.get(key)
        if cached is not None:
            return cached
        payload = await self._request(f"search/{quote(name.strip())}")
        heroes = (
            [] if payload is None else [_hero_from_payload(r) for r in payload.get("results", [])]
        )
        self._cache[key] = heroes
        return heroes

    async def get(self, hero_id: str) -> Hero | None:
        """One character by id, or ``None`` if the id is unknown."""
        key = f"id:{hero_id}"
        if key in self._cache:
            hero: Hero | None = self._cache[key]
            return hero
        payload = await self._request(quote(hero_id))
        result = None if payload is None else _hero_from_payload(payload)
        self._cache[key] = result
        return result

    async def _request(self, path: str) -> dict[str, Any] | None:
        """GET ``path`` under the token. ``None`` means "not found"; failures raise ``AppError``."""
        if not self._breaker.allow():
            raise UpstreamUnavailableError("Superhero API is unavailable; circuit breaker is open")

        url = f"{self._base_url}/{self._token}/{path}"
        try:
            response = await self._fetch_with_retries(url)
        except httpx.TransportError as exc:
            self._breaker.record_failure()
            raise UpstreamUnavailableError(
                f"Superhero API unreachable: {self._redact(exc)}"
            ) from None
        except _TransientStatusError as exc:
            self._breaker.record_failure()
            if exc.status_code == 429:
                raise UpstreamRateLimitedError("Superhero API rate limit reached") from None
            raise UpstreamUnavailableError(
                f"Superhero API returned HTTP {exc.status_code}"
            ) from None

        if response.status_code != 200:
            self._breaker.record_failure()
            raise UpstreamError(f"Superhero API returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            self._breaker.record_failure()
            raise UpstreamError("Superhero API returned a non-JSON body") from None

        if not isinstance(payload, dict):
            self._breaker.record_failure()
            raise UpstreamError("Superhero API returned an unexpected JSON shape")

        self._breaker.record_success()
        if payload.get("response") == "success":
            return payload
        message = str(payload.get("error", "unknown error")).strip().lower()
        if message in _NOT_FOUND_MESSAGES:
            return None
        if message == _ACCESS_DENIED:
            raise UpstreamError("Superhero API rejected the token; check SUPERHERO_API_TOKEN")
        raise UpstreamError(f"Superhero API error: {self._redact(message)}")

    async def _fetch_with_retries(self, url: str) -> httpx.Response:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((httpx.TransportError, _TransientStatusError)),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=0.5, max=self._retry_wait_max_s),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    log.warning("superhero_retry", attempt=attempt.retry_state.attempt_number)
                response = await self._http.get(url)
                if response.status_code in _TRANSIENT_STATUSES:
                    raise _TransientStatusError(response.status_code)
        return response

    def _redact(self, value: object) -> str:
        """Replace the token wherever it appears, since httpx puts URLs in its messages."""
        return str(value).replace(self._token, _TOKEN_PLACEHOLDER)


# --------------------------------------------------------------------------- tool functions
async def search_superheroes(
    client: SuperheroClient, name: str, limit: int = 5
) -> HeroSearchResult | ToolError:
    """Find characters by name. Several may match; the note tells the agent what to do next."""
    try:
        heroes = await client.search(name)
    except AppError as exc:
        return ToolError(error=exc.message, hint="The superhero source is unavailable right now.")
    summary_fields = set(HeroSummary.model_fields)
    summaries = [HeroSummary.model_validate(h.model_dump(include=summary_fields)) for h in heroes]
    note = None
    if not summaries:
        note = (
            f"No character named '{name}'. Try an alias, an alternate spelling, "
            "or the character's real name."
        )
    elif len(summaries) > 1:
        note = (
            f"{len(summaries)} characters match '{name}'. Use get_superhero with the id of the "
            "one the question means, or compare their publishers and full names."
        )
    return HeroSearchResult(query=name, count=len(summaries), results=summaries[:limit], note=note)


async def get_superhero(client: SuperheroClient, hero_id: str) -> Hero | ToolError:
    """Full record for one character id returned by ``search_superheroes``."""
    try:
        hero = await client.get(hero_id)
    except AppError as exc:
        return ToolError(error=exc.message, hint="The superhero source is unavailable right now.")
    if hero is None:
        return ToolError(
            error=f"No character with id '{hero_id}'.",
            hint="Use an id from a search_superheroes result.",
        )
    return hero
