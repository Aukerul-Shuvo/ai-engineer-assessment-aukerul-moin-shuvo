from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.api.errors import UpstreamError, UpstreamRateLimitedError, UpstreamUnavailableError
from app.tools.circuit_breaker import CircuitBreaker
from app.tools.common import ToolError
from app.tools.superhero import (
    Hero,
    HeroSearchResult,
    SuperheroClient,
    get_superhero,
    search_superheroes,
)

TOKEN = "0123456789abcdef"
BASE = "https://superheroapi.com/api"
WWW = "https://www.superheroapi.com/api.php"


def hero_record(hero_id: str, name: str, full_name: str, strength: str = "26") -> dict:
    """A record shaped like the API's documentation sample, with the quirks included."""
    return {
        "id": hero_id,
        "name": name,
        "powerstats": {
            "intelligence": "100",
            "strength": strength,
            "speed": "27",
            "durability": "50",
            "power": "47",
            "combat": "null",
        },
        "biography": {
            "full-name": full_name,
            "alter-egos": "No alter egos found.",
            "aliases": ["Insider", "Matches Malone"],
            "place-of-birth": "Crest Hill, Bristol Township; Gotham County",
            "first-appearance": "Detective Comics #27",
            "publisher": "DC Comics",
            "alignment": "good",
        },
        "appearance": {
            "gender": "Male",
            "race": "Human",
            "height": ["6'2", "188 cm"],
            "weight": ["210 lb", "95 kg"],
            "eye-color": "blue",
            "hair-color": "black",
        },
        "work": {"occupation": "Businessman", "base": "Batcave, Stately Wayne Manor, Gotham City"},
        "connections": {
            "group-affiliation": "Batman Family, Justice League",
            "relatives": "Damian Wayne (son), Dick Grayson (adopted son)",
        },
        "image": {"url": "https://www.superherodb.com/pictures2/portraits/10/100/639.jpg"},
    }


BATMEN = [
    hero_record("69", "Batman", "Terry McGinnis"),
    hero_record("70", "Batman", "Bruce Wayne", strength="26"),
    hero_record("71", "Batman II", "Dick Grayson", strength="16"),
]


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    # Mirrors the lifespan client: redirects must be followed for this API to work at all.
    async with httpx.AsyncClient(follow_redirects=True, timeout=5) as client:
        yield client


@pytest.fixture
def client(http: httpx.AsyncClient) -> SuperheroClient:
    return SuperheroClient(
        http,
        base_url=BASE,
        token=TOKEN,
        max_retries=2,
        retry_wait_max_s=0,
        breaker=CircuitBreaker(failure_threshold=3, recovery_s=60),
    )


# ---------------------------------------------------------------- happy paths and quirks
@respx.mock
async def test_search_follows_the_redirect_and_normalizes_records(
    client: SuperheroClient,
) -> None:
    redirect = respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(302, headers={"location": f"{WWW}/{TOKEN}/search/batman"})
    )
    respx.get(f"{WWW}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(
            200, json={"response": "success", "results-for": "batman", "results": BATMEN}
        )
    )

    heroes = await client.search("batman")

    assert redirect.called
    assert [h.id for h in heroes] == ["69", "70", "71"]
    bruce = heroes[1]
    assert bruce.full_name == "Bruce Wayne"
    assert bruce.powerstats.intelligence == 100
    assert bruce.powerstats.combat is None, '"null" becomes None'
    assert bruce.alter_egos is None, "the API's 'No alter egos found.' becomes None"
    assert bruce.aliases == ["Insider", "Matches Malone"]
    assert bruce.height == ["6'2", "188 cm"]
    assert bruce.image_url.endswith("639.jpg")


@respx.mock
async def test_get_by_id_returns_the_full_record(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/70").mock(
        return_value=httpx.Response(200, json={"response": "success", **BATMEN[1]})
    )

    hero = await client.get("70")

    assert isinstance(hero, Hero)
    assert hero.name == "Batman"
    assert hero.first_appearance == "Detective Comics #27"
    assert hero.group_affiliation == "Batman Family, Justice League"


@respx.mock
async def test_not_found_is_a_normal_empty_result_not_an_error(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/nobody").mock(
        return_value=httpx.Response(
            200, json={"response": "error", "error": "character with given name not found"}
        )
    )
    respx.get(f"{BASE}/{TOKEN}/999999").mock(
        return_value=httpx.Response(200, json={"response": "error", "error": "invalid id"})
    )

    assert await client.search("nobody") == []
    assert await client.get("999999") is None


@respx.mock
async def test_results_are_cached_for_repeat_lookups(client: SuperheroClient) -> None:
    route = respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "success", "results": BATMEN})
    )

    await client.search("batman")
    await client.search("Batman")
    await client.search("  batman ")

    assert route.call_count == 1, "same name in any casing or spacing hits the cache"


# ---------------------------------------------------------------- failures
@respx.mock
async def test_access_denied_names_the_misconfiguration(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "error", "error": "access denied"})
    )

    with pytest.raises(UpstreamError, match="SUPERHERO_API_TOKEN"):
        await client.search("batman")


@respx.mock
async def test_transport_errors_are_retried_then_reported_without_the_token(
    client: SuperheroClient,
) -> None:
    route = respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        side_effect=httpx.ConnectTimeout("timed out", request=None)
    )

    with pytest.raises(UpstreamUnavailableError) as excinfo:
        await client.search("batman")

    assert route.call_count == 3, "one attempt plus two retries"
    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in excinfo.value.message


@respx.mock
async def test_server_errors_are_retried_then_reported_as_unavailable(
    client: SuperheroClient,
) -> None:
    route = respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(503, text="down")
    )

    with pytest.raises(UpstreamUnavailableError, match="HTTP 503"):
        await client.search("batman")

    assert route.call_count == 3


@respx.mock
async def test_rate_limit_maps_to_its_own_error(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(return_value=httpx.Response(429))

    with pytest.raises(UpstreamRateLimitedError):
        await client.search("batman")


@respx.mock
async def test_non_json_body_is_an_upstream_error(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, text="<html>maintenance</html>")
    )

    with pytest.raises(UpstreamError, match="non-JSON"):
        await client.search("batman")


@respx.mock
async def test_circuit_opens_after_repeated_failures_and_stops_calling_upstream(
    http: httpx.AsyncClient,
) -> None:
    client = SuperheroClient(
        http,
        base_url=BASE,
        token=TOKEN,
        max_retries=0,
        retry_wait_max_s=0,
        breaker=CircuitBreaker(failure_threshold=2, recovery_s=60),
    )
    route = respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        side_effect=httpx.ConnectError("refused", request=None)
    )

    for _ in range(2):
        with pytest.raises(UpstreamUnavailableError):
            await client.search("batman")
    assert route.call_count == 2

    with pytest.raises(UpstreamUnavailableError, match="circuit breaker"):
        await client.search("batman")
    assert route.call_count == 2, "the open breaker short-circuits before any HTTP call"


# ---------------------------------------------------------------- tool functions
@respx.mock
async def test_search_tool_summarizes_and_flags_ambiguity(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "success", "results": BATMEN})
    )

    result = await search_superheroes(client, "batman", limit=2)

    assert isinstance(result, HeroSearchResult)
    assert result.count == 3
    assert len(result.results) == 2, "limit caps the payload, count reports the truth"
    assert result.results[1].full_name == "Bruce Wayne"
    assert result.results[1].powerstats.strength == 26
    assert result.note is not None and "3 characters" in result.note
    assert not hasattr(result.results[0], "relatives"), "summaries stay compact"


@respx.mock
async def test_search_tool_turns_zero_matches_into_guidance(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/caped%20crusader").mock(
        return_value=httpx.Response(
            200, json={"response": "error", "error": "character with given name not found"}
        )
    )

    result = await search_superheroes(client, "caped crusader")

    assert isinstance(result, HeroSearchResult)
    assert result.count == 0
    assert result.note is not None and "alias" in result.note


@respx.mock
async def test_tools_return_failures_as_data(client: SuperheroClient) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(return_value=httpx.Response(503))
    respx.get(f"{BASE}/{TOKEN}/404").mock(
        return_value=httpx.Response(200, json={"response": "error", "error": "invalid id"})
    )

    search = await search_superheroes(client, "batman")
    missing = await get_superhero(client, "404")

    assert isinstance(search, ToolError)
    assert "HTTP 503" in search.error
    assert isinstance(missing, ToolError)
    assert "404" in missing.error and missing.hint is not None
