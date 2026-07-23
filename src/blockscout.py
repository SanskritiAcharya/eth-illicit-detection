"""Thin client for the Blockscout Ethereum GraphQL API.

Endpoint: https://eth.blockscout.com/api/v1/graphql  (free, no API key, no signup)

Everything this project knows about an address is fetched through here. We take
only *labels* from external lists; every feature is derived from this raw data.

Seven quirks were measured against the live API on 2026-08-13. Each one is handled
below, and each is commented where it is handled, because they are not in the docs:

1. A browser User-Agent is mandatory. Python's default urllib/requests UA gets
   403 Forbidden. curl works without it; Python does not.
2. The query-complexity limit is 100. Cost scales with `first`, so for our
   8-field transaction node the maximum page size is `first: 8` (cost 96).
   `first: 9` costs 108 and is rejected.
3. Asking for `cursor` or `pageInfo.endCursor` pushes the same query over the
   limit. Cursors are plain base64 of "arrayconnection:<index>", so we build
   them ourselves and never pay for the field.
4. `transactionsCount` is often null, so we never rely on it; we count by paging.
5. Mega-addresses (exchanges, 100k+ txs) hang the request, so pages per address
   are capped and slow requests are abandoned rather than retried forever.
6. `gasUsed` is unreliable: it comes back as "0" on many ordinary transfers that
   certainly burned 21000 gas. Since quirk 2 leaves room for only eight fields,
   `gasUsed` is the one we drop, in favour of `gas` (the limit), which is always
   populated and costs the same.
7. There is an undocumented rate limit of 500 GraphQL requests per roughly
   fifteen minutes, and it is not announced until you hit it. A short burst runs
   happily at 4 requests per second, which is misleading: it is quietly spending
   a budget that then leaves every later request answered with 429 for a quarter
   of an hour. The server does expose `x-ratelimit-remaining` and
   `x-ratelimit-reset` (milliseconds until the window resets), so the RateLimiter
   below reads them and paces the collection to fit, instead of racing ahead and
   stalling.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request

ENDPOINT = "https://eth.blockscout.com/api/v1/graphql"
REST_BASE = "https://eth.blockscout.com"

# Quirk 1: without a browser User-Agent every request from Python is 403.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# Quirk 2: eight fields at `first: 8` costs 96 of the 100 complexity budget.
# Adding a ninth field costs 104 and is rejected, so the field list below is full.
# `gasUsed` is the one we chose to drop, because quirk 6 makes it untrustworthy
# anyway; `gas` (the limit) is always populated and costs the same.
PAGE_SIZE = 8
TX_FIELDS = "hash fromAddressHash toAddressHash value gas gasPrice blockNumber status"


class BlockscoutError(RuntimeError):
    """Raised when the API returns GraphQL errors we cannot recover from."""


class RateLimiter:
    """Keeps the collection inside the server's advertised request budget.

    Quirk 7: the API allows roughly 500 GraphQL requests per fifteen-minute
    window. Rather than guess at a safe delay, this reads the budget the server
    reports on every response and spreads the remaining requests evenly over the
    time left in the window.

    The effect is that collection settles into a steady, polite pace instead of
    sprinting into a wall of 429s. It is shared across threads, so the pace is
    global rather than per worker.
    """

    def __init__(self, reserve: int = 15):
        # Leave a few requests unspent so a burst of retries never fully drains
        # the window and blocks everything else.
        self.reserve = reserve
        self.remaining: int | None = None
        self.reset_at: float = 0.0
        self._lock = threading.Lock()

    def observe(self, headers) -> None:
        """Record the budget reported on a response."""
        try:
            remaining = headers.get("x-ratelimit-remaining")
            reset_ms = headers.get("x-ratelimit-reset")
            if remaining is None or reset_ms is None:
                return
            with self._lock:
                self.remaining = int(remaining)
                self.reset_at = time.time() + int(reset_ms) / 1000.0
        except (TypeError, ValueError):
            return

    def wait(self) -> None:
        """Block until it is safe to send the next request."""
        with self._lock:
            remaining = self.remaining
            reset_at = self.reset_at

        if remaining is None:
            return

        seconds_left = max(reset_at - time.time(), 0.0)
        usable = remaining - self.reserve

        if usable <= 0:
            # Budget spent: nothing to do but wait for the window to roll over.
            if seconds_left > 0:
                time.sleep(min(seconds_left + 1.0, 900))
            with self._lock:
                self.remaining = None
            return

        # Spread what is left evenly across the time remaining in the window.
        if seconds_left > 0:
            time.sleep(min(seconds_left / usable, 5.0))


limiter = RateLimiter()


def _cursor(index: int) -> str:
    """Build the opaque cursor for a given zero-based edge index.

    Quirk 3: the API's cursors are just base64("arrayconnection:<n>"), verified by
    decoding real ones. Constructing them keeps the query inside the complexity
    budget, because we never have to request the `cursor` field itself.
    """
    return base64.b64encode(f"arrayconnection:{index}".encode()).decode()


def graphql(
    query: str,
    timeout: int = 30,
    retries: int = 4,
    backoff: float = 2.0,
    max_rate_limit_waits: int = 4,
) -> dict:
    """POST a GraphQL query, pacing against the rate limit and retrying failures."""
    payload = json.dumps({"query": query}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    last_error: Exception | None = None

    attempt = 0
    rate_limit_waits = 0
    while attempt < retries:
        limiter.wait()
        request = urllib.request.Request(ENDPOINT, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                limiter.observe(response.headers)
                body = json.loads(response.read())
            if "errors" in body:
                # Complexity or validation errors are our bug, not the network's;
                # retrying them just wastes a free public service's time.
                raise BlockscoutError(body["errors"][0].get("message", "unknown"))
            return body["data"]
        except BlockscoutError:
            raise
        except urllib.error.HTTPError as exc:
            limiter.observe(exc.headers)
            if exc.code == 429:
                # Quirk 7. The response carries the milliseconds until the window
                # resets, so we wait exactly that long rather than guessing.
                # Waiting out a rate limit is not a failed attempt -- the request
                # was never really tried -- so it does not consume the retry
                # budget. It is capped separately so a permanently throttled key
                # cannot wedge the collection forever.
                last_error = exc
                rate_limit_waits += 1
                if rate_limit_waits > max_rate_limit_waits:
                    break
                wait_for = 60.0
                try:
                    wait_for = min(int(exc.headers.get("x-ratelimit-reset", 60000)) / 1000 + 1, 900)
                except (TypeError, ValueError):
                    pass
                time.sleep(wait_for)
                continue
            last_error = exc
            attempt += 1
            if attempt < retries:
                time.sleep(backoff * attempt)
        except Exception as exc:  # timeouts, 5xx, connection resets
            last_error = exc
            attempt += 1
            if attempt < retries:
                time.sleep(backoff * attempt)

    raise BlockscoutError(f"gave up after {attempt} tries: {last_error}")


def fetch_transactions(address: str, max_pages: int = 5, pause: float = 0.05) -> list[dict]:
    """Fetch up to ``max_pages * PAGE_SIZE`` transactions for one address.

    Blockscout returns transactions newest-first and includes both directions
    (the address as sender and as receiver), which is exactly what we need to
    build an address-to-address graph.

    Quirk 5: `max_pages` is the guard against mega-addresses. An exchange wallet
    with 784k transactions will never finish, so we take a bounded, most-recent
    sample and record how much we took, rather than hanging the whole collection.
    """
    transactions: list[dict] = []
    offset = 0

    for _ in range(max_pages):
        after = f', after: "{_cursor(offset - 1)}"' if offset else ""
        query = (
            f'{{ address(hash: "{address}") {{ '
            f"transactions(first: {PAGE_SIZE}{after}) {{ "
            f"edges {{ node {{ {TX_FIELDS} }} }} "
            f"pageInfo {{ hasNextPage }} }} }} }}"
        )

        data = graphql(query)
        node = (data or {}).get("address")
        if not node or not node.get("transactions"):
            break

        connection = node["transactions"]
        edges = connection.get("edges") or []
        transactions.extend(edge["node"] for edge in edges)
        offset += len(edges)

        if not edges or not connection["pageInfo"]["hasNextPage"]:
            break
        if pause:
            time.sleep(pause)

    return transactions


def fetch_address_meta(addresses: list[dict] | list[str]) -> list[dict]:
    """Batch-fetch account-level metadata for up to a handful of addresses.

    `addresses(hashes: [...])` lets us pull balance, nonce and contract status
    for several accounts in one call. `contractCode` is non-null only for smart
    contracts, which is how we tell contracts from externally-owned accounts.

    Quirk 4 applies here too: `transactionsCount` is frequently null, so we fetch
    it for interest but never depend on it downstream.
    """
    hashes = [a if isinstance(a, str) else a["address"] for a in addresses]
    quoted = ", ".join(f'"{h}"' for h in hashes)
    query = (
        f"{{ addresses(hashes: [{quoted}]) {{ "
        f"hash fetchedCoinBalance nonce contractCode transactionsCount }} }}"
    )
    return graphql(query).get("addresses") or []


def fetch_block_timestamp(number: int) -> str | None:
    """Return the ISO timestamp of a block, used to build a block-to-time map.

    Transactions only carry `blockNumber`. Requesting the nested `block { timestamp }`
    on every transaction is both expensive in complexity terms and prone to
    timing out, so instead we fetch a sparse set of anchor blocks once and
    interpolate between them (see src/blocktime.py).
    """
    data = graphql(f"{{ block(number: {number}) {{ timestamp }} }}")
    block = (data or {}).get("block")
    return block.get("timestamp") if block else None


def rest_block_transactions(number: int, timeout: int = 30) -> list[dict]:
    """List the transactions in a block via Blockscout's REST endpoint.

    This is the one place we step outside GraphQL, and only to build a *sampling
    frame*: the GraphQL schema's Block type exposes no transaction list, so there
    is no way to discover ordinary addresses through it. We use this purely to
    draw random licit candidate addresses from a given era; every feature those
    addresses contribute is still fetched through GraphQL.
    """
    url = f"{REST_BASE}/api/v2/blocks/{number}/transactions"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()).get("items") or []
    except Exception:
        return []
