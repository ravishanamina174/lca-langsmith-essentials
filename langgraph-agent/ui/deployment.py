"""Self-discovery of the server's own public URL.

The UI runs as a custom route *inside* the LangGraph server, and talks back to
that same server through the LangGraph SDK. It cannot know its own URL at import
time (locally it depends on the ``--port`` you passed; in a deployment it is
assigned after the build), so it derives it from the incoming request.

Behind a load balancer TLS is terminated upstream, so ``X-Forwarded-Proto`` and
``X-Forwarded-Host`` win over the raw request URL when present.
"""

from __future__ import annotations

from starlette.requests import Request


def get_deployment_url(request: Request) -> str:
    """Derive the server's public URL from the incoming request headers."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    return f"{scheme}://{host}"
