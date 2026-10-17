"""Shared conservative URL and publisher identity helpers for untrusted sources."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

_COMPOUND_PUBLIC_SUFFIXES = {
    "co.uk", "org.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.nz", "org.nz",
    "com.br", "com.cn", "com.sg", "co.jp",
}
_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def safe_https_source_url(value: object, *, max_length: int = 2_000) -> str | None:
    """Return an HTTPS URL only when its authority is syntactically safe.

    This does not prove that the domain is reputable or reachable. The helper
    rejects credentials, explicit ports, IP literals and malformed host labels.
    It performs no network requests.
    """
    if not isinstance(value, str) or not value or len(value) > max_length:
        return None
    try:
        parsed = urlparse(value)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
        ):
            return None
        hostname = parsed.hostname.lower().rstrip(".")
        if not hostname or len(hostname) > 253 or "." not in hostname:
            return None
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            return None
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            return None
        labels = ascii_host.split(".")
        if len(labels) < 2 or any(not _LABEL.fullmatch(label) for label in labels):
            return None
        return value
    except ValueError:
        # Includes malformed ports and malformed bracketed IP-literal syntax.
        return None


def registered_publisher_domain(url: str) -> str:
    """Approximate a publisher's registered domain conservatively.

    This intentionally uses a small suffix list rather than claiming full Public
    Suffix List coverage. Unsupported multi-label suffixes may under-count
    independence; they must not be treated as a complete domain-ownership proof.
    """
    if safe_https_source_url(url) is None:
        raise ValueError("url must be a syntactically safe public HTTPS source URL")
    host = urlparse(url).hostname.lower().rstrip(".")
    labels = host.split(".")
    last_two = ".".join(labels[-2:])
    if len(labels) >= 3 and last_two in _COMPOUND_PUBLIC_SUFFIXES:
        return ".".join(labels[-3:])
    return last_two
