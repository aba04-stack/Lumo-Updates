from __future__ import annotations

import pytest

from adaptive_core.source_safety import registered_publisher_domain, safe_https_source_url


@pytest.mark.parametrize("url", [
    "http://example.org/article",
    "https://user@example.org/article",
    "https://user:pass@example.org/article",
    "https://example.org:8443/article",
    "https://127.0.0.1/article",
    "https://10.0.0.5/article",
    "https://localhost/article",
    "https://bad host.example/article",
    "https://bad_label.example/article",
    "https://example.org:invalid/article",
    "https://example.org:443/article",
    "https://example.org..bad/article",
])
def test_unsafe_or_malformed_source_urls_are_rejected(url: str) -> None:
    assert safe_https_source_url(url) is None


@pytest.mark.parametrize("value", [None, "", 123, "x" * 2_001])
def test_url_type_and_length_limits(value: object) -> None:
    assert safe_https_source_url(value) is None


def test_syntactically_valid_https_urls_are_accepted() -> None:
    assert safe_https_source_url("https://source-a.example/claim") == "https://source-a.example/claim"
    assert safe_https_source_url("https://xn--bcher-kva.example/article") is not None


def test_publisher_identity_uses_known_compound_suffixes() -> None:
    assert registered_publisher_domain("https://en.wikipedia.org/wiki/Claim") == "wikipedia.org"
    assert registered_publisher_domain("https://ru.wikipedia.org/wiki/Claim") == "wikipedia.org"
    assert registered_publisher_domain("https://www.publisher.example.co.uk/article") == "example.co.uk"


def test_publisher_identity_rejects_unsafe_url() -> None:
    with pytest.raises(ValueError, match="public HTTPS"):
        registered_publisher_domain("http://example.org/article")


def test_idna_and_punycode_variants_share_the_same_publisher_identity() -> None:
    unicode_domain = registered_publisher_domain("https://bücher.example/article")
    ascii_domain = registered_publisher_domain("https://xn--bcher-kva.example/other")
    assert unicode_domain == ascii_domain == "xn--bcher-kva.example"
