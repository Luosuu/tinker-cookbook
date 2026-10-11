import io
import json
import urllib.error

import pytest

from .mine import GitHubAPI, GitHubAPIError, github_error_metadata


def test_rate_limit_deadlines_and_missing_headers():
    primary = github_error_metadata(403, 'repos/org/repo?token=secret', {'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1200', 'Retry-After': '300', 'Authorization': 'secret', 'Set-Cookie': 'secret'}, 'API rate limit exceeded', now=1000)
    assert primary['category'] == 'primary_rate_limit'
    assert primary['next_retry_at_epoch'] == 1300
    assert primary['path'] == 'repos/org/repo'
    assert 'secret' not in json.dumps(primary)
    unknown = github_error_metadata(403, 'repos/a', {}, 'API rate limit exceeded', now=1000)
    assert unknown['next_retry_at_epoch'] is None
    secondary = github_error_metadata(403, 'repos/a', {'Retry-After': 'Thu, 01 Jan 1970 00:30:00 GMT'}, 'secondary rate limit', now=1000)
    assert secondary['category'] == 'secondary_rate_limit'
    assert secondary['next_retry_at_epoch'] == 1800
    auth = github_error_metadata(403, 'repos/a', {}, 'Forbidden', now=1000)
    assert auth['category'] == 'http_error'
    malformed = github_error_metadata(429, 'repos/a', {'Retry-After': 'invalid'}, '', now=1000)
    assert malformed['next_retry_at_epoch'] is None


def test_http_errors_record_safe_headers_without_retry(tmp_path, monkeypatch):
    calls = []
    def fail(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '2000', 'Set-Cookie': 'secret'}, io.BytesIO(b'{"message":"API rate limit exceeded"}'))
    monkeypatch.setattr('urllib.request.urlopen', fail)
    api = GitHubAPI('private-token', error_records_dir=tmp_path)
    with pytest.raises(GitHubAPIError) as caught:
        api._request('repos/org/repo', None, 'application/vnd.github+json')
    assert len(calls) == 1
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    recorded = json.loads(files[0].read_text())
    assert recorded == caught.value.response_metadata
    assert 'private-token' not in files[0].read_text()
    assert 'Set-Cookie' not in files[0].read_text()
