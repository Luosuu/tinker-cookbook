import pytest

from .pin_dependency import pin


class API:
    def __init__(self):
        self.calls = []

    async def json(self, path, params=None):
        self.calls.append((path, params))
        if path.startswith("/repos/kokkos/kokkos-kernels/commits/"):
            return {"commit": {"committer": {"date": "2022-03-04T05:06:07Z"}}}
        return [{"sha": "d" * 40}]


@pytest.mark.asyncio
async def test_pins_only_unpinned_kernels_rows():
    api = API()
    rows = [
        {"repo": "kokkos/kokkos-kernels", "base_commit": "b" * 40, "metadata": {}},
        {
            "repo": "kokkos/kokkos-kernels",
            "base_commit": "c" * 40,
            "metadata": {"kokkos_dependency_commit": "e" * 40},
        },
        {"repo": "kokkos/kokkos", "base_commit": "f" * 40, "metadata": {}},
    ]
    pinned = await pin(rows, api)
    assert pinned[0]["metadata"]["kokkos_dependency_commit"] == "d" * 40
    assert pinned[1]["metadata"]["kokkos_dependency_commit"] == "e" * 40
    assert "kokkos_dependency_commit" not in pinned[2]["metadata"]
    assert api.calls[1] == (
        "/repos/kokkos/kokkos/commits",
        {"sha": "develop", "until": "2022-03-04T05:06:07Z", "per_page": 1},
    )
