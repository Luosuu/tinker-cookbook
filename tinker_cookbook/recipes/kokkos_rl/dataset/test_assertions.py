"""Exact-source repairs for vacuous upstream regression assertions."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from tinker_cookbook.recipes.kokkos_rl.dataset.models import KokkosInstance

# Parenthesize the intended conditional value, preserving every allowed value.
# The original comparison bound more tightly than ?: and selected a nonzero
# integer regardless of stride. No test or assertion is removed by this repair.
REPAIRS = {
    "kokkos__kokkos-8370": (
        "b277466868bf94a32b95ffc3a314b41c020f5033",
        "17c094e30df1eaf331c7fe691b032b7e64db9c6d8d14a94b8a25297000136768",
        "252ecaf4b3df9a7cd88f587f7fb10cb8d33a98ffd0501933ec94bb20ffdbabed",
        tuple(
            (
                f"(l.stride == is_ll ? {extent}lu : 5lu || l.stride == KOKKOS_INVALID_INDEX)",
                f"(l.stride == (is_ll ? {extent}lu : 5lu) || l.stride == KOKKOS_INVALID_INDEX)",
            )
            for extent in (11, 7)
        ),
    ),
    "kokkos__kokkos-8838": (
        "63fd412b488606b557e1456b00f4cf728586faf4",
        "55f29e1206586356f99beb983c27ee8cb3e8b5000e14056e0fd6f7032c85de5e",
        "a91286e07dcb5e628df716e55b63b97e988c6ae3d48a26eca1de042824a2efa6",
        tuple(
            (
                f"(l.stride == is_ll ? {extent}lu : {right})",
                f"(l.stride == (is_ll ? {extent}lu : {right}))",
            )
            for extent in (11, 7)
            for right in ("KOKKOS_INVALID_INDEX", "5lu")
        ),
    ),
}


def repair_assertions(instance: KokkosInstance) -> tuple[KokkosInstance, str]:
    """Return the repaired instance and original annotation digest for role checks."""
    digest = hashlib.sha256(instance.test_patch.encode()).hexdigest()
    entry = REPAIRS.get(instance.instance_id)
    if entry is None or entry[0] != instance.base_commit:
        return instance, digest
    _base, original, repaired, replacements = entry
    if digest == repaired:
        return instance, original
    if digest != original:
        raise ValueError("Reviewed assertion repair has a different hidden test patch")
    patch = instance.test_patch
    for before, after in replacements:
        if patch.count(before) != 1:
            raise ValueError("Reviewed assertion occurrence differs")
        patch = patch.replace(before, after)
    if hashlib.sha256(patch.encode()).hexdigest() != repaired:
        raise ValueError("Repaired assertion payload differs from reviewed digest")
    return replace(instance, test_patch=patch), original
