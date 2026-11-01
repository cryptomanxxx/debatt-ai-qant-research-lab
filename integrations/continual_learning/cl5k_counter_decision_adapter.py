"""CL-5K disconnected integration adapter: class-first quota/duplicate decision.

The actual generator is unchanged. The adapter takes an already computed
class-quota count and a *lazy* duplicate lookup, not a feature vector. It does
not sample, generate datasets, run models or mutate the generator's used set.
"""
from __future__ import annotations

from collections.abc import Callable

from integrations.continual_learning.cl5j_disconnected_counter_ledger import (
    ProposalCounterLedger,
)


def account_proposed_candidate(
    ledger: ProposalCounterLedger, *, remaining_for_label: int,
    duplicate_lookup_if_open: Callable[[], bool],
) -> str:
    """Count a candidate after feature/label creation, before acceptance logic.

    The class-full branch never calls duplicate_lookup_if_open, exactly as the
    generator's short-circuit OR expression behaves. A real integration must
    separately perform the existing used.add/append/decrement on acceptance;
    this adapter neither replaces nor performs those mutations.
    """
    if not isinstance(ledger, ProposalCounterLedger):
        raise ValueError("expected disconnected CL-5J ledger")
    if type(remaining_for_label) is not int or remaining_for_label < 0:
        raise ValueError("remaining class quota must be a nonnegative int")
    if not callable(duplicate_lookup_if_open):
        raise ValueError("duplicate lookup must be callable")
    ledger.begin_proposal()
    if remaining_for_label == 0:
        return ledger.resolve_proposal(
            class_full=True, duplicate_if_class_open=None
        )
    duplicate = duplicate_lookup_if_open()
    if type(duplicate) is not bool:
        raise ValueError("duplicate lookup must return bool")
    return ledger.resolve_proposal(
        class_full=False, duplicate_if_class_open=duplicate
    )
