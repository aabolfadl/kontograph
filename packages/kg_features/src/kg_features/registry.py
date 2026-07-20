"""The KONTOGRAPH feature set.

These are the features, written once and compiled to all three backends. Each
is here because it corresponds to something an investigator would actually look
at, and the grouping below follows how a case is worked rather than how the
columns happen to be laid out.

Two things are deliberately *absent*, and their absence is the point:

**No feature reads the label, the typology, or the campaign.** Those live in
oracle tables that ``kg_lake`` refuses to open without an explicit flag. There
is no feature here that could reach them even if someone wanted it to.

**No feature reads ``ingest_ts`` or any processing-time column.** Ingest lag
correlates with system load at scoring time, not with the payment. A model that
learns it has learned an artefact of our own pipeline, and the artefact changes
the moment the pipeline is tuned.
"""

from __future__ import annotations

from kg_features.dsl import (
    count,
    distinct,
    maximum,
    mean,
    stddev,
    time_since_last,
    total,
    when,
)
from kg_features.ir import Entity, FeatureSet

# Filters reused across features. Named so the intent is visible at the use site.
NOT_MATCHED = (when("vop_result", "!=", "MATCH"),)
OVERRIDDEN = (when("vop_overridden", "==", True),)
NEW_BENEFICIARY = (when("is_new_beneficiary", "==", True),)
CROSS_BORDER = (when("is_cross_border", "==", True),)
INSTANT_CHANNELS = (when("channel", "in", ["MOBILE_APP", "ONLINE_BANKING"]),)

PAYMENTS_V1 = FeatureSet(
    name="payments",
    version=1,
    description="Behavioural features for SEPA Instant fraud and AML detection.",
    features=(
        # ---------------------------------------------------------- velocity --
        # Burst behaviour. Standard laundering typologies move value in minutes;
        # the legitimate economy almost never does, outside payroll day. These
        # are the features a rules engine would already have, included so the
        # ablation can answer honestly whether the graph adds anything to them.
        count(
            "debtor_txn_1h",
            Entity.DEBTOR,
            "1h",
            description="transfers this account sent in the last hour",
        ),
        count(
            "debtor_txn_24h",
            Entity.DEBTOR,
            "24h",
            description="transfers this account sent in the last day",
        ),
        count(
            "debtor_txn_7d",
            Entity.DEBTOR,
            "7d",
            description="transfers this account sent in the last week",
        ),
        time_since_last(
            "debtor_seconds_since_last",
            Entity.DEBTOR,
            description="milliseconds since this account's previous outbound payment",
        ),
        # ------------------------------------------------------------ value --
        total(
            "debtor_sent_24h",
            Entity.DEBTOR,
            "amount_cents",
            "24h",
            description="total value sent in the last day",
        ),
        total(
            "debtor_sent_7d",
            Entity.DEBTOR,
            "amount_cents",
            "7d",
            description="total value sent in the last week",
        ),
        mean(
            "debtor_mean_amount_30d",
            Entity.DEBTOR,
            "amount_cents",
            "30d",
            description="this account's typical payment size",
        ),
        stddev(
            "debtor_stddev_amount_30d",
            Entity.DEBTOR,
            "amount_cents",
            "30d",
            description=(
                "dispersion of this account's payment sizes; with the mean, this "
                "is what makes 'unusual for THIS account' expressible rather than "
                "'unusual in absolute terms'"
            ),
        ),
        maximum(
            "debtor_max_amount_30d",
            Entity.DEBTOR,
            "amount_cents",
            "30d",
            description="largest payment this account has sent recently",
        ),
        # ------------------------------------------------------- fan-out ------
        # The mule signature. Note the confusable twin: an employer running
        # payroll has enormous fan-out on the last banking day of the month, and
        # the simulator generates it deliberately so this feature cannot be a
        # giveaway.
        distinct(
            "debtor_distinct_creditors_24h",
            Entity.DEBTOR,
            "creditor_account_id",
            "24h",
            description="how many different accounts this one paid in a day",
        ),
        distinct(
            "debtor_distinct_creditors_7d",
            Entity.DEBTOR,
            "creditor_account_id",
            "7d",
            description="how many different accounts this one paid in a week",
        ),
        # ------------------------------------------------------- fan-in -------
        # Keyed by CREDITOR: the collector side of a mule network. Utility
        # providers have legitimately enormous in-degree, which is why this is a
        # feature to be learned rather than a rule to be thresholded.
        count(
            "creditor_txn_1h",
            Entity.CREDITOR,
            "1h",
            description="payments this account received in the last hour",
        ),
        count(
            "creditor_txn_24h",
            Entity.CREDITOR,
            "24h",
            description="payments this account received in the last day",
        ),
        total(
            "creditor_received_24h",
            Entity.CREDITOR,
            "amount_cents",
            "24h",
            description="total value received in the last day",
        ),
        distinct(
            "creditor_distinct_debtors_24h",
            Entity.CREDITOR,
            "debtor_account_id",
            "24h",
            description="how many different accounts paid this one in a day",
        ),
        distinct(
            "creditor_distinct_debtors_7d",
            Entity.CREDITOR,
            "debtor_account_id",
            "7d",
            description="how many different accounts paid this one in a week",
        ),
        time_since_last(
            "creditor_seconds_since_last_inbound",
            Entity.CREDITOR,
            description="milliseconds since this account last received anything",
        ),
        # --------------------------------------------------- relationship -----
        # Keyed by PAIR. This is the one that separates a mule collector from a
        # busy merchant: each individual relationship looks unremarkable, and it
        # is the *newness* of every relationship at once that is the signal.
        count(
            "pair_txn_30d",
            Entity.PAIR,
            "30d",
            description="how often this exact debtor-creditor pair has transacted",
        ),
        total(
            "pair_sent_30d",
            Entity.PAIR,
            "amount_cents",
            "30d",
            description="value moved between this pair in the last 30 days",
        ),
        time_since_last(
            "pair_seconds_since_last",
            Entity.PAIR,
            description="milliseconds since this pair last transacted",
        ),
        # ------------------------------------------ Verification of Payee -----
        # Mandatory across the euro area since 2025-10-09. The *rate* of
        # mismatches for an account is far more informative than any single
        # mismatch: one CLOSE_MATCH is a typo, twenty is a pattern.
        count(
            "debtor_vop_mismatch_7d",
            Entity.DEBTOR,
            "7d",
            where=NOT_MATCHED,
            description="payments this account sent where VoP did not match",
        ),
        count(
            "debtor_vop_override_7d",
            Entity.DEBTOR,
            "7d",
            where=OVERRIDDEN,
            description=(
                "times this account proceeded past a VoP warning; legally "
                "significant, and a strong but NOT conclusive fraud signal - "
                "roughly 40% of honest payers override too"
            ),
        ),
        mean(
            "debtor_mean_vop_similarity_30d",
            Entity.DEBTOR,
            "vop_name_similarity",
            "30d",
            default=1.0,
            description="average payee-name match quality for this account",
        ),
        count(
            "creditor_vop_mismatch_7d",
            Entity.CREDITOR,
            "7d",
            where=NOT_MATCHED,
            description=(
                "inbound payments to this account where the payer's typed name "
                "did not match; the impersonation-target side of APP fraud"
            ),
        ),
        # -------------------------------------------------- new beneficiary ---
        count(
            "debtor_new_beneficiaries_24h",
            Entity.DEBTOR,
            "24h",
            where=NEW_BENEFICIARY,
            description="first-time payees this account paid in a day",
        ),
        count(
            "debtor_new_beneficiaries_7d",
            Entity.DEBTOR,
            "7d",
            where=NEW_BENEFICIARY,
            description="first-time payees this account paid in a week",
        ),
        # ------------------------------------------------------- structural ---
        count(
            "debtor_cross_border_30d",
            Entity.DEBTOR,
            "30d",
            where=CROSS_BORDER,
            description="cross-border payments this account has sent",
        ),
        distinct(
            "debtor_distinct_channels_7d",
            Entity.DEBTOR,
            "channel",
            "7d",
            description=(
                "channel diversity; a compromised account often shows a channel "
                "it has never used before"
            ),
        ),
        count(
            "debtor_instant_channel_1h",
            Entity.DEBTOR,
            "1h",
            where=INSTANT_CHANNELS,
            description="self-service payments in the last hour, excluding standing orders",
        ),
    ),
)

#: Every registered feature set, by name.
FEATURE_SETS: dict[str, FeatureSet] = {"payments": PAYMENTS_V1}

#: What the models use unless told otherwise.
DEFAULT_FEATURE_SET = PAYMENTS_V1


def get_feature_set(name: str = "payments") -> FeatureSet:
    try:
        return FEATURE_SETS[name]
    except KeyError:
        raise KeyError(f"unknown feature set {name!r}; available: {sorted(FEATURE_SETS)}") from None
