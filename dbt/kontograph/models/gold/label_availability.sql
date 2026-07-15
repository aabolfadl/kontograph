{{ config(materialized='table') }}

-- When each label became usable, and how long it took to arrive.
--
-- This is the mart that makes design principle P7 operational rather than a
-- caveat in a README. Two dates matter and they are not the same:
--
--   event_ts    when the transaction happened
--   decided_ts  when somebody confirmed what it was
--
-- The gap between them is verification latency, and it is typically weeks. A
-- chronological split that filters on event_ts alone will happily train on
-- labels that did not exist yet -- a time-travel bug that looks like excellent
-- model performance right up until production.
--
-- So the rule this mart encodes is: a label is available for training at time T
-- only if decided_ts < T. `is_available_at` below is the function every split
-- should be asking, and having it in the warehouse means the answer is the same
-- for the model, the dashboard, and the auditor.

with labelled as (
    select
        l.label_id,
        l.subject_id                    as tx_id,
        l.label,
        l.source,
        l.sar_filed,
        l.confidence,
        l.event_ts,
        l.decided_ts,
        l.decided_date,
        l.decided_ts - l.event_ts       as verification_latency_ms,
        (l.decided_ts - l.event_ts) / 86400000.0 as verification_latency_days,
        p.amount_cents,
        p.event_date                    as transaction_date,
        p.debtor_account_id,
        p.creditor_account_id,
        p.vop_result,
        p.is_new_beneficiary
    from {{ source('silver', 'labels') }} l
    -- Inner join deliberately: a label whose transaction is not in silver is a
    -- referential defect worth surfacing as a missing row rather than papering
    -- over with a left join and a pile of nulls.
    inner join {{ source('silver', 'payments') }} p
        on l.subject_id = p.tx_id
)

select
    *,
    case
        when label in ('CONFIRMED_FRAUD', 'CONFIRMED_LAUNDERING') then true
        else false
    end as is_positive,
    -- SUSPICIOUS_UNPROVEN is neither. Training on unproven suspicion teaches a
    -- model to imitate the incumbent alerting system rather than to detect
    -- crime, so the evaluation protocol drops those rows entirely.
    case when label = 'SUSPICIOUS_UNPROVEN' then true else false end as is_excluded_from_training,
    -- Bucketed latency, matching the 14/30/60-day retraining experiments the
    -- protocol commits to.
    case
        when (decided_ts - event_ts) / 86400000.0 <= 14 then '0-14d'
        when (decided_ts - event_ts) / 86400000.0 <= 30 then '15-30d'
        when (decided_ts - event_ts) / 86400000.0 <= 60 then '31-60d'
        else '60d+'
    end as latency_bucket
from labelled
