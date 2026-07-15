{{ config(materialized='table') }}

-- Per-account, per-day payment activity.
--
-- The serving mart the investigator console reads when an analyst opens an
-- account. Deliberately *descriptive*: it summarises what an account did, and
-- contains no model score and no label. Mixing a prediction into a fact table
-- is how a downstream feature ends up trained on yesterday's prediction.
--
-- Note the union rather than two separate models: an account's behaviour is the
-- combination of what it sent and what it received, and a mart that only shows
-- outflow makes a mule collector look inactive.

with outbound as (
    select
        debtor_account_id                as account_id,
        event_date,
        count(*)                         as sent_count,
        sum(amount_cents)                as sent_cents,
        count(distinct creditor_account_id) as distinct_creditors,
        sum(case when is_new_beneficiary then 1 else 0 end) as new_beneficiary_count,
        sum(case when vop_result <> 'MATCH' then 1 else 0 end) as vop_non_match_count,
        sum(case when vop_overridden then 1 else 0 end) as vop_override_count,
        sum(case when is_cross_border then 1 else 0 end) as cross_border_sent,
        min(acceptance_ts)               as first_sent_ts,
        max(acceptance_ts)               as last_sent_ts
    from {{ source('silver', 'payments') }}
    group by 1, 2
),

inbound as (
    select
        creditor_account_id              as account_id,
        event_date,
        count(*)                         as received_count,
        sum(amount_cents)                as received_cents,
        count(distinct debtor_account_id) as distinct_debtors,
        min(acceptance_ts)               as first_received_ts,
        max(acceptance_ts)               as last_received_ts
    from {{ source('silver', 'payments') }}
    group by 1, 2
),

combined as (
    select
        coalesce(o.account_id, i.account_id) as account_id,
        coalesce(o.event_date, i.event_date) as event_date,
        coalesce(o.sent_count, 0)            as sent_count,
        coalesce(o.sent_cents, 0)            as sent_cents,
        coalesce(o.distinct_creditors, 0)    as distinct_creditors,
        coalesce(o.new_beneficiary_count, 0) as new_beneficiary_count,
        coalesce(o.vop_non_match_count, 0)   as vop_non_match_count,
        coalesce(o.vop_override_count, 0)    as vop_override_count,
        coalesce(o.cross_border_sent, 0)     as cross_border_sent,
        coalesce(i.received_count, 0)        as received_count,
        coalesce(i.received_cents, 0)        as received_cents,
        coalesce(i.distinct_debtors, 0)      as distinct_debtors,
        least(
            coalesce(o.first_sent_ts, 9223372036854775807),
            coalesce(i.first_received_ts, 9223372036854775807)
        ) as first_activity_ts,
        greatest(
            coalesce(o.last_sent_ts, 0),
            coalesce(i.last_received_ts, 0)
        ) as last_activity_ts
    from outbound o
    full outer join inbound i
        on o.account_id = i.account_id
       and o.event_date = i.event_date
)

select
    c.*,
    a.kind          as account_kind,
    a.bank_blz,
    a.country,
    -- Net flow. A pass-through account -- value in, value straight out, near
    -- zero retained -- is the structural signature of a mule, and it is the
    -- kind of thing that only shows up when inbound and outbound sit in the
    -- same row.
    c.received_cents - c.sent_cents as net_cents,
    case
        when c.received_cents > 0
        then round(1.0 * c.sent_cents / c.received_cents, 4)
    end as passthrough_ratio
from combined c
left join {{ source('silver', 'accounts') }} a
    on c.account_id = a.account_id
