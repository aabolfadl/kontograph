{{ config(materialized='table') }}

-- Daily flow between institutions, plus the domestic/cross-border split.
--
-- Germany's banking system is federated by design (Sparkassen and
-- Genossenschaftsbanken are regionally bounded networks, not national banks),
-- so "which institution pairs move value, and how much" is a genuinely
-- meaningful question here in a way it is not in a single-bank market. It is
-- also the aggregate a supervisor asks for first.

select
    event_date,
    coalesce(debtor_bank_blz, 'UNKNOWN')   as debtor_bank_blz,
    coalesce(creditor_bank_blz, 'UNKNOWN') as creditor_bank_blz,
    coalesce(debtor_country, 'UNKNOWN')    as debtor_country,
    coalesce(creditor_country, 'UNKNOWN')  as creditor_country,
    is_cross_border,
    is_cross_institution,
    count(*)                                as payment_count,
    sum(amount_cents)                       as total_cents,
    round(avg(amount_cents), 2)             as mean_cents,
    max(amount_cents)                       as max_cents,
    -- Median via DuckDB's exact quantile. The mean of a lognormal payment
    -- distribution is dragged around by the tail, so a corridor summarised by
    -- its mean alone tells a supervisor very little.
    quantile_cont(amount_cents, 0.5)        as median_cents,
    quantile_cont(amount_cents, 0.99)       as p99_cents,
    count(distinct debtor_account_id)       as distinct_debtors,
    count(distinct creditor_account_id)     as distinct_creditors,
    sum(case when settlement_status = 'RJCT' then 1 else 0 end) as rejected_count
from {{ source('silver', 'payments') }}
group by 1, 2, 3, 4, 5, 6, 7
