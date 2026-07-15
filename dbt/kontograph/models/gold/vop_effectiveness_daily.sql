{{ config(materialized='table') }}

-- Verification of Payee outcomes, daily.
--
-- Mandatory across the euro area since 2025-10-09 (Reg. EU 2024/886). This mart
-- exists because VoP is simultaneously the highest-signal real-time feature on
-- the German rail and a false-positive machine: umlaut transliteration, married
-- names and trading-vs-registered names all produce genuine mismatches on
-- entirely legitimate payments.
--
-- The operationally important column is the CLOSE_MATCH rate. A bank tuning its
-- VoP threshold is choosing where to sit between blocking honest customers and
-- letting impersonation through, and this is the table that decision is made
-- from. The override rate is the other half: a payer who proceeds past a warning
-- shifts liability, so its base rate among *legitimate* payments is what stops
-- "overridden" being treated as proof of fraud.

select
    event_date,
    count(*)                                                    as payment_count,
    sum(case when vop_result = 'MATCH' then 1 else 0 end)       as match_count,
    sum(case when vop_result = 'CLOSE_MATCH' then 1 else 0 end) as close_match_count,
    sum(case when vop_result = 'NO_MATCH' then 1 else 0 end)    as no_match_count,
    sum(case when vop_result = 'UNAVAILABLE' then 1 else 0 end) as unavailable_count,
    sum(case when vop_overridden then 1 else 0 end)             as override_count,

    round(1.0 * sum(case when vop_result = 'MATCH' then 1 else 0 end) / count(*), 4)
        as match_rate,
    round(1.0 * sum(case when vop_result = 'CLOSE_MATCH' then 1 else 0 end) / count(*), 4)
        as close_match_rate,
    round(1.0 * sum(case when vop_result = 'NO_MATCH' then 1 else 0 end) / count(*), 4)
        as no_match_rate,

    -- Overrides as a share of the warnings actually shown, not of all payments:
    -- dividing by every payment would bury the number and make an alarming
    -- override rate look negligible.
    round(
        1.0 * sum(case when vop_overridden then 1 else 0 end)
        / nullif(sum(case when vop_result in ('CLOSE_MATCH', 'NO_MATCH') then 1 else 0 end), 0),
        4
    ) as override_rate_given_warning,

    -- An UNAVAILABLE spike is a creditor-PSP outage, not a fraud wave. Kept
    -- adjacent so the two cannot be confused on a dashboard.
    round(1.0 * sum(case when vop_result = 'UNAVAILABLE' then 1 else 0 end) / count(*), 4)
        as unavailable_rate,
    round(avg(vop_name_similarity), 4) as mean_name_similarity
from {{ source('silver', 'payments') }}
group by 1
