{{ config(materialized='table') }}

-- The regulatory reporting mart.
--
-- Shaped by three real German/EU thresholds rather than by round numbers, which
-- is the point: a reporting mart whose buckets are 100/1000/10000 because those
-- are nice numbers demonstrates nothing, while one whose buckets are the
-- thresholds that actually trigger obligations demonstrates that someone read
-- the rules.
--
--   EUR    1 000  occasional-transaction customer due diligence
--   EUR   10 000  cash/reporting threshold that structuring is designed to
--                 stay beneath -- hence the "just below" band, which is the
--                 single most diagnostic column in this table
--   EUR   25 000  AnaCredit granular credit reporting threshold
--   EUR  100 000  the SCT Inst cap that was removed in 2025; retained because
--                 the regime change is itself a reportable structural shift
--
-- The 9 000-10 000 band deserves comment. Structuring campaigns crowd just
-- under a reporting threshold, so an elevated count there is a classic
-- indicator. It is *not* a rule: legitimate payments live there too, which is
-- exactly why the count is reported for a human rather than used to alert.

with banded as (
    select
        event_date,
        debtor_country,
        creditor_country,
        is_cross_border,
        amount_cents,
        settlement_status,
        case
            when amount_cents <  100000    then 'under_1k'
            when amount_cents <  900000    then '1k_to_9k'
            when amount_cents < 1000000    then 'just_below_10k'
            when amount_cents < 2500000    then '10k_to_25k'
            when amount_cents < 10000000   then '25k_to_100k'
            else 'over_100k'
        end as amount_band
    from {{ source('silver', 'payments') }}
)

select
    event_date,
    amount_band,
    is_cross_border,
    count(*)                     as payment_count,
    sum(amount_cents)            as total_cents,
    count(distinct debtor_country)   as debtor_countries,
    count(distinct creditor_country) as creditor_countries,
    sum(case when settlement_status = 'RJCT' then 1 else 0 end) as rejected_count,

    -- Share of the day's volume in this band. A structuring campaign shows up
    -- as a shift in this share, not as an absolute count -- absolute counts move
    -- with overall volume and a weekend looks like a drop in everything.
    round(
        1.0 * count(*) / sum(count(*)) over (partition by event_date),
        4
    ) as share_of_day
from banded
group by 1, 2, 3
