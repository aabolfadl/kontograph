// k6 load test for the KONTOGRAPH decision gateway.
//
// Design principle P4: the SLA is measured under load, not with a single-request
// timer in a notebook. A p99 taken from sequential calls on an idle process is
// not a p99 -- it is a best case with a confident-sounding name.
//
//   k6 run loadtest/score.js
//   k6 run -e RATE=500 -e DURATION=60s loadtest/score.js
//
// The thresholds below FAIL THE RUN when breached, which is what makes this a
// gate rather than a report.

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const BASE = __ENV.BASE || 'http://localhost:8000';
const RATE = parseInt(__ENV.RATE || '200', 10);
const DURATION = __ENV.DURATION || '30s';
const SLA_MS = parseInt(__ENV.SLA_MS || '200', 10);

// Server-reported per-stage timings, so the budget table can be reconstructed
// from the load test rather than from a separate measurement.
const stageFeature = new Trend('stage_feature_fetch_ms');
const stageInference = new Trend('stage_inference_ms');
const stageExplain = new Trend('stage_explain_ms');
const stageCounterfactual = new Trend('stage_counterfactual_ms');
const serverTotal = new Trend('server_total_ms');
const alerted = new Rate('alert_rate');

export const options = {
  scenarios: {
    // Arrival rate, not virtual users: a payment rail has an arrival process,
    // and a closed-loop VU model would quietly throttle itself whenever the
    // service slowed down -- hiding exactly the degradation being tested for.
    constant_arrival: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: Math.max(20, Math.ceil(RATE / 4)),
      maxVUs: Math.max(100, RATE * 2),
    },
  },
  thresholds: {
    // The SLA, as a build gate.
    'http_req_duration': [`p(99)<${SLA_MS}`],
    'http_req_failed': ['rate<0.001'],
    'server_total_ms': [`p(99)<${SLA_MS}`],
  },
};

const ACCOUNTS = Array.from({ length: 2000 }, (_, i) => `ACC-${String(i).padStart(8, '0')}`);
const CHANNELS = ['MOBILE_APP', 'ONLINE_BANKING', 'POS', 'STANDING_ORDER'];
const VOP = ['MATCH', 'MATCH', 'MATCH', 'CLOSE_MATCH', 'NO_MATCH'];

function pick(xs) { return xs[Math.floor(Math.random() * xs.length)]; }

export default function () {
  const debtor = pick(ACCOUNTS);
  let creditor = pick(ACCOUNTS);
  while (creditor === debtor) creditor = pick(ACCOUNTS);

  const payload = JSON.stringify({
    tx_id: `k6-${__VU}-${__ITER}-${Date.now()}`,
    acceptance_ts: Date.now(),
    debtor_account_id: debtor,
    creditor_account_id: creditor,
    // Lognormal-ish: a uniform amount would never exercise the large-value
    // paths the cost model cares about.
    amount_cents: Math.floor(Math.exp(Math.random() * 8 + 4)),
    currency: 'EUR',
    vop_result: pick(VOP),
    vop_name_similarity: Math.random(),
    vop_overridden: Math.random() < 0.05,
    is_new_beneficiary: Math.random() < 0.15,
    is_cross_border: Math.random() < 0.05,
    channel: pick(CHANNELS),
    settlement_status: 'ACSC',
  });

  const response = http.post(`${BASE}/score`, payload, {
    headers: { 'Content-Type': 'application/json' },
  });

  const ok = check(response, {
    'status 200': (r) => r.status === 200,
    'has a decision': (r) => r.json('decision') !== undefined,
    'within SLA': (r) => r.json('within_sla') === true,
  });

  if (ok && response.status === 200) {
    const latency = response.json('latency_ms');
    if (latency) {
      serverTotal.add(latency.total);
      stageFeature.add(latency.feature_fetch);
      stageInference.add(latency.inference);
      stageExplain.add(latency.explain || 0);
      stageCounterfactual.add(latency.counterfactual || 0);
    }
    alerted.add(response.json('decision') !== 'ALLOW');
  }
}

export function handleSummary(data) {
  const p99 = data.metrics.http_req_duration.values['p(99)'];
  const verdict = p99 <= SLA_MS ? 'PASS' : 'FAIL';
  const lines = [
    '',
    '  KONTOGRAPH latency budget (k6, server-reported stages)',
    '  ' + '-'.repeat(56),
    `  requests          ${data.metrics.http_reqs.values.count}`,
    `  arrival rate      ${RATE}/s for ${DURATION}`,
    '',
    `  end-to-end p50    ${data.metrics.http_req_duration.values['p(50)'].toFixed(2)} ms`,
    `  end-to-end p95    ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)} ms`,
    `  end-to-end p99    ${p99.toFixed(2)} ms   (SLA ${SLA_MS} ms)  ${verdict}`,
    '',
  ];
  for (const [label, metric] of [
    ['feature_fetch', 'stage_feature_fetch_ms'],
    ['inference', 'stage_inference_ms'],
    ['explain', 'stage_explain_ms'],
    ['counterfactual', 'stage_counterfactual_ms'],
    ['server total', 'server_total_ms'],
  ]) {
    const m = data.metrics[metric];
    if (m) {
      lines.push(`  ${label.padEnd(17)} p99 ${m.values['p(99)'].toFixed(3)} ms`);
    }
  }
  lines.push('');
  lines.push(`  alert rate        ${(data.metrics.alert_rate.values.rate * 100).toFixed(2)}%`);
  lines.push('');

  return {
    stdout: lines.join('\n'),
    'artifacts/loadtest/summary.json': JSON.stringify(data, null, 2),
  };
}
