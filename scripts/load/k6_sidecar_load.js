import http from 'k6/http';
import { sleep, check } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// Service Level Objective thresholds
const SLO_SUCCESS_RATE = 0.999; // 99.9% of requests succeed
const SLO_P95 = 800; // p95 latency in ms
const SLO_P99 = 1200; // p99 latency in ms

export const options = {
  stages: [
    { duration: '10s', target: 5 },
    { duration: '30s', target: 5 },
    { duration: '5s', target: 0 },
  ],
  summaryTrendStats: ['avg', 'p(95)', 'p(99)', 'min', 'max'],
  thresholds: {
    http_req_failed: [`rate<=${1 - SLO_SUCCESS_RATE}`],
    http_req_duration: [
      `p(95)<=${SLO_P95}`,
      `p(99)<=${SLO_P99}`,
    ],
  },
};

const BASE_URL = __ENV.OSIRIS_URL || 'http://localhost:8000';

export default function () {
  const payload = JSON.stringify({ prompt: 'hello', model_id: 'phi3', max_length: 16 });
  const params = { headers: { 'Content-Type': 'application/json' } };

  const genRes = http.post(`${BASE_URL}/generate/`, payload, params);
  check(genRes, { 'generate status 200': (r) => r.status === 200 });

  const metricsRes = http.get(`${BASE_URL}/metrics`);
  check(metricsRes, { 'metrics status 200': (r) => r.status === 200 });

  sleep(1);
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data, { enableColors: true }),
  };
}
