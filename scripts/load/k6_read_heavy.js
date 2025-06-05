import http from 'k6/http';
import { sleep, check } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

const SLO_SUCCESS_RATE = 0.999; // 99.9%
const SLO_P95 = 400; // ms

export const options = {
  stages: [
    { duration: '5s', target: 5 },
    { duration: '20s', target: 5 },
    { duration: '5s', target: 0 },
  ],
  summaryTrendStats: ['avg', 'p(95)', 'p(99)', 'min', 'max'],
  thresholds: {
    http_req_failed: [`rate<=${1 - SLO_SUCCESS_RATE}`],
    http_req_duration: [`p(95)<=${SLO_P95}`],
  },
};

const BASE_URL = __ENV.OSIRIS_URL || 'http://localhost:8000';

export default function () {
  const health = http.get(`${BASE_URL}/health`);
  check(health, { 'health status 200': (r) => r.status === 200 });

  const metrics = http.get(`${BASE_URL}/metrics`);
  check(metrics, { 'metrics status 200': (r) => r.status === 200 });

  sleep(1);
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data, { enableColors: true }),
  };
}
