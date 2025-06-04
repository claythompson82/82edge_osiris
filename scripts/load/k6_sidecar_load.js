import http from 'k6/http';
import { sleep, check } from 'k6';

export const options = {
  stages: [
    { duration: '10s', target: 5 },
    { duration: '30s', target: 5 },
    { duration: '5s', target: 0 },
  ],
  summaryTrendStats: ['avg', 'p(95)', 'p(99)', 'min', 'max'],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<1000', 'p(99)<2000'],
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
