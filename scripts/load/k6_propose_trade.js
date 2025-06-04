import http from 'k6/http';
import { sleep, check } from 'k6';
import { SharedArray } from 'k6/data';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

const prompts = new SharedArray('prompts', () => JSON.parse(open('prompts.json')));

export const options = {
  stages: [
    { duration: '5s', target: 3 },
    { duration: '20s', target: 3 },
    { duration: '5s', target: 0 },
  ],
  summaryTrendStats: ['avg', 'p(95)', 'p(99)', 'min', 'max'],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<1500'],
  },
};

const BASE_URL = __ENV.OSIRIS_URL || 'http://localhost:8000';

export default function () {
  const prompt = randomItem(prompts);
  const payload = JSON.stringify({ prompt: prompt, max_length: 64 });
  const params = { headers: { 'Content-Type': 'application/json' } };

  const res = http.post(`${BASE_URL}/propose_trade_adjustments/`, payload, params);
  check(res, { 'pta status 200': (r) => r.status === 200 });

  sleep(1);
}
