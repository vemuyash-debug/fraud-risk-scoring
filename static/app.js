const form = document.getElementById('txn-form');
const results = document.getElementById('results');
const fraudScore = document.getElementById('fraud-score');
const riskBadge = document.getElementById('risk-badge');
const summary = document.getElementById('summary');
const shapBars = document.getElementById('shap-bars');
const modelMeta = document.getElementById('model-meta');
const modelInfo = document.getElementById('model-info');

function payloadFromForm() {
  const fd = new FormData(form);
  const raw = Object.fromEntries(fd.entries());
  return {
    amount: Number(raw.amount),
    hour: Number(raw.hour),
    is_weekend: Number(raw.is_weekend),
    txn_type: raw.txn_type,
    balance_delta_origin: Number(raw.balance_delta_origin),
    balance_delta_dest: Number(raw.balance_delta_dest),
    velocity_1h: Number(raw.velocity_1h),
    velocity_24h: Number(raw.velocity_24h),
    amount_zscore: Number(raw.amount_zscore),
    merchant_risk_score: Number(raw.merchant_risk_score),
    device_trust_score: Number(raw.device_trust_score),
    geo_distance_km: Number(raw.geo_distance_km),
  };
}

function renderResult(data, withReasons = false) {
  results.hidden = false;
  fraudScore.textContent = `${(data.fraud_score * 100).toFixed(1)}%`;
  riskBadge.textContent = data.risk_level;
  riskBadge.className = `badge ${data.risk_level}`;
  summary.textContent = withReasons ? data.summary : `Risk level: ${data.risk_level}. Flagged: ${data.is_flagged ? 'yes' : 'no'}.`;
  modelMeta.textContent = `Model ${data.model_version}`;

  shapBars.innerHTML = '';
  if (withReasons && data.top_reasons) {
    const max = Math.max(...data.top_reasons.map((r) => Math.abs(r.shap_value)), 0.001);
    data.top_reasons.forEach((r) => {
      const pct = (Math.abs(r.shap_value) / max) * 100;
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `
        <div class="bar-label"><span>${r.feature}</span><span>${r.shap_value.toFixed(4)}</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      `;
      shapBars.appendChild(row);
    });
  }
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

document.getElementById('score-btn').addEventListener('click', async () => {
  try {
    const data = await post('/predict', payloadFromForm());
    renderResult(data, false);
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById('explain-btn').addEventListener('click', async () => {
  try {
    const data = await post('/explain', payloadFromForm());
    renderResult(data, true);
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById('sample-btn').addEventListener('click', () => {
  const fields = {
    amount: 48500,
    hour: 3,
    is_weekend: 1,
    txn_type: 'transfer',
    balance_delta_origin: -48500,
    balance_delta_dest: 48200,
    velocity_1h: 11,
    velocity_24h: 38,
    amount_zscore: 3.4,
    merchant_risk_score: 0.81,
    device_trust_score: 0.22,
    geo_distance_km: 240,
  };
  Object.entries(fields).forEach(([k, v]) => {
    const el = form.elements.namedItem(k);
    if (el) el.value = v;
  });
});

async function loadModelInfo() {
  try {
    const data = await fetch('/model/info').then((r) => r.json());
    modelInfo.textContent = JSON.stringify(data, null, 2);
  } catch {
    modelInfo.textContent = 'Model not trained yet. Run: python scripts/train_all.py';
  }
}

loadModelInfo();
