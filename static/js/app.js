/**
 * CardioPulse AI - Interactive Dashboard & Inference Engine Client
 */

let globalMetrics = null;
let chartsInitialized = false;
let chartInstances = {};

// Color Palette Constants
const COLORS = {
  cyan: '#38bdf8',
  indigo: '#818cf8',
  emerald: '#34d399',
  amber: '#fbbf24',
  coral: '#f87171',
  slateDark: '#1e293b',
  textMuted: '#94a3b8'
};

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  fetchInitialMetrics();
  // Automatically run baseline prediction on initial load
  submitPrediction();
});

/* ==========================================================================
   TABS NAVIGATION
   ========================================================================== */
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      // Initialize or resize charts if entering comparison or analytics tab
      if (targetId === 'tab-comparison' || targetId === 'tab-analytics') {
        if (!chartsInitialized && globalMetrics) {
          renderAllCharts(globalMetrics);
        } else {
          resizeAllCharts();
        }
      }
    });
  });
}

function resizeAllCharts() {
  Object.values(chartInstances).forEach(chart => {
    if (chart && typeof chart.resize === 'function') {
      chart.resize();
    }
  });
}

/* ==========================================================================
   FORM CONTROLS & LIVE VALUE UPDATES
   ========================================================================== */
function updateSliderVal(field, val, unit) {
  const label = document.getElementById(`val-${field}`);
  if (label) {
    label.textContent = `${val}${unit}`;
  }
}

function getFormData() {
  return {
    age: parseFloat(document.getElementById('input-age').value),
    sex: parseInt(document.getElementById('input-sex').value),
    cp: parseInt(document.getElementById('input-cp').value),
    trestbps: parseFloat(document.getElementById('input-trestbps').value),
    chol: parseFloat(document.getElementById('input-chol').value),
    fbs: parseInt(document.getElementById('input-fbs').value),
    restecg: parseInt(document.getElementById('input-restecg').value),
    thalach: parseFloat(document.getElementById('input-thalach').value),
    exang: parseInt(document.getElementById('input-exang').value),
    oldpeak: parseFloat(document.getElementById('input-oldpeak').value),
    slope: parseInt(document.getElementById('input-slope').value),
    ca: parseInt(document.getElementById('input-ca').value),
    thal: parseInt(document.getElementById('input-thal').value)
  };
}

function setFormData(data) {
  if (!data) return;

  const setVal = (id, val, updateTag = null, unit = '') => {
    const el = document.getElementById(id);
    if (el) {
      el.value = val;
      if (updateTag) {
        updateSliderVal(updateTag, val, unit);
      }
    }
  };

  setVal('input-age', data.age, 'age', ' yrs');
  setVal('input-sex', data.sex);
  setVal('input-cp', data.cp);
  setVal('input-trestbps', data.trestbps, 'trestbps', ' mm Hg');
  setVal('input-chol', data.chol, 'chol', ' mg/dl');
  setVal('input-fbs', data.fbs);
  setVal('input-restecg', data.restecg);
  setVal('input-thalach', data.thalach, 'thalach', ' bpm');
  setVal('input-exang', data.exang);
  setVal('input-oldpeak', data.oldpeak, 'oldpeak', ' mm');
  setVal('input-slope', data.slope);
  setVal('input-ca', data.ca);
  setVal('input-thal', data.thal);
}

function resetForm() {
  const defaults = {
    age: 54, sex: 1, cp: 2, trestbps: 130, chol: 240,
    fbs: 0, restecg: 0, thalach: 150, exang: 0, oldpeak: 1.0,
    slope: 1, ca: 0, thal: 2
  };
  setFormData(defaults);
  submitPrediction();
}

/* ==========================================================================
   PRESET LOADERS
   ========================================================================== */
async function loadPreset(presetKey) {
  try {
    const res = await fetch('/api/sample-patients');
    const presets = await res.json();
    if (presets && presets[presetKey]) {
      setFormData(presets[presetKey].data);
      submitPrediction();
    }
  } catch (err) {
    console.error('Failed to load preset:', err);
  }
}

async function loadRandomPreset() {
  try {
    const res = await fetch('/api/sample-patients');
    const presets = await res.json();
    if (presets && presets.random_uci) {
      setFormData(presets.random_uci.data);
      submitPrediction();
      
      const adviceEl = document.getElementById('clinical-advice-text');
      if (adviceEl && presets.random_uci.actual_target !== undefined) {
        const actualLabel = presets.random_uci.actual_target === 1 ? 'Heart Disease (Positive)' : 'Normal / No Disease (Negative)';
        setTimeout(() => {
          adviceEl.innerHTML += `<br><strong style="color: var(--cyan-light);">Ground Truth in UCI Dataset:</strong> ${actualLabel}`;
        }, 150);
      }
    }
  } catch (err) {
    console.error('Failed to load random UCI patient:', err);
  }
}

/* ==========================================================================
   PREDICTION SUBMISSION & RESULTS RENDERING
   ========================================================================== */
async function submitPrediction() {
  const btn = document.getElementById('btn-predict');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="pulse-dot"></span> Analyzing Biomarkers...`;
  }

  const patientData = getFormData();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientData)
    });

    if (!res.ok) {
      throw new Error(`Prediction API failed with status ${res.status}`);
    }

    const result = await res.json();
    renderPredictionResults(result);

  } catch (err) {
    console.error('Prediction error:', err);
    alert('Error running risk prediction: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg> Calculate Cardiac Risk`;
    }
  }
}

function renderPredictionResults(res) {
  const ensemble = res.ensemble;
  const models = res.models;

  // 1. Update Circular Risk Gauge
  const percent = ensemble.risk_percentage;
  const percText = document.getElementById('gauge-percentage-text');
  if (percText) {
    animateCount(percText, percent, '%');
  }

  const circle = document.getElementById('gauge-fill-circle');
  if (circle) {
    const totalCircumference = 534; // 2 * PI * 85 approx 534
    const offset = totalCircumference - (totalCircumference * (percent / 100));
    circle.style.strokeDasharray = `${totalCircumference}`;
    circle.style.strokeDashoffset = `${offset}`;
    circle.style.stroke = ensemble.risk_color;
  }

  // 2. Risk Badge & Advice
  const badge = document.getElementById('risk-badge');
  if (badge) {
    badge.textContent = ensemble.risk_tier;
    badge.className = 'risk-badge ' + (ensemble.risk_tier.toLowerCase().includes('low') ? 'low' : ensemble.risk_tier.toLowerCase().includes('mod') ? 'moderate' : 'high');
  }

  const advice = document.getElementById('clinical-advice-text');
  if (advice) {
    advice.textContent = ensemble.clinical_advice;
  }

  // 3. Tri-Model Consensus Cards
  const updateModelCard = (prefix, modelName, barColor) => {
    const m = models[modelName];
    if (!m) return;
    const probEl = document.getElementById(`prob-${prefix}`);
    const verdictEl = document.getElementById(`verdict-${prefix}`);
    const progEl = document.getElementById(`progress-${prefix}`);

    const p = Math.round(m.probability * 100);
    if (probEl) probEl.textContent = `${p}%`;
    if (progEl) {
      progEl.style.width = `${p}%`;
      progEl.style.background = barColor;
    }
    if (verdictEl) {
      const isDisease = m.prediction === 1;
      verdictEl.textContent = isDisease ? 'Ischemia Likely' : 'Normal / Low Risk';
      verdictEl.style.color = isDisease ? COLORS.coral : COLORS.emerald;
    }
  };

  updateModelCard('lr', 'Logistic Regression', COLORS.cyan);
  updateModelCard('svm', 'SVM', COLORS.indigo);
  updateModelCard('rf', 'Random Forest', COLORS.emerald);

  // 4. Personalized Contributing Factors
  const factorsContainer = document.getElementById('factors-container');
  if (factorsContainer) {
    factorsContainer.innerHTML = '';

    const riskFactors = res.risk_factors || [];
    const protectiveFactors = res.protective_factors || [];

    if (riskFactors.length === 0 && protectiveFactors.length === 0) {
      factorsContainer.innerHTML = `<div style="font-size:0.82rem; color:var(--text-muted); text-align:center;">All clinical parameters within standard expected ranges.</div>`;
    } else {
      riskFactors.forEach(rf => {
        const item = document.createElement('div');
        const isSevere = rf.severity === 'High';
        item.className = `factor-item ${isSevere ? 'risk-high' : 'risk-mod'}`;
        item.innerHTML = `
          <div>
            <div class="factor-name">${isSevere ? '⚠️' : '⚡'} ${rf.factor}</div>
            <div style="font-size: 0.72rem; opacity: 0.85;">${rf.description}</div>
          </div>
          <div class="factor-value">${rf.value}</div>
        `;
        factorsContainer.appendChild(item);
      });

      protectiveFactors.forEach(pf => {
        const item = document.createElement('div');
        item.className = 'factor-item protective';
        item.innerHTML = `
          <div>
            <div class="factor-name">🛡️ ${pf.factor}</div>
            <div style="font-size: 0.72rem; opacity: 0.85;">${pf.description}</div>
          </div>
          <div class="factor-value">${pf.value}</div>
        `;
        factorsContainer.appendChild(item);
      });
    }
  }
}

function animateCount(el, target, suffix = '') {
  const duration = 600;
  const start = 0;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const current = Math.floor(progress * target);
    el.textContent = `${current}${suffix}`;
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      el.textContent = `${target}${suffix}`;
    }
  }

  requestAnimationFrame(update);
}

/* ==========================================================================
   METRICS & VISUAL ANALYTICS FETCH
   ========================================================================== */
async function fetchInitialMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('Failed to load metrics.json');
    globalMetrics = await res.json();
    
    // Update top header stat if available
    const rfAcc = globalMetrics?.model_comparison?.['Random Forest']?.test_metrics?.accuracy;
    if (rfAcc) {
      const headerAcc = document.getElementById('header-best-acc');
      if (headerAcc) headerAcc.textContent = `${(rfAcc * 100).toFixed(1)}%`;
      const statAcc = document.getElementById('stat-rf-acc');
      if (statAcc) statAcc.textContent = `${(rfAcc * 100).toFixed(2)}%`;
    }

    // Default confusion matrix view
    switchConfusionMatrix('Random Forest');

  } catch (err) {
    console.error('Error loading initial metrics:', err);
  }
}

/* ==========================================================================
   INTERACTIVE CONFUSION MATRIX SWITCHER
   ========================================================================== */
function switchConfusionMatrix(modelName, btnElement = null) {
  if (!globalMetrics || !globalMetrics.model_comparison) return;

  const modelData = globalMetrics.model_comparison[modelName];
  if (!modelData || !modelData.confusion_matrix) return;

  const cm = modelData.confusion_matrix;
  document.getElementById('cm-tn').textContent = cm.tn;
  document.getElementById('cm-fp').textContent = cm.fp;
  document.getElementById('cm-fn').textContent = cm.fn;
  document.getElementById('cm-tp').textContent = cm.tp;
  document.getElementById('cm-total').textContent = `${cm.total} patients`;

  // Toggle active button state
  if (btnElement) {
    const buttons = document.querySelectorAll('.cm-toggle-btn');
    buttons.forEach(b => b.classList.remove('active'));
    btnElement.classList.add('active');
  }
}

/* ==========================================================================
   CHART.JS VISUAL ANALYTICS INITIALIZATION
   ========================================================================== */
function renderAllCharts(metrics) {
  if (chartsInitialized) return;
  chartsInitialized = true;

  Chart.defaults.color = COLORS.textMuted;
  Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

  renderModelComparisonChart(metrics.model_comparison);
  renderRocCurvesChart(metrics.model_comparison);
  renderFeatureImportanceChart(metrics.feature_importances);
  renderDatasetAnalyticsCharts(metrics.visual_analytics);
}

// 1. Model Comparison Bar Chart
function renderModelComparisonChart(models) {
  const canvas = document.getElementById('chart-model-comparison');
  if (!canvas || !models) return;

  const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];
  const lr = models['Logistic Regression']?.test_metrics || {};
  const svm = models['SVM']?.test_metrics || {};
  const rf = models['Random Forest']?.test_metrics || {};

  chartInstances.comparison = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Logistic Regression',
          data: [lr.accuracy * 100, lr.precision * 100, lr.recall * 100, lr.f1_score * 100, lr.roc_auc * 100],
          backgroundColor: 'rgba(56, 189, 248, 0.85)',
          borderColor: '#38bdf8',
          borderWidth: 1,
          borderRadius: 6
        },
        {
          label: 'SVM (RBF Kernel)',
          data: [svm.accuracy * 100, svm.precision * 100, svm.recall * 100, svm.f1_score * 100, svm.roc_auc * 100],
          backgroundColor: 'rgba(129, 140, 248, 0.85)',
          borderColor: '#818cf8',
          borderWidth: 1,
          borderRadius: 6
        },
        {
          label: 'Random Forest',
          data: [rf.accuracy * 100, rf.precision * 100, rf.recall * 100, rf.f1_score * 100, rf.roc_auc * 100],
          backgroundColor: 'rgba(52, 211, 153, 0.85)',
          borderColor: '#34d399',
          borderWidth: 1,
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 14, font: { weight: '600' } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`
          }
        }
      },
      scales: {
        y: {
          min: 60,
          max: 100,
          ticks: { callback: v => `${v}%` },
          grid: { color: 'rgba(148, 163, 184, 0.08)' }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

// 2. Interactive ROC Curves Chart
function renderRocCurvesChart(models) {
  const canvas = document.getElementById('chart-roc-curves');
  if (!canvas || !models) return;

  const getPoints = (mName) => {
    const pts = models[mName]?.roc_curve || [];
    return pts.map(p => ({ x: p.fpr, y: p.tpr }));
  };

  chartInstances.roc = new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        {
          label: `Logistic Regression (AUC = ${models['Logistic Regression']?.test_metrics?.roc_auc || '0.951'})`,
          data: getPoints('Logistic Regression'),
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.05)',
          borderWidth: 2.5,
          tension: 0.1,
          pointRadius: 2,
          fill: false
        },
        {
          label: `SVM (AUC = ${models['SVM']?.test_metrics?.roc_auc || '0.947'})`,
          data: getPoints('SVM'),
          borderColor: '#818cf8',
          borderWidth: 2.5,
          tension: 0.1,
          pointRadius: 2,
          fill: false
        },
        {
          label: `Random Forest (AUC = ${models['Random Forest']?.test_metrics?.roc_auc || '0.956'})`,
          data: getPoints('Random Forest'),
          borderColor: '#34d399',
          borderWidth: 3,
          tension: 0.1,
          pointRadius: 3,
          fill: false
        },
        {
          label: 'Random Chance Baseline',
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: '#64748b',
          borderDash: [6, 6],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: FPR=${ctx.raw.x.toFixed(3)}, TPR=${ctx.raw.y.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: 1,
          title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: COLORS.textMuted },
          grid: { color: 'rgba(148, 163, 184, 0.08)' }
        },
        y: {
          min: 0,
          max: 1.02,
          title: { display: true, text: 'True Positive Rate (Sensitivity)', color: COLORS.textMuted },
          grid: { color: 'rgba(148, 163, 184, 0.08)' }
        }
      }
    }
  });
}

// 3. Feature Importance Horizontal Bar Chart
function renderFeatureImportanceChart(importances) {
  const canvas = document.getElementById('chart-feature-importance');
  if (!canvas || !importances) return;

  const rfData = importances['Random Forest'] || [];
  const topFeatures = rfData.slice(0, 10);
  const labels = topFeatures.map(item => item.feature);
  const values = topFeatures.map(item => (item.importance * 100));

  chartInstances.importance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean Decrease Impurity (%)',
        data: values,
        backgroundColor: 'rgba(56, 189, 248, 0.85)',
        borderColor: '#38bdf8',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Importance: ${ctx.raw.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { callback: v => `${v}%` }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}

// 4. Dataset Exploratory Analytics Charts
function renderDatasetAnalyticsCharts(analytics) {
  if (!analytics) return;

  // Chest Pain Chart
  const cpCanvas = document.getElementById('chart-cp-dist');
  if (cpCanvas && analytics.cp_stats) {
    const cpLabels = analytics.cp_stats.map(s => s.label);
    const cpRates = analytics.cp_stats.map(s => (s.rate * 100));

    chartInstances.cp = new Chart(cpCanvas, {
      type: 'bar',
      data: {
        labels: cpLabels,
        datasets: [{
          label: 'Heart Disease Rate (%)',
          data: cpRates,
          backgroundColor: [
            'rgba(239, 68, 68, 0.85)',
            'rgba(245, 158, 11, 0.85)',
            'rgba(56, 189, 248, 0.85)',
            'rgba(129, 140, 248, 0.85)'
          ],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Disease Prevalence: ${ctx.raw.toFixed(1)}%`
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: { callback: v => `${v}%` },
            grid: { color: 'rgba(148, 163, 184, 0.08)' }
          },
          x: { grid: { display: false } }
        }
      }
    });
  }

  // Sex Breakdown Chart
  const sexCanvas = document.getElementById('chart-sex-dist');
  if (sexCanvas && analytics.sex_stats) {
    const sexLabels = analytics.sex_stats.map(s => s.sex);
    const sexRates = analytics.sex_stats.map(s => (s.rate * 100));

    chartInstances.sex = new Chart(sexCanvas, {
      type: 'bar',
      data: {
        labels: sexLabels,
        datasets: [{
          label: 'Heart Disease Rate (%)',
          data: sexRates,
          backgroundColor: [
            'rgba(244, 114, 182, 0.85)',
            'rgba(56, 189, 248, 0.85)'
          ],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(1)}% Disease Rate`
            }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: { callback: v => `${v}%` },
            grid: { color: 'rgba(148, 163, 184, 0.08)' }
          },
          x: { grid: { display: false } }
        }
      }
    });
  }
}
