/**
 * Dashboard wiring: fetch from the API, render each panel, keep the controls
 * in sync. Charts come from ./charts.js; everything here is plain DOM.
 */
import { BarChart, DispatchChart, PALETTE, TimeSeriesChart, formatNumber } from './charts.js';

/* --------------------------------------------------------------- helpers */

const euro = new Intl.NumberFormat('en-GB', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 0,
});
const pct = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${(value * 100).toFixed(digits)}%`;
const signedPct = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`;

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function api(path) {
  const response = await fetch(`/api${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ?? `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function showError(message) {
  const banner = $('#error-banner');
  banner.innerHTML = message;
  banner.hidden = false;
}

function kpi(label, value, { sub, tone } = {}) {
  const node = document.createElement('div');
  node.className = 'kpi';
  node.innerHTML = `
    <div class="kpi-label">${label}</div>
    <div class="kpi-value${tone ? ` ${tone}` : ''}">${value}</div>
    ${sub ? `<div class="kpi-sub">${sub}</div>` : ''}`;
  return node;
}

function shortDate(value) {
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(value));
}

/** Wire a .button-group so exactly one button is active, then call back. */
function buttonGroup(root, onSelect, attribute) {
  $$('button', root).forEach((button) => {
    button.addEventListener('click', () => {
      $$('button', root).forEach((b) => b.classList.toggle('is-active', b === button));
      onSelect(button.dataset[attribute]);
    });
  });
}

/* ----------------------------------------------------------------- tabs */

buttonGroup(
  $('#tabs'),
  (tab) => {
    $$('[data-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.panel !== tab;
    });
    // Charts sized while hidden measure zero; nudge them once visible.
    window.dispatchEvent(new Event('resize'));
  },
  'tab',
);

/* ------------------------------------------------------------- dataset */

const SERIES_COLUMNS = [
  { key: 'demand_mw', label: 'Demand', unit: 'MW', family: 'power', color: PALETTE[0], on: true },
  {
    key: 'price_eur_mwh',
    label: 'Price',
    unit: 'EUR/MWh',
    family: 'price',
    color: PALETTE[1],
    on: true,
  },
  { key: 'solar_mw', label: 'Solar', unit: 'MW', family: 'power', color: PALETTE[2], on: false },
  { key: 'wind_mw', label: 'Wind', unit: 'MW', family: 'power', color: PALETTE[4], on: false },
  {
    key: 'net_load_mw',
    label: 'Net load',
    unit: 'MW',
    family: 'power',
    color: PALETTE[5],
    on: false,
  },
  {
    key: 'temperature_c',
    label: 'Temperature',
    unit: '°C',
    family: 'temp',
    color: PALETTE[3],
    on: false,
  },
];

const state = {
  rangeDays: 30,
  selected: new Set(SERIES_COLUMNS.filter((c) => c.on).map((c) => c.key)),
  datasetEnd: null,
};

let seriesChart;

async function renderSeries() {
  if (!state.datasetEnd || !state.selected.size) return;

  const columns = SERIES_COLUMNS.filter((c) => state.selected.has(c.key));
  // A year of hourly points is finer than the chart can resolve, so long
  // ranges are aggregated server-side rather than shipped and thrown away.
  const resolution = state.rangeDays > 120 ? 'day' : 'hour';
  const from = new Date(state.datasetEnd.getTime() - state.rangeDays * 86400_000);

  const payload = await api(
    `/series?from=${from.toISOString()}&columns=${columns.map((c) => c.key).join(',')}` +
      `&resolution=${resolution}`,
  );

  // Split the axes by unit family: the first selected family owns the left
  // axis, anything else shares the right one.
  const primary = columns[0].family;
  const series = columns.map((column) => ({
    name: `${column.label} (${column.unit})`,
    values: payload.rows.map((row) => row[column.key]),
    color: column.color,
    axis: column.family === primary ? 'left' : 'right',
  }));

  const config = {
    timestamps: payload.rows.map((row) => row.timestamp),
    series,
    formatValue: (value, s) => formatNumber(value, 1) + (s.name.includes('°C') ? '' : ''),
  };

  if (seriesChart) seriesChart.update(config);
  else seriesChart = new TimeSeriesChart($('#series-chart'), config);

  $('#series-note').textContent =
    `${payload.count.toLocaleString()} points · ${resolution === 'day' ? 'daily means' : 'hourly'}` +
    ` · ${shortDate(payload.from)} to ${shortDate(payload.to)}`;
}

function renderColumnToggles() {
  const host = $('#column-toggles');
  host.replaceChildren(
    ...SERIES_COLUMNS.map((column) => {
      const chip = document.createElement('button');
      chip.className = `chip${state.selected.has(column.key) ? ' is-active' : ''}`;
      chip.style.color = state.selected.has(column.key) ? column.color : '';
      chip.innerHTML = `<span class="chip-swatch" style="background:${column.color}"></span>${column.label}`;
      chip.addEventListener('click', () => {
        if (state.selected.has(column.key)) {
          // Keep at least one series on screen.
          if (state.selected.size === 1) return;
          state.selected.delete(column.key);
        } else {
          state.selected.add(column.key);
        }
        renderColumnToggles();
        renderSeries().catch((err) => showError(err.message));
      });
      return chip;
    }),
  );
}

async function initOverview() {
  const overview = await api('/overview');
  state.datasetEnd = new Date(`${overview.span.end}Z`);

  const { demand_mw: demand, price_eur_mwh: price, renewables } = overview.stats;
  $('#overview-kpis').replaceChildren(
    kpi('Hours of data', overview.span.hours.toLocaleString(), {
      sub: `${shortDate(`${overview.span.start}Z`)} → ${shortDate(`${overview.span.end}Z`)}`,
    }),
    kpi('Peak demand', `${formatNumber(demand.max, 0)} MW`, {
      sub: `mean ${formatNumber(demand.mean, 0)} MW`,
    }),
    kpi('Mean price', `${price.mean.toFixed(1)}`, { sub: 'EUR/MWh' }),
    kpi('Mean daily spread', `${price.mean_daily_spread.toFixed(1)}`, {
      sub: 'EUR/MWh — the arbitrage opportunity',
    }),
    kpi('Negative-price hours', price.negative_hours.toLocaleString(), {
      sub: `${pct(price.negative_share, 1)} of the year`,
      tone: 'negative',
    }),
    kpi('Renewable share', pct(renewables.solar_share + renewables.wind_share, 0), {
      sub: `peaks at ${pct(renewables.max_instant_share, 0)} of load`,
    }),
  );

  renderColumnToggles();
  await renderSeries();

  new TimeSeriesChart($('#profile-chart'), {
    timestamps: overview.hourly_profile.hour.map(
      (hour) => `2024-01-01T${String(hour).padStart(2, '0')}:00:00`,
    ),
    series: [
      { name: 'Demand (MW)', values: overview.hourly_profile.demand_mw, color: PALETTE[0], area: true },
      {
        name: 'Price (EUR/MWh)',
        values: overview.hourly_profile.price_eur_mwh,
        color: PALETTE[1],
        axis: 'right',
      },
    ],
  });

  new TimeSeriesChart($('#spread-chart'), {
    timestamps: overview.daily.dates,
    series: [
      { name: 'Daily max', values: overview.daily.price_max, color: PALETTE[3] },
      { name: 'Daily mean', values: overview.daily.price_mean, color: PALETTE[1] },
      { name: 'Daily min', values: overview.daily.price_min, color: PALETTE[0] },
    ],
  });
}

buttonGroup(
  $('#range-buttons'),
  (days) => {
    state.rangeDays = Number(days);
    renderSeries().catch((err) => showError(err.message));
  },
  'range',
);

/* ------------------------------------------------------------ forecast */

const forecastState = { target: 'demand', zoom: 168, data: null };
let forecastChart;
let importanceChart;
let errorHourChart;

function renderForecastChart() {
  const { data, zoom } = forecastState;
  if (!data) return;

  const { timestamps, actual, predicted } = data.series;
  const start = zoom > 0 ? Math.max(timestamps.length - zoom, 0) : 0;

  const config = {
    timestamps: timestamps.slice(start),
    series: [
      { name: `Actual (${data.unit})`, values: actual.slice(start), color: PALETTE[0], width: 2 },
      {
        name: `Forecast (${data.unit})`,
        values: predicted.slice(start),
        color: PALETTE[1],
        dashed: true,
        width: 2,
      },
    ],
    formatValue: (value) => `${formatNumber(value, 1)} ${data.unit}`,
  };

  if (forecastChart) forecastChart.update(config);
  else forecastChart = new TimeSeriesChart($('#forecast-chart'), config);
}

async function renderForecast() {
  const data = await api(`/forecast/${forecastState.target}`);
  forecastState.data = data;

  const { metrics, baseline } = data;
  $('#forecast-kpis').replaceChildren(
    kpi('WMAPE', pct(metrics.wmape, 2), { sub: 'weighted mean abs. % error' }),
    kpi('Skill vs naive', signedPct(data.skill_vs_naive, 1), {
      sub: `baseline WMAPE ${pct(baseline.wmape, 2)}`,
      tone: data.skill_vs_naive > 0 ? 'positive' : 'negative',
    }),
    kpi('Bias', signedPct(metrics.bias, 2), {
      sub: metrics.bias >= 0 ? 'over-forecasts on average' : 'under-forecasts on average',
    }),
    kpi('MAE', `${formatNumber(metrics.mae, 1)}`, { sub: data.unit }),
    kpi('RMSE', `${formatNumber(metrics.rmse, 1)}`, { sub: data.unit }),
    kpi('Test hours', metrics.n_test.toLocaleString(), { sub: `${data.feature_count} features` }),
  );

  $('#forecast-window').textContent =
    `Trained on ${shortDate(`${data.train_span.start}Z`)} → ${shortDate(`${data.train_span.end}Z`)}` +
    ` (${data.train_span.hours.toLocaleString()} h), tested on the following ` +
    `${data.test_span.hours.toLocaleString()} h. Chronological split — the model never sees the test period.`;

  renderForecastChart();

  const importance = data.feature_importance;
  const importanceConfig = {
    labels: importance.map((row) => row.feature),
    values: importance.map((row) => row.share),
    horizontal: true,
    color: PALETTE[0],
    formatValue: (value) => pct(value, 1),
  };
  if (importanceChart) importanceChart.update(importanceConfig);
  else importanceChart = new BarChart($('#importance-chart'), importanceConfig);

  const errorConfig = {
    labels: data.error_by_hour.hour.map((h) => String(h).padStart(2, '0')),
    values: data.error_by_hour.mae,
    color: PALETTE[3],
    formatValue: (value) => `${formatNumber(value, 2)} ${data.unit}`,
  };
  if (errorHourChart) errorHourChart.update(errorConfig);
  else errorHourChart = new BarChart($('#error-hour-chart'), errorConfig);
}

buttonGroup(
  $('#forecast-target'),
  (target) => {
    forecastState.target = target;
    renderForecast().catch((err) => showError(err.message));
  },
  'target',
);

buttonGroup(
  $('#forecast-zoom'),
  (zoom) => {
    forecastState.zoom = Number(zoom);
    renderForecastChart();
  },
  'zoom',
);

/* ----------------------------------------------------------- backtesting */

const REGIME_GLOSSARY = {
  temporal: 'Chronological split — train on the past, predict the future. The honest default.',
  random:
    'Random holdout across the whole history. Optimistic by construction: the model sees hours on either side of every test hour. The gap to temporal shows how much it relies on interpolating within familiar conditions.',
  recency_weighted:
    'Chronological, with training rows exponentially down-weighted by age (30-day half-life). Beating temporal here means the series has drifted and old data is actively misleading.',
  short_horizon:
    'Chronological, evaluated only on the next 24 hours — the horizon a day-ahead dispatch decision actually needs.',
  ablation:
    'Chronological, refit with one feature group removed. Shows which features are load-bearing rather than decorative; a group whose removal barely moves WMAPE is not earning its complexity.',
};

function regimeKey(name) {
  if (name.startsWith('ablation_')) return 'ablation';
  if (name.startsWith('short_horizon')) return 'short_horizon';
  return name;
}

async function renderBacktest(target) {
  const data = await api(`/backtest/${target}`);
  const rows = data.regimes;
  const worstWmape = Math.max(...rows.map((row) => row.wmape));

  $('#backtest-table tbody').replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement('tr');
      if (row.regime === 'temporal') tr.className = 'is-highlight';

      const label = row.regime.startsWith('ablation_')
        ? `${row.regime.replace('ablation_without_', 'without ')} <span class="sub-label">${row.n_features} features</span>`
        : row.regime.replace(/_/g, ' ');

      tr.innerHTML = `
        <td class="row-label">${label}</td>
        <td class="num bar-cell">
          <div class="bar-fill" style="width:${(row.wmape / worstWmape) * 100}%"></div>
          <span>${pct(row.wmape, 2)}</span>
        </td>
        <td class="num ${Math.abs(row.bias) > 0.02 ? 'negative' : ''}">${signedPct(row.bias, 2)}</td>
        <td class="num">${formatNumber(row.mae, 1)}</td>
        <td class="num">${formatNumber(row.rmse, 1)}</td>
        <td class="num">${pct(row.baseline_wmape, 2)}</td>
        <td class="num ${row.skill_vs_naive > 0 ? 'positive' : 'negative'}">${signedPct(row.skill_vs_naive, 1)}</td>
        <td class="num">${row.n_test.toLocaleString()}</td>`;
      return tr;
    }),
  );

  const seen = new Set(rows.map((row) => regimeKey(row.regime)));
  $('#regime-glossary').replaceChildren(
    ...[...seen].flatMap((key) => {
      const dt = document.createElement('dt');
      dt.textContent = key.replace(/_/g, ' ');
      const dd = document.createElement('dd');
      dd.textContent = REGIME_GLOSSARY[key] ?? '';
      return [dt, dd];
    }),
  );
}

buttonGroup(
  $('#backtest-target'),
  (target) => {
    renderBacktest(target).catch((err) => showError(err.message));
  },
  'target',
);

/* -------------------------------------------------------------- dispatch */

const dispatchState = { data: null, strategy: 'forecast_driven' };
let dispatchChart;

function renderDispatchWeek() {
  const { data, strategy } = dispatchState;
  if (!data) return;

  const week = data.demo_week;
  const schedule = week.strategies[strategy];

  const config = {
    timestamps: week.timestamps,
    bars: [
      { name: 'Charge (MWh)', values: schedule.charge_mwh, color: PALETTE[0] },
      { name: 'Discharge (MWh)', values: schedule.discharge_mwh.map((v) => -v), color: PALETTE[2] },
    ],
    lines: [
      { name: 'State of charge (MWh)', values: schedule.soc_mwh, color: PALETTE[4], axis: 'left' },
      { name: 'Price (EUR/MWh)', values: week.actual_price, color: PALETTE[1], axis: 'right' },
    ],
    formatValue: (value, s) => `${formatNumber(value, 2)}${s.name.includes('Price') ? '' : ''}`,
  };

  if (dispatchChart) dispatchChart.update(config);
  else dispatchChart = new DispatchChart($('#dispatch-chart'), config);
}

function cumulative(values) {
  let total = 0;
  return values.map((value) => (total += value ?? 0));
}

async function renderDispatch() {
  const data = await api('/dispatch');
  dispatchState.data = data;

  const strategies = data.strategies;
  const forecastDriven = strategies.forecast_driven.summary;
  const perfect = strategies.perfect_foresight.summary;
  const fixed = strategies.fixed_rule.summary;
  const battery = data.battery;

  $('#dispatch-kpis').replaceChildren(
    kpi('Forecast-driven profit', euro.format(forecastDriven.profit), {
      sub: `over ${data.window.hours.toLocaleString()} hours`,
      tone: 'positive',
    }),
    kpi('Value captured', pct(forecastDriven.pct_of_perfect, 1), {
      sub: `of the ${euro.format(perfect.profit)} perfect-foresight ceiling`,
    }),
    kpi('Uplift vs fixed rule', euro.format(forecastDriven.profit - fixed.profit), {
      sub: `fixed rule earns ${euro.format(fixed.profit)}`,
      tone: 'positive',
    }),
    kpi('Battery', `${battery.capacity_mwh} MWh`, {
      sub: `${battery.max_power_mw} MW · ${battery.duration_hours}h duration · ${pct(battery.round_trip_efficiency, 0)} round-trip`,
    }),
    kpi('Full cycles', formatNumber(forecastDriven.equivalent_full_cycles, 1), {
      sub: 'equivalent full cycles in the window',
    }),
    kpi('Spread captured', `${formatNumber(forecastDriven.avg_discharge_price - forecastDriven.avg_charge_price, 1)}`, {
      sub: `EUR/MWh — buys at ${formatNumber(forecastDriven.avg_charge_price, 1)}, sells at ${formatNumber(forecastDriven.avg_discharge_price, 1)}`,
    }),
  );

  $('#dispatch-window').innerHTML =
    `${shortDate(`${data.window.start}Z`)} → ${shortDate(`${data.window.end}Z`)} — the same held-out ` +
    'window the price model was tested on, so the forecast driving these decisions never saw it. ' +
    `Solved with ${data.solver}.`;

  $('#strategy-table tbody').replaceChildren(
    ...Object.entries(strategies).map(([key, strategy]) => {
      const summary = strategy.summary;
      const tr = document.createElement('tr');
      if (key === 'forecast_driven') tr.className = 'is-highlight';
      tr.innerHTML = `
        <td class="row-label">${strategy.label}<span class="sub-label">${strategy.description}</span></td>
        <td class="num ${summary.profit >= 0 ? 'positive' : 'negative'}">${euro.format(summary.profit)}</td>
        <td class="num">${pct(summary.pct_of_perfect, 1)}</td>
        <td class="num">${key === 'fixed_rule' ? '—' : euro.format(summary.uplift_vs_fixed_rule)}</td>
        <td class="num">${formatNumber(summary.avg_charge_price, 1)}</td>
        <td class="num">${formatNumber(summary.avg_discharge_price, 1)}</td>
        <td class="num">${formatNumber(summary.equivalent_full_cycles, 1)}</td>`;
      return tr;
    }),
  );

  // Strategy switcher for the week chart, built from whatever the pipeline exported.
  const switcher = $('#dispatch-strategy');
  switcher.replaceChildren(
    ...Object.entries(strategies).map(([key, strategy]) => {
      const button = document.createElement('button');
      button.dataset.strategy = key;
      button.textContent = strategy.label;
      button.className = key === dispatchState.strategy ? 'is-active' : '';
      return button;
    }),
  );
  buttonGroup(
    switcher,
    (strategy) => {
      dispatchState.strategy = strategy;
      renderDispatchWeek();
    },
    'strategy',
  );

  renderDispatchWeek();

  new TimeSeriesChart($('#profit-chart'), {
    timestamps: data.daily_profit.dates,
    series: Object.entries(data.daily_profit.series).map(([key, values], index) => ({
      name: strategies[key]?.label ?? key,
      values: cumulative(values),
      color: [PALETTE[2], PALETTE[0], PALETTE[3]][index] ?? PALETTE[index % PALETTE.length],
    })),
    formatValue: (value) => euro.format(value),
    formatLeft: (value) => euro.format(value),
  });

  if (data.duration_sweep) {
    $('#sweep-table tbody').replaceChildren(
      ...data.duration_sweep.map((row) => {
        const tr = document.createElement('tr');
        if (row.max_power_mw === battery.max_power_mw) tr.className = 'is-highlight';
        tr.innerHTML = `
          <td class="row-label">${row.label}</td>
          <td class="num">${euro.format(row.profit)}</td>
          <td class="num">${formatNumber(row.equivalent_full_cycles, 1)}</td>
          <td class="num">${formatNumber(row.avg_discharge_price - row.avg_charge_price, 1)}</td>`;
        return tr;
      }),
    );
  } else {
    $('#sweep-table').closest('.card').hidden = true;
  }
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  try {
    const metadata = await api('/metadata');
    $('#build-info').innerHTML = `
      artifacts built ${new Date(metadata.generated_at).toISOString().slice(0, 16).replace('T', ' ')} UTC<br />
      ${metadata.rows.toLocaleString()} hourly rows · ${metadata.model.split(',')[0]}`;
  } catch (err) {
    showError(
      `${err.message} — run <code>python scripts/build_dashboard_data.py</code> from the repo root, then reload.`,
    );
    return;
  }

  const panels = [
    ['dataset', initOverview()],
    ['forecast', renderForecast()],
    ['backtest', renderBacktest('demand')],
    ['dispatch', renderDispatch()],
  ];

  // Render every panel independently so one failure does not blank the page.
  for (const [name, promise] of panels) {
    promise.catch((err) => showError(`Failed to load the ${name} panel: ${err.message}`));
  }
}

boot();
