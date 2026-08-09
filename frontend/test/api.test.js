/**
 * API tests. Run with `npm test` (node --test), no test framework needed.
 *
 * The server is started on an ephemeral port so the suite never collides with
 * a dev server already running on 3000.
 */
import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { createApp } from '../server.js';

let server;
let base;

before(async () => {
  server = createApp().listen(0);
  await new Promise((resolve) => server.once('listening', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server?.close());

const get = (path) => fetch(`${base}${path}`);
const getJson = async (path) => {
  const response = await get(path);
  return { status: response.status, body: await response.json() };
};

describe('GET /api/health', () => {
  it('reports the loaded dataset', async () => {
    const { status, body } = await getJson('/api/health');
    assert.equal(status, 200);
    assert.equal(body.status, 'ok');
    assert.ok(body.rows > 8000, `expected a year of hours, got ${body.rows}`);
    assert.match(body.start, /^\d{4}-\d{2}-\d{2}T/);
  });
});

describe('GET /api/series', () => {
  it('returns every numeric column by default', async () => {
    const { status, body } = await getJson('/api/series?from=2024-03-01&to=2024-03-02');
    assert.equal(status, 200);
    assert.equal(body.resolution, 'hour');
    assert.equal(body.count, 25); // inclusive of both endpoints
    assert.ok(body.columns.includes('demand_mw'));
    assert.ok(body.columns.includes('price_eur_mwh'));
    assert.equal(typeof body.rows[0].demand_mw, 'number');
  });

  it('restricts to the requested columns', async () => {
    const { body } = await getJson('/api/series?from=2024-03-01&to=2024-03-02&columns=demand_mw');
    assert.deepEqual(body.columns, ['demand_mw']);
    assert.deepEqual(Object.keys(body.rows[0]).sort(), ['demand_mw', 'timestamp']);
  });

  it('aggregates to daily means', async () => {
    const hourly = await getJson('/api/series?from=2024-03-01&to=2024-03-07&columns=demand_mw');
    const daily = await getJson(
      '/api/series?from=2024-03-01&to=2024-03-07&columns=demand_mw&resolution=day',
    );

    assert.equal(daily.body.count, 7);

    // The first daily bucket must equal the mean of that day's 24 hours.
    const firstDay = hourly.body.rows.filter((row) => row.timestamp.startsWith('2024-03-01'));
    const expected = firstDay.reduce((sum, row) => sum + row.demand_mw, 0) / firstDay.length;
    assert.ok(
      Math.abs(daily.body.rows[0].demand_mw - expected) < 0.01,
      `daily mean ${daily.body.rows[0].demand_mw} != hourly mean ${expected}`,
    );
  });

  it('slices the same rows regardless of the host timezone', async () => {
    // A bare timestamp must be read as UTC, not local time.
    const { body } = await getJson(
      '/api/series?from=2024-03-01T00:00:00&to=2024-03-01T02:00:00&columns=demand_mw',
    );
    assert.equal(body.count, 3);
    assert.equal(body.rows[0].timestamp, '2024-03-01T00:00:00');
    assert.equal(body.rows[2].timestamp, '2024-03-01T02:00:00');
  });

  it('rejects an unknown column', async () => {
    const { status, body } = await getJson('/api/series?columns=nope');
    assert.equal(status, 400);
    assert.match(body.error, /Unknown column/);
  });

  it('rejects an invalid date', async () => {
    const { status } = await getJson('/api/series?from=not-a-date');
    assert.equal(status, 400);
  });

  it('rejects a reversed range', async () => {
    const { status, body } = await getJson('/api/series?from=2024-06-01&to=2024-05-01');
    assert.equal(status, 400);
    assert.match(body.error, /must not be after/);
  });

  it('rejects an unknown resolution', async () => {
    const { status } = await getJson('/api/series?resolution=weekly');
    assert.equal(status, 400);
  });
});

describe('artifact routes', () => {
  it('serves the pipeline metadata', async () => {
    const { status, body } = await getJson('/api/metadata');
    assert.equal(status, 200);
    assert.ok(body.generated_at);
    assert.ok(body.rows > 0);
  });

  it('serves the dataset overview', async () => {
    const { body } = await getJson('/api/overview');
    assert.ok(body.stats.price_eur_mwh.mean_daily_spread > 0);
    assert.equal(body.hourly_profile.hour.length, 24);
    assert.equal(body.daily.dates.length, body.daily.price_mean.length);
  });

  it('serves a forecast with aligned actual/predicted series', async () => {
    const { body } = await getJson('/api/forecast/demand');
    assert.equal(body.target, 'demand_mw');
    assert.equal(body.series.actual.length, body.series.predicted.length);
    assert.equal(body.series.timestamps.length, body.series.actual.length);
    assert.ok(body.metrics.wmape > 0 && body.metrics.wmape < 1);
    assert.equal(body.error_by_hour.hour.length, 24);
    assert.ok(body.feature_importance.length > 0);
  });

  it('serves backtest regimes', async () => {
    const { body } = await getJson('/api/backtest/price');
    assert.equal(body.target, 'price_eur_mwh');
    const names = body.regimes.map((r) => r.regime);
    assert.ok(names.includes('temporal'));
    assert.ok(names.includes('random'));
    assert.ok(names.some((n) => n.startsWith('ablation_')));
  });

  it('serves dispatch results for all three strategies', async () => {
    const { body } = await getJson('/api/dispatch');
    const { strategies, demo_week: week } = body;

    for (const key of ['perfect_foresight', 'forecast_driven', 'fixed_rule']) {
      assert.ok(strategies[key], `missing strategy ${key}`);
      assert.equal(typeof strategies[key].summary.profit, 'number');
      assert.equal(week.strategies[key].charge_mwh.length, week.timestamps.length);
    }

    // Perfect foresight is an upper bound by construction — if this ever
    // fails, the forecast-driven schedule is being settled at forecast
    // prices instead of actual ones.
    assert.ok(
      strategies.perfect_foresight.summary.profit >= strategies.forecast_driven.summary.profit,
      'forecast-driven profit exceeded perfect foresight',
    );
  });

  it('404s on an unknown forecast target', async () => {
    const { status, body } = await getJson('/api/forecast/voltage');
    assert.equal(status, 404);
    assert.match(body.error, /Unknown target/);
  });

  it('404s on an unknown API route', async () => {
    const { status } = await getJson('/api/nothing-here');
    assert.equal(status, 404);
  });
});

describe('static site', () => {
  it('serves the dashboard shell', async () => {
    const response = await get('/');
    assert.equal(response.status, 200);
    const html = await response.text();
    assert.match(html, /Energy Demand Forecasting/);
  });

  it('serves the client modules', async () => {
    for (const asset of ['/js/app.js', '/js/charts.js', '/styles.css']) {
      assert.equal((await get(asset)).status, 200, `${asset} did not load`);
    }
  });
});
