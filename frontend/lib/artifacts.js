/**
 * Loader for the JSON artifacts written by scripts/build_dashboard_data.py.
 *
 * Artifacts are read lazily and cached. If one is missing the error names the
 * command that produces it — a fresh clone that has not run the pipeline
 * should get a fixable message, not a stack trace.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { DASHBOARD_DIR } from './paths.js';

export const ARTIFACTS = {
  metadata: 'metadata.json',
  overview: 'overview.json',
  dispatch: 'dispatch.json',
  'forecast:demand': 'forecast_demand_mw.json',
  'forecast:price': 'forecast_price_eur_mwh.json',
  'backtest:demand': 'backtest_demand_mw.json',
  'backtest:price': 'backtest_price_eur_mwh.json',
};

const cache = new Map();

export async function loadArtifact(key) {
  if (cache.has(key)) return cache.get(key);

  const filename = ARTIFACTS[key];
  if (!filename) {
    throw Object.assign(new Error(`Unknown artifact "${key}"`), { status: 404 });
  }

  const file = path.join(DASHBOARD_DIR, filename);
  let text;
  try {
    text = await readFile(file, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') {
      throw Object.assign(
        new Error(
          `Dashboard artifact ${filename} is missing. ` +
            'Build it with: python scripts/build_dashboard_data.py',
        ),
        { status: 503 },
      );
    }
    throw err;
  }

  const parsed = JSON.parse(text);
  cache.set(key, parsed);
  return parsed;
}

export function clearArtifactCache() {
  cache.clear();
}

/**
 * Map a short URL segment ("demand"/"price") onto an artifact key.
 * Keeps the routes readable without leaking column names into URLs.
 */
export function resolveTarget(kind, target) {
  const key = `${kind}:${target}`;
  if (!(key in ARTIFACTS)) {
    throw Object.assign(
      new Error(`Unknown target "${target}". Expected one of: demand, price`),
      { status: 404 },
    );
  }
  return key;
}
