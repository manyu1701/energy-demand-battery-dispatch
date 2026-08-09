/**
 * Reader for the committed hourly sample dataset.
 *
 * The CSV is the single source of truth for the raw series — the Python
 * pipeline exports only model outputs, so the dashboard reads the same file
 * the model was trained on rather than a duplicated copy of it.
 *
 * ~8,800 rows parse in a few milliseconds and sit in memory for the process
 * lifetime, so slicing and downsampling happen per request without any I/O.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { DATA_ROOT } from './paths.js';

const CSV_PATH = path.join(DATA_ROOT, 'sample', 'hourly_energy_sample.csv');

const NUMERIC_COLUMNS = [
  'demand_mw',
  'price_eur_mwh',
  'temperature_c',
  'solar_mw',
  'wind_mw',
  'net_load_mw',
];
const BOOLEAN_COLUMNS = ['is_weekend', 'is_holiday'];

let cache = null;

/**
 * Parse a CSV with no quoted fields.
 *
 * The generator writes plain numeric columns and ISO timestamps, none of
 * which can contain a comma, so a full CSV parser would be dead weight here.
 * Anything richer (a real download with quoted headers) should go through a
 * proper parser instead — see loadDataset's error message.
 */
function parseCsv(text) {
  const lines = text.trim().split('\n');
  const header = lines[0].split(',').map((h) => h.trim());

  if (!header.includes('timestamp')) {
    throw new Error(`Expected a "timestamp" column in ${CSV_PATH}, got: ${header.join(', ')}`);
  }

  const rows = new Array(lines.length - 1);
  for (let i = 1; i < lines.length; i += 1) {
    const parts = lines[i].split(',');
    const row = {};
    for (let c = 0; c < header.length; c += 1) {
      const key = header[c];
      const raw = parts[c];
      if (key === 'timestamp') {
        // pandas writes "2024-01-01 00:00:00"; normalise to ISO so both this
        // file and the pipeline's JSON artifacts speak the same format.
        row.timestamp = raw.replace(' ', 'T');
        // Parsed once here rather than per request. The series carries no
        // zone, and the whole project treats it as UTC.
        row._ms = Date.parse(`${row.timestamp}Z`);
      } else if (BOOLEAN_COLUMNS.includes(key)) {
        row[key] = raw === 'True' || raw === 'true';
      } else {
        row[key] = raw === '' ? null : Number(raw);
      }
    }
    rows[i - 1] = row;
  }
  return { header, rows };
}

export async function loadDataset() {
  if (cache) return cache;

  let text;
  try {
    text = await readFile(CSV_PATH, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') {
      throw new Error(
        `Sample dataset not found at ${CSV_PATH}. ` +
          'Regenerate it with: python scripts/generate_sample_data.py',
      );
    }
    throw err;
  }

  if (text.includes('"')) {
    throw new Error(
      `${CSV_PATH} contains quoted fields, which this minimal parser does not handle. ` +
        'Regenerate it with: python scripts/generate_sample_data.py',
    );
  }

  const { header, rows } = parseCsv(text);
  cache = {
    header,
    rows,
    columns: header.filter((h) => h !== 'timestamp'),
    numericColumns: NUMERIC_COLUMNS.filter((c) => header.includes(c)),
    start: rows.length ? rows[0].timestamp : null,
    end: rows.length ? rows[rows.length - 1].timestamp : null,
    startMs: rows.length ? rows[0]._ms : 0,
    endMs: rows.length ? rows[rows.length - 1]._ms : 0,
  };
  return cache;
}

/** Drop the in-memory copy; used by the tests and the dev watcher. */
export function clearDatasetCache() {
  cache = null;
}

/**
 * Parse a query-string date as UTC.
 *
 * A bare "2024-03-01T06:00:00" is local time per the ECMAScript spec, which
 * would silently shift the slice on a non-UTC host, so an explicit zone is
 * added when the caller did not supply one.
 */
function parseUtc(value, fallbackMs) {
  if (value === undefined || value === null || value === '') return fallbackMs;
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value);
  const normalised = value.replace(' ', 'T');
  const ms = Date.parse(hasZone ? normalised : `${normalised}Z`);
  if (Number.isNaN(ms)) {
    throw Object.assign(new Error(`Invalid date: ${value}`), { status: 400 });
  }
  return ms;
}

/**
 * Aggregate hourly rows to daily means.
 *
 * A year of hourly data is 8,760 points per series — more than a 900px-wide
 * chart can resolve, so the yearly view asks for daily resolution and gets a
 * twelfth of the payload for the same visible detail.
 */
function toDaily(rows, columns) {
  const buckets = new Map();
  for (const row of rows) {
    const day = row.timestamp.slice(0, 10);
    let bucket = buckets.get(day);
    if (!bucket) {
      bucket = { timestamp: `${day}T00:00:00`, _count: 0 };
      for (const col of columns) bucket[col] = 0;
      buckets.set(day, bucket);
    }
    bucket._count += 1;
    for (const col of columns) {
      if (typeof row[col] === 'number') bucket[col] += row[col];
    }
  }

  return [...buckets.values()].map((bucket) => {
    const out = { timestamp: bucket.timestamp };
    for (const col of columns) out[col] = round(bucket[col] / bucket._count, 2);
    return out;
  });
}

function round(value, digits) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/**
 * Return a slice of the dataset.
 *
 * @param {object} options
 * @param {string} [options.from]        ISO date/timestamp, inclusive
 * @param {string} [options.to]          ISO date/timestamp, inclusive
 * @param {string[]} [options.columns]   columns to include (default: all numeric)
 * @param {'hour'|'day'} [options.resolution]
 */
export async function getSeries({ from, to, columns, resolution = 'hour' } = {}) {
  const data = await loadDataset();

  if (resolution !== 'hour' && resolution !== 'day') {
    throw Object.assign(new Error(`resolution must be "hour" or "day", got "${resolution}"`), {
      status: 400,
    });
  }

  const requested = columns?.length ? columns : data.numericColumns;
  const unknown = requested.filter((c) => !data.columns.includes(c));
  if (unknown.length) {
    throw Object.assign(
      new Error(`Unknown column(s): ${unknown.join(', ')}. Available: ${data.columns.join(', ')}`),
      { status: 400 },
    );
  }

  const fromMs = parseUtc(from, data.startMs);
  const toMs = parseUtc(to, data.endMs);
  if (fromMs > toMs) {
    throw Object.assign(new Error('"from" must not be after "to"'), { status: 400 });
  }

  const selected = data.rows.filter((row) => row._ms >= fromMs && row._ms <= toMs);

  const rows =
    resolution === 'day' ? toDaily(selected, requested) : selected.map((row) => pick(row, requested));

  return {
    resolution,
    columns: requested,
    from: new Date(fromMs).toISOString(),
    to: new Date(toMs).toISOString(),
    count: rows.length,
    rows,
  };
}

function pick(row, columns) {
  const out = { timestamp: row.timestamp };
  for (const col of columns) out[col] = row[col];
  return out;
}
