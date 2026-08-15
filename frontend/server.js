/**
 * Dashboard server for the energy demand + battery dispatch project.
 *
 * Serves two kinds of data:
 *   /api/series      slices of the raw hourly dataset, read from the same CSV
 *                    the model was trained on
 *   /api/{overview,forecast,backtest,dispatch}
 *                    model outputs, precomputed by
 *                    scripts/build_dashboard_data.py
 *
 * No model runs at request time — the server is a viewer, so it starts in
 * milliseconds and needs no Python environment.
 *
 *   npm install && npm start      then open http://localhost:3000
 */
import express from 'express';

import { loadArtifact, resolveTarget } from './lib/artifacts.js';
import { getSeries, loadDataset } from './lib/dataset.js';
import { PUBLIC_DIR } from './lib/paths.js';

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? '127.0.0.1';

export function createApp() {
  const app = express();
  app.disable('x-powered-by');

  const api = express.Router();

  api.get('/health', async (_req, res, next) => {
    try {
      const data = await loadDataset();
      res.json({ status: 'ok', rows: data.rows.length, start: data.start, end: data.end });
    } catch (err) {
      next(err);
    }
  });

  api.get('/metadata', artifactRoute(() => 'metadata'));
  api.get('/overview', artifactRoute(() => 'overview'));
  api.get('/dispatch', artifactRoute(() => 'dispatch'));
  api.get('/forecast/:target', artifactRoute((req) => resolveTarget('forecast', req.params.target)));
  api.get('/backtest/:target', artifactRoute((req) => resolveTarget('backtest', req.params.target)));

  api.get('/series', async (req, res, next) => {
    try {
      const columns = req.query.columns
        ? String(req.query.columns)
            .split(',')
            .map((c) => c.trim())
            .filter(Boolean)
        : undefined;

      res.json(
        await getSeries({
          from: req.query.from,
          to: req.query.to,
          columns,
          resolution: req.query.resolution ? String(req.query.resolution) : 'hour',
        }),
      );
    } catch (err) {
      next(err);
    }
  });

  api.use((_req, res) => {
    res.status(404).json({ error: 'No such API route' });
  });

  app.use('/api', api);
  app.use(express.static(PUBLIC_DIR, { extensions: ['html'] }));

  // Errors carry a `status` when they describe a bad request or a missing
  // artifact; anything else is a genuine bug and stays a 500.
  app.use((err, _req, res, _next) => {
    const status = err.status ?? 500;
    if (status >= 500) console.error(err);
    res.status(status).json({ error: err.message });
  });

  return app;
}

function artifactRoute(keyFor) {
  return async (req, res, next) => {
    try {
      res.json(await loadArtifact(keyFor(req)));
    } catch (err) {
      next(err);
    }
  };
}

// Only listen when run directly, so tests can import createApp() without
// binding a port.
if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) {
  createApp().listen(PORT, HOST, () => {
    console.log(`Energy dispatch dashboard running at http://${HOST}:${PORT}`);
  });
}
