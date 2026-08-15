/**
 * Filesystem locations shared by the server modules.
 *
 * The dashboard reads from the repository's data/ directory rather than
 * keeping its own copy, so regenerating the pipeline is immediately visible
 * to the frontend with nothing to sync.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const FRONTEND_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
export const REPO_ROOT = path.dirname(FRONTEND_ROOT);
export const DATA_ROOT = path.join(REPO_ROOT, 'data');
export const DASHBOARD_DIR = path.join(DATA_ROOT, 'dashboard');
export const PUBLIC_DIR = path.join(FRONTEND_ROOT, 'public');
