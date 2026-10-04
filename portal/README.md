# BACON Portal

Next.js portal for viewing and editing BACON interpretable graded-logic trees.
Migrated from the ScreenWise portal scaffold.

## What is included

- A modern portal shell with a persistent left navigation panel.
- A scenario/model dropdown in the title bar.
- A dashboard with tree view, prediction curve, threshold tuning, and overview tiles.
- An interactive tree editor built on `@xyflow/react`.

## Run locally

Install Node.js, then from this folder run:

```bash
npm install
npm run dev
```

The portal runs standalone with built-in demo data. Optional backends can be
wired in via `next.config.mjs` rewrites (`/api/*` and `/ai/*`); when they are
not running the portal falls back to the built-in scenario data.
