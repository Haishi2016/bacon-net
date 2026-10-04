"use client";

import { useEffect, useState } from "react";
import type { DataSet } from "./DataSetsPage";

type TableData = {
  header: string[];
  rows: string[][];
  total: number;
  truncated: boolean;
};

const MAX_ROWS = 500;

function isUrl(value: string | undefined): value is string {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

// Minimal CSV parse (comma-separated, no quoted-field handling — sufficient for
// the simple tabular/boolean datasets used here).
function parseCsv(text: string): TableData {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length === 0) {
    return { header: [], rows: [], total: 0, truncated: false };
  }
  const header = lines[0].split(",").map((cell) => cell.trim());
  const dataLines = lines.slice(1);
  const rows = dataLines.slice(0, MAX_ROWS).map((line) => line.split(",").map((cell) => cell.trim()));
  return { header, rows, total: dataLines.length, truncated: dataLines.length > rows.length };
}

type BrowserState = "loading" | "table" | "external" | "unavailable" | "error";

export default function DataBrowserPage({ dataset }: { dataset: DataSet | null }) {
  const [state, setState] = useState<BrowserState>("loading");
  const [table, setTable] = useState<TableData | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset) {
      return;
    }
    let active = true;
    setTable(null);
    setMessage(null);

    // 1) Inline CSV (uploaded or generated in this session) — render directly.
    if (dataset.csv) {
      setTable(parseCsv(dataset.csv));
      setState("table");
      return;
    }

    // 2) External link — we don't proxy remote files; offer the link.
    if (isUrl(dataset.location)) {
      setState("external");
      return;
    }

    // 3) Local catalog entry — ask the server to read a CSV at/under location.
    setState("loading");
    fetch(`/catalog/preview?id=${encodeURIComponent(dataset.id)}`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((payload: { kind?: string; header?: string[]; rows?: string[][]; total?: number; truncated?: boolean; url?: string; message?: string }) => {
        if (!active) {
          return;
        }
        if (payload.kind === "table" && Array.isArray(payload.header)) {
          setTable({
            header: payload.header,
            rows: Array.isArray(payload.rows) ? payload.rows : [],
            total: payload.total ?? 0,
            truncated: Boolean(payload.truncated)
          });
          setState("table");
        } else if (payload.kind === "url" && payload.url) {
          setState("external");
        } else {
          setMessage(payload.message ?? "No tabular data file is available for this dataset.");
          setState("unavailable");
        }
      })
      .catch((err: unknown) => {
        if (!active) {
          return;
        }
        setMessage(err instanceof Error ? err.message : String(err));
        setState("error");
      });

    return () => {
      active = false;
    };
  }, [dataset]);

  if (!dataset) {
    return (
      <div className="browser-empty">
        <p>Select a dataset from the Datasets page to browse it here.</p>
      </div>
    );
  }

  return (
    <div className="data-browser">
      <div className="browser-meta">
        <span className="browser-chip">{dataset.modality}</span>
        <span>{dataset.samples.toLocaleString()} samples</span>
        <span>{dataset.features} features</span>
        <span>{dataset.source}</span>
        {dataset.location ? (
          isUrl(dataset.location) ? (
            <a href={dataset.location} target="_blank" rel="noopener noreferrer" className="browser-loc">
              {dataset.location}
            </a>
          ) : (
            <span className="browser-loc">{dataset.location}</span>
          )
        ) : null}
      </div>

      {state === "loading" ? <p className="browser-note">Loading data…</p> : null}

      {state === "external" ? (
        <div className="browser-note">
          This dataset is an external source.{" "}
          {isUrl(dataset.location) ? (
            <a href={dataset.location} target="_blank" rel="noopener noreferrer">
              Open it in a new tab
            </a>
          ) : null}
          .
        </div>
      ) : null}

      {state === "unavailable" ? <p className="browser-note">{message}</p> : null}

      {state === "error" ? <p className="browser-note browser-error">Could not load data: {message}</p> : null}

      {state === "table" && table ? (
        <>
          <p className="browser-note">
            Showing {table.rows.length.toLocaleString()} of {table.total.toLocaleString()} rows
            {table.truncated ? " (truncated)" : ""}.
          </p>
          <div className="browser-table-wrap">
            <table className="browser-table">
              <thead>
                <tr>
                  <th className="browser-rownum">#</th>
                  {table.header.map((col, idx) => (
                    <th key={`${col}-${idx}`}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    <td className="browser-rownum">{rowIdx + 1}</td>
                    {table.header.map((_, colIdx) => (
                      <td key={colIdx}>{row[colIdx] ?? ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
}
