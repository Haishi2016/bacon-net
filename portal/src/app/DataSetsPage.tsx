"use client";

import { useEffect, useRef, useState, type ChangeEvent } from "react";

export type DataSet = {
  id: string;
  name: string;
  source: string;
  modality: string;
  samples: number;
  features: number;
  labeled: number;
  updated: string;
  status: "ready" | "syncing" | "review";
  location?: string;
  notes?: string;
  csv?: string;
};

const SEED_DATASETS: DataSet[] = [];

// ---- Synthetic dataset templates -------------------------------------------

type SyntheticTemplate = "hello-world";

type OperatorMode = "random" | "and" | "or";

type GeneratedDataset = {
  varNames: string[];
  expression: string;
  rows: number[][];
  labels: number[];
  csv: string;
};

const MAX_VARS = 12; // 2^12 = 4096 rows (full truth table)

function varName(index: number): string {
  return index < 26 ? String.fromCharCode(65 + index) : `V${index + 1}`;
}

// Mirrors bacon.utils.generate_classic_boolean_data: a left-associative classic
// Boolean expression over `numVars` binary inputs, evaluated across the full
// truth table (2^numVars rows). See samples/hello-world/main.py.
function generateHelloWorld(numVars: number, opMode: OperatorMode): GeneratedDataset {
  const n = Math.max(2, Math.min(MAX_VARS, Math.floor(numVars)));
  const varNames = Array.from({ length: n }, (_, i) => varName(i));
  const ops = Array.from({ length: n - 1 }, () =>
    opMode === "and" ? "and" : opMode === "or" ? "or" : Math.random() < 0.5 ? "and" : "or"
  );

  let expression = varNames[0];
  for (let i = 1; i < n; i += 1) {
    expression = `(${expression} ${ops[i - 1]} ${varNames[i]})`;
  }

  const total = 2 ** n;
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let r = 0; r < total; r += 1) {
    const bits: number[] = [];
    for (let v = 0; v < n; v += 1) {
      bits.push((r >> (n - 1 - v)) & 1);
    }
    let result = bits[0] === 1;
    for (let i = 1; i < n; i += 1) {
      result = ops[i - 1] === "and" ? result && bits[i] === 1 : result || bits[i] === 1;
    }
    rows.push(bits);
    labels.push(result ? 1 : 0);
  }

  const header = [...varNames, "label"].join(",");
  const csv = [header, ...rows.map((row, idx) => [...row, labels[idx]].join(","))].join("\n");

  return { varNames, expression, rows, labels, csv };
}

function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename.toLowerCase().endsWith(".csv") ? filename : `${filename}.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

// Lightweight CSV header/row parse; treats a trailing column named
// label/target/y/class/output as the label when present.
function summarizeCsv(text: string): { rows: number; features: number; labeled: number } {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length === 0) {
    return { rows: 0, features: 0, labeled: 0 };
  }
  const header = lines[0].split(",").map((cell) => cell.trim().toLowerCase());
  const labelNames = new Set(["label", "target", "y", "class", "output"]);
  const hasLabel = labelNames.has(header[header.length - 1]);
  const dataRows = lines.length - 1;
  const features = hasLabel ? header.length - 1 : header.length;
  return { rows: dataRows, features, labeled: hasLabel ? dataRows : 0 };
}

function statusClass(status: DataSet["status"]): string {
  return status === "ready" ? "production" : status === "syncing" ? "staging" : "draft";
}

function isUrl(value: string | undefined): value is string {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

// YAML dates parse to ISO timestamps (e.g. "2026-03-10T00:00:00.000Z"); show a
// clean calendar date instead. Falls back to the raw value if unparseable.
function formatDate(value: string | undefined): string {
  if (!value) {
    return "—";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

export default function DataSetsPage({ onOpenDataset }: { onOpenDataset: (dataset: DataSet) => void }) {
  const [datasets, setDatasets] = useState<DataSet[]>(SEED_DATASETS);
  const [catalogState, setCatalogState] = useState<"loading" | "ready" | "error">("loading");
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [showGenerator, setShowGenerator] = useState(false);
  const [template, setTemplate] = useState<SyntheticTemplate>("hello-world");
  const [numVars, setNumVars] = useState(3);
  const [opMode, setOpMode] = useState<OperatorMode>("random");
  const [datasetName, setDatasetName] = useState("");
  const [preview, setPreview] = useState<GeneratedDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingRemove, setPendingRemove] = useState<DataSet | null>(null);
  const [alsoDeleteData, setAlsoDeleteData] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [removeError, setRemoveError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // The dataset table is driven by data/datasets.yaml, served as JSON by the
  // catalog route handler. CSV uploads and synthetic generations are added on
  // top of the catalog for the current session.
  useEffect(() => {
    let active = true;
    fetch("/catalog/datasets")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((payload: { datasets?: DataSet[]; error?: string }) => {
        if (!active) {
          return;
        }
        setDatasets(Array.isArray(payload.datasets) ? payload.datasets : []);
        setCatalogState("ready");
        setCatalogError(payload.error ?? null);
      })
      .catch((err: unknown) => {
        if (!active) {
          return;
        }
        setCatalogState("error");
        setCatalogError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, []);

  // Persist a new dataset (uploaded or generated) to datasets.yaml: the server
  // writes its CSV to a data file and returns the catalog entry, which we add to
  // the list. Because it lives in the catalog, it survives navigation/reloads.
  const createDataset = async (payload: {
    name: string;
    source: string;
    modality: string;
    samples: number;
    features: number;
    labeled: number;
    csv: string;
  }) => {
    const res = await fetch("/catalog/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = (await res.json().catch(() => ({}))) as { dataset?: DataSet; error?: string };
    if (!res.ok || !data.dataset) {
      throw new Error(data.error ?? `HTTP ${res.status}`);
    }
    setDatasets((current) => [data.dataset as DataSet, ...current]);
  };

  const handleCsv = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = async () => {
      const text = String(reader.result ?? "");
      const { rows, features, labeled } = summarizeCsv(text);
      if (rows === 0) {
        setError(`"${file.name}" appears to be empty.`);
        return;
      }
      setError(null);
      try {
        await createDataset({
          name: file.name.replace(/\.csv$/i, ""),
          source: "CSV upload",
          modality: "Tabular",
          samples: rows,
          features,
          labeled,
          csv: text
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    reader.onerror = () => setError(`Could not read "${file.name}".`);
    reader.readAsText(file);
    // reset so the same file can be re-selected
    event.target.value = "";
  };

  const generate = () => {
    if (numVars < 2 || numVars > MAX_VARS) {
      setError(`Choose between 2 and ${MAX_VARS} variables.`);
      setPreview(null);
      return;
    }
    setError(null);
    setPreview(generateHelloWorld(numVars, opMode));
  };

  const addGenerated = async () => {
    if (!preview) {
      return;
    }
    const name = datasetName.trim() || `Hello World · ${preview.expression}`;
    try {
      await createDataset({
        name,
        source: "Synthetic · Boolean expression",
        modality: "Boolean",
        samples: preview.rows.length,
        features: preview.varNames.length,
        labeled: preview.rows.length,
        csv: preview.csv
      });
      setPreview(null);
      setDatasetName("");
      setShowGenerator(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const requestRemove = (ds: DataSet) => {
    setPendingRemove(ds);
    setAlsoDeleteData(false); // keep the data folder/file by default
    setRemoveError(null);
  };

  const cancelRemove = () => {
    setPendingRemove(null);
    setAlsoDeleteData(false);
    setRemoveError(null);
  };

  const confirmRemove = async () => {
    if (!pendingRemove) {
      return;
    }
    const ds = pendingRemove;
    setRemoving(true);
    setRemoveError(null);
    try {
      const res = await fetch("/catalog/datasets", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: ds.id, deleteData: alsoDeleteData })
      });
      const payload = (await res.json().catch(() => ({}))) as { error?: string };
      if (!res.ok) {
        throw new Error(payload.error ?? `HTTP ${res.status}`);
      }
      setDatasets((current) => current.filter((d) => d.id !== ds.id));
      cancelRemove();
    } catch (err) {
      setRemoveError(err instanceof Error ? err.message : String(err));
    } finally {
      setRemoving(false);
    }
  };

  const canDeleteLocalData =
    pendingRemove != null && !isUrl(pendingRemove.location) && Boolean(pendingRemove.location);

  const previewRows = preview ? preview.rows.slice(0, 8) : [];

  return (
    <div className="datasets-page">
      {pendingRemove ? (
        <div className="ds-modal-backdrop" onClick={cancelRemove}>
          <div className="ds-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <h3>Remove dataset</h3>
            <p>
              Remove <strong>{pendingRemove.name}</strong> from the catalog?
            </p>
            {canDeleteLocalData ? (
              <label className="ds-modal-check">
                <input
                  type="checkbox"
                  checked={alsoDeleteData}
                  onChange={(event) => setAlsoDeleteData(event.target.checked)}
                />
                <span>
                  Also delete the data at <code>{pendingRemove.location}</code>
                  <em> — off by default; the folder/file is kept.</em>
                </span>
              </label>
            ) : (
              <p className="ds-modal-note">
                {isUrl(pendingRemove.location)
                  ? "The source is an external link and will not be touched."
                  : "No local data file is associated with this entry."}
              </p>
            )}
            {removeError ? <p className="ds-gen-error">{removeError}</p> : null}
            <div className="ds-modal-actions">
              <button type="button" className="ghost-button" onClick={cancelRemove} disabled={removing}>
                Cancel
              </button>
              <button type="button" className="danger-button" onClick={confirmRemove} disabled={removing}>
                {removing ? "Removing…" : alsoDeleteData ? "Remove + delete data" : "Remove"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="ds-toolbar">
        <p className="ds-toolbar-info">
          {catalogState === "loading"
            ? "Loading catalog…"
            : catalogState === "error"
              ? `Catalog error: ${catalogError ?? "unknown"}`
              : `${datasets.length} dataset${datasets.length === 1 ? "" : "s"} · from datasets.yaml`}
        </p>
        <div className="ds-toolbar-actions">
          <button type="button" className="ghost-button" onClick={() => fileRef.current?.click()}>
            Load CSV…
          </button>
          <button
            type="button"
            className={showGenerator ? "primary-button" : "ghost-button"}
            onClick={() => {
              setShowGenerator((open) => !open);
              setPreview(null);
              setError(null);
            }}
          >
            Generate synthetic…
          </button>
          <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={handleCsv} />
        </div>
      </div>

      {showGenerator ? (
        <div className="ds-generate">
          <div className="ds-gen-controls">
            <label className="ds-field">
              <span>Template</span>
              <select value={template} onChange={(event) => setTemplate(event.target.value as SyntheticTemplate)}>
                <option value="hello-world">Hello World · Boolean expression</option>
              </select>
            </label>
            <label className="ds-field">
              <span>Variables</span>
              <input
                type="number"
                min={2}
                max={MAX_VARS}
                value={numVars}
                onChange={(event) => setNumVars(Number(event.target.value))}
              />
            </label>
            <label className="ds-field">
              <span>Operators</span>
              <select value={opMode} onChange={(event) => setOpMode(event.target.value as OperatorMode)}>
                <option value="random">Random AND / OR</option>
                <option value="and">All AND</option>
                <option value="or">All OR</option>
              </select>
            </label>
            <label className="ds-field ds-field-grow">
              <span>Name (optional)</span>
              <input
                type="text"
                placeholder="Auto from expression"
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
              />
            </label>
            <button type="button" className="primary-button ds-gen-run" onClick={generate}>
              Generate
            </button>
          </div>

          {error ? <p className="ds-gen-error">{error}</p> : null}

          {preview ? (
            <div className="ds-gen-preview">
              <div className="ds-gen-summary">
                <span>
                  Expression: <code>{preview.expression}</code>
                </span>
                <span>
                  {preview.rows.length} rows · {preview.varNames.length} features (full truth table)
                </span>
              </div>
              <table className="ds-preview-table">
                <thead>
                  <tr>
                    {preview.varNames.map((name) => (
                      <th key={name}>{name}</th>
                    ))}
                    <th>label</th>
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((row, idx) => (
                    <tr key={idx}>
                      {row.map((bit, col) => (
                        <td key={col}>{bit}</td>
                      ))}
                      <td className="ds-preview-label">{preview.labels[idx]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {preview.rows.length > previewRows.length ? (
                <p className="ds-gen-more">+{preview.rows.length - previewRows.length} more rows…</p>
              ) : null}
              <div className="ds-gen-preview-actions">
                <button type="button" className="primary-button" onClick={addGenerated}>
                  Add to datasets
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => downloadCsv(datasetName.trim() || "hello-world", preview.csv)}
                >
                  Download CSV
                </button>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="ds-table">
        <div className="ds-row ds-row-head">
          <span>Dataset</span>
          <span>Source</span>
          <span>Modality</span>
          <span className="ds-num-head">Samples</span>
          <span className="ds-num-head">Features</span>
          <span>Updated</span>
          <span>Status</span>
          <span className="ds-actions-head" aria-label="Actions" />
        </div>
        {datasets.map((ds) => (
          <div key={ds.id} className="ds-row">
            <button
              type="button"
              className="ds-name ds-name-open"
              title={ds.location ? `Open in Data Browser · ${ds.location}` : "Open in Data Browser"}
              onClick={() => onOpenDataset(ds)}
            >
              {ds.name}
            </button>
            <span className="ds-source" title={ds.location ?? undefined}>
              {ds.source}
            </span>
            <span>
              <i className="ds-modality">{ds.modality}</i>
            </span>
            <span className="ds-num">{ds.samples.toLocaleString()}</span>
            <span className="ds-num">{ds.features}</span>
            <span className="ds-updated">{formatDate(ds.updated)}</span>
            <span>
              <i className={`status-pill ${statusClass(ds.status)}`}>{ds.status}</i>
            </span>
            <span className="ds-actions">
              <button
                type="button"
                className="ds-remove"
                title="Remove dataset"
                aria-label={`Remove ${ds.name}`}
                onClick={() => requestRemove(ds)}
              >
                ×
              </button>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
