"use client";

import { useEffect, useState } from "react";

export type SavedModel = {
  id: string;
  name: string;
  dataset?: string;
  datasetName?: string;
  aggregator?: string;
  accuracy?: number | null;
  features?: string[];
  nodes?: number;
  links?: number;
  status: "draft" | "staging" | "production";
  updated: string;
  location?: string;
};

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function formatDate(value: string | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

export default function ModelsPage({ onEdit }: { onEdit?: (model: SavedModel) => void }) {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setState("loading");
    fetch("/catalog/models")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((payload: { models?: SavedModel[]; error?: string }) => {
        setModels(Array.isArray(payload.models) ? payload.models : []);
        setState("ready");
        setError(payload.error ?? null);
      })
      .catch((err: unknown) => {
        setState("error");
        setError(err instanceof Error ? err.message : String(err));
      });
  };

  useEffect(load, []);

  const remove = async (model: SavedModel) => {
    try {
      const res = await fetch("/catalog/models", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: model.id })
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(data.error ?? `HTTP ${res.status}`);
      }
      setModels((current) => current.filter((m) => m.id !== model.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (state === "loading") {
    return <div className="models-empty">Loading models…</div>;
  }
  if (state === "error") {
    return <div className="models-empty models-error">Could not load models: {error}</div>;
  }
  if (models.length === 0) {
    return (
      <div className="models-empty">
        No saved models yet. Open the <strong>Model Editor</strong>, train a model, and click <strong>Save</strong>.
      </div>
    );
  }

  return (
    <div className="models-page">
      <div className="model-table">
        <div className="model-row model-row-head">
          <span>Model</span>
          <span>Dataset</span>
          <span>Aggregator</span>
          <span>Accuracy</span>
          <span>Nodes</span>
          <span>Status</span>
          <span>Updated</span>
          <span />
        </div>
        {models.map((model) => (
          <div key={model.id} className="model-row">
            <span className="model-version">
              <strong>{model.name}</strong>
            </span>
            <span className="model-sub">{model.datasetName ?? model.dataset ?? "—"}</span>
            <span className="model-sub">{model.aggregator ?? "—"}</span>
            <span className="model-metric">{typeof model.accuracy === "number" ? pct(model.accuracy) : "—"}</span>
            <span className="model-metric">{model.nodes ?? "—"}</span>
            <span>
              <i className={`status-pill ${model.status}`}>{model.status}</i>
            </span>
            <span className="model-updated">{formatDate(model.updated)}</span>
            <span className="model-actions">
              <button type="button" className="link-button" onClick={() => onEdit?.(model)}>
                Edit
              </button>
              <button type="button" className="link-button link-danger" onClick={() => remove(model)}>
                Delete
              </button>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
