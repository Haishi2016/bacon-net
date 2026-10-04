"use client";

import { useMemo, useState } from "react";

type FhirVersion = "R4" | "R4B" | "R5";
type AuthMode = "open" | "smart" | "bearer";
type IngestMode = "inference" | "training";
type IngestMethod = "rest" | "bulk";

type ResourceInfo = {
  type: string;
  count: number;
  supported: boolean;
};

type FeatureMapping = {
  feature: string;
  resource: string;
  system: string;
  code: string;
  path: string;
};

// known LOINC / FHIR bindings for the demo feature sets; anything unmapped
// falls back to a placeholder so the user can fill it in
const KNOWN_BINDINGS: Record<string, Omit<FeatureMapping, "feature">> = {
  "Systolic BP": { resource: "Observation", system: "LOINC", code: "8480-6", path: "valueQuantity.value" },
  Cholesterol: { resource: "Observation", system: "LOINC", code: "2093-3", path: "valueQuantity.value" },
  Age: { resource: "Patient", system: "FHIRPath", code: "—", path: "birthDate → age" },
  "Smoking status": { resource: "Observation", system: "LOINC", code: "72166-2", path: "valueCodeableConcept" },
  "Radius mean": { resource: "Observation", system: "LOINC", code: "imaging", path: "component.valueQuantity" },
  "Texture mean": { resource: "Observation", system: "LOINC", code: "imaging", path: "component.valueQuantity" },
  "Concavity mean": { resource: "Observation", system: "LOINC", code: "imaging", path: "component.valueQuantity" },
  "Fractal dimension": { resource: "Observation", system: "LOINC", code: "imaging", path: "component.valueQuantity" },
  Spiculation: { resource: "ImagingStudy", system: "DICOM", code: "SR", path: "series.instance" },
  Diameter: { resource: "Observation", system: "LOINC", code: "33728-7", path: "valueQuantity.value" },
  Lobulation: { resource: "ImagingStudy", system: "DICOM", code: "SR", path: "series.instance" },
  "Growth rate": { resource: "Observation", system: "LOINC", code: "derived", path: "trend(valueQuantity)" }
};

const RESOURCE_CATALOG: ResourceInfo[] = [
  { type: "Patient", count: 12480, supported: true },
  { type: "Observation", count: 184320, supported: true },
  { type: "Condition", count: 38211, supported: true },
  { type: "DiagnosticReport", count: 14903, supported: true },
  { type: "ImagingStudy", count: 5217, supported: true },
  { type: "Procedure", count: 22158, supported: true },
  { type: "MedicationRequest", count: 41096, supported: true },
  { type: "RiskAssessment", count: 0, supported: true }
];

const LABEL_BINDINGS = [
  { label: "Confirmed malignancy", resource: "Condition", code: "SNOMED 363346000" },
  { label: "Positive DiagnosticReport", resource: "DiagnosticReport", code: "LOINC 24604-1" },
  { label: "Procedure performed", resource: "Procedure", code: "SNOMED 387713003" }
];

function defaultMappings(features: string[]): FeatureMapping[] {
  return features.map((feature) => {
    const known = KNOWN_BINDINGS[feature];
    return known
      ? { feature, ...known }
      : { feature, resource: "Observation", system: "LOINC", code: "—", path: "valueQuantity.value" };
  });
}

export default function IntegrationPlayground({ features }: { features: string[] }) {
  const [baseUrl, setBaseUrl] = useState("https://launch.smarthealthit.org/v/r4/fhir");
  const [version, setVersion] = useState<FhirVersion>("R4");
  const [auth, setAuth] = useState<AuthMode>("smart");
  const [connection, setConnection] = useState<"idle" | "testing" | "connected" | "error">("idle");
  const [serverName, setServerName] = useState<string | null>(null);

  const [mappings, setMappings] = useState<FeatureMapping[]>(() => defaultMappings(features));
  const [selectedResource, setSelectedResource] = useState<string>("Observation");

  const [ingestMode, setIngestMode] = useState<IngestMode>("training");
  const [ingestMethod, setIngestMethod] = useState<IngestMethod>("bulk");
  const [labelBinding, setLabelBinding] = useState(LABEL_BINDINGS[0].label);
  const [ingestState, setIngestState] = useState<"idle" | "running" | "done">("idle");
  const [ingestResult, setIngestResult] = useState<{ rows: number; cols: number; labeled: number } | null>(null);

  const mappedCount = useMemo(() => mappings.filter((m) => m.code && m.code !== "—").length, [mappings]);
  const allMapped = mappedCount === mappings.length;

  const testConnection = () => {
    setConnection("testing");
    setServerName(null);
    window.setTimeout(() => {
      if (!/^https?:\/\//i.test(baseUrl)) {
        setConnection("error");
        return;
      }
      setServerName("SMART Health IT Sandbox · HAPI FHIR 6.4");
      setConnection("connected");
    }, 700);
  };

  const updateMapping = (feature: string, field: keyof FeatureMapping, value: string) => {
    setMappings((prev) => prev.map((m) => (m.feature === feature ? { ...m, [field]: value } : m)));
  };

  const runIngestion = () => {
    setIngestState("running");
    setIngestResult(null);
    window.setTimeout(() => {
      const rows = ingestMethod === "bulk" ? 12480 : 240;
      setIngestResult({
        rows,
        cols: mappings.length,
        labeled: ingestMode === "training" ? Math.round(rows * 0.86) : 0
      });
      setIngestState("done");
    }, 1100);
  };

  const isConnected = connection === "connected";

  return (
    <div className="playground">
      <div className="playground-grid">
        {/* connection */}
        <section className="pg-card pg-connect">
          <div className="pg-card-head">
            <h3>FHIR server</h3>
            <span className={`pg-status ${connection}`}>
              {connection === "connected"
                ? "Connected"
                : connection === "testing"
                ? "Testing…"
                : connection === "error"
                ? "Unreachable"
                : "Not connected"}
            </span>
          </div>

          <label className="pg-field">
            <span>Base URL</span>
            <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://example.org/fhir" />
          </label>

          <div className="pg-field-row">
            <label className="pg-field">
              <span>Version</span>
              <select value={version} onChange={(e) => setVersion(e.target.value as FhirVersion)}>
                <option value="R4">R4 (4.0.1)</option>
                <option value="R4B">R4B</option>
                <option value="R5">R5 (5.0.0)</option>
              </select>
            </label>
            <label className="pg-field">
              <span>Auth</span>
              <select value={auth} onChange={(e) => setAuth(e.target.value as AuthMode)}>
                <option value="open">Open / no auth</option>
                <option value="smart">SMART on FHIR (OAuth2)</option>
                <option value="bearer">Bearer token</option>
              </select>
            </label>
          </div>

          <button type="button" className="primary-button pg-test" onClick={testConnection}>
            Test connection
          </button>

          {serverName ? (
            <div className="pg-server-meta">
              <span className="pg-ok-dot" />
              <div>
                <strong>{serverName}</strong>
                <span>CapabilityStatement read · {RESOURCE_CATALOG.filter((r) => r.supported).length} resources supported</span>
              </div>
            </div>
          ) : null}
        </section>

        {/* resource browser */}
        <section className={`pg-card pg-browse${isConnected ? "" : " pg-disabled"}`}>
          <div className="pg-card-head">
            <h3>Resource browser</h3>
            {isConnected ? <span className="pg-hint">{version} · live counts</span> : <span className="pg-hint">connect first</span>}
          </div>
          <div className="pg-resource-list">
            {RESOURCE_CATALOG.map((res) => (
              <button
                key={res.type}
                type="button"
                className={`pg-resource${selectedResource === res.type ? " active" : ""}`}
                onClick={() => setSelectedResource(res.type)}
                disabled={!isConnected}
              >
                <span className="pg-resource-name">{res.type}</span>
                <span className="pg-resource-count">{res.count.toLocaleString()}</span>
              </button>
            ))}
          </div>
        </section>

        {/* feature mapping */}
        <section className={`pg-card pg-map${isConnected ? "" : " pg-disabled"}`}>
          <div className="pg-card-head">
            <h3>Feature mapping</h3>
            <span className={`pg-hint${allMapped ? " ok" : ""}`}>
              {mappedCount}/{mappings.length} features mapped
            </span>
          </div>
          <div className="pg-map-table">
            <div className="pg-map-row pg-map-header">
              <span>Model feature</span>
              <span>Resource</span>
              <span>System</span>
              <span>Code</span>
              <span>Path</span>
            </div>
            {mappings.map((m) => {
              const unmapped = !m.code || m.code === "—";
              return (
                <div key={m.feature} className={`pg-map-row${unmapped ? " unmapped" : ""}`}>
                  <span className="pg-feature-name">{m.feature}</span>
                  <input value={m.resource} onChange={(e) => updateMapping(m.feature, "resource", e.target.value)} />
                  <input value={m.system} onChange={(e) => updateMapping(m.feature, "system", e.target.value)} />
                  <input
                    value={m.code}
                    onChange={(e) => updateMapping(m.feature, "code", e.target.value)}
                    placeholder="LOINC code"
                  />
                  <input value={m.path} onChange={(e) => updateMapping(m.feature, "path", e.target.value)} />
                </div>
              );
            })}
          </div>
        </section>

        {/* ingestion */}
        <section className={`pg-card pg-ingest${isConnected ? "" : " pg-disabled"}`}>
          <div className="pg-card-head">
            <h3>Ingest data</h3>
            <span className="pg-hint">{allMapped ? "ready" : "map all features first"}</span>
          </div>

          <div className="pg-toggle-row">
            <div className="pg-toggle">
              <button className={ingestMode === "inference" ? "active" : ""} onClick={() => setIngestMode("inference")} type="button">
                Inference
              </button>
              <button className={ingestMode === "training" ? "active" : ""} onClick={() => setIngestMode("training")} type="button">
                Training
              </button>
            </div>
            <div className="pg-toggle">
              <button className={ingestMethod === "rest" ? "active" : ""} onClick={() => setIngestMethod("rest")} type="button">
                REST query
              </button>
              <button className={ingestMethod === "bulk" ? "active" : ""} onClick={() => setIngestMethod("bulk")} type="button">
                Bulk $export
              </button>
            </div>
          </div>

          {ingestMode === "training" ? (
            <label className="pg-field">
              <span>Outcome label binding</span>
              <select value={labelBinding} onChange={(e) => setLabelBinding(e.target.value)}>
                {LABEL_BINDINGS.map((l) => (
                  <option key={l.label} value={l.label}>
                    {l.label} · {l.resource} ({l.code})
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <p className="pg-note">
              Scores are written back as <code>RiskAssessment</code> resources to the source server.
            </p>
          )}

          <button
            type="button"
            className="primary-button pg-run"
            onClick={runIngestion}
            disabled={!isConnected || !allMapped || ingestState === "running"}
          >
            {ingestState === "running"
              ? "Ingesting…"
              : ingestMode === "training"
              ? "Build training set"
              : "Run inference batch"}
          </button>

          {ingestResult ? (
            <div className="pg-result">
              <div className="pg-result-figure">
                <strong>{ingestResult.rows.toLocaleString()}</strong>
                <span>patients</span>
              </div>
              <div className="pg-result-figure">
                <strong>{ingestResult.cols}</strong>
                <span>features</span>
              </div>
              {ingestMode === "training" ? (
                <div className="pg-result-figure">
                  <strong>{ingestResult.labeled.toLocaleString()}</strong>
                  <span>labeled</span>
                </div>
              ) : (
                <div className="pg-result-figure">
                  <strong>{ingestResult.rows.toLocaleString()}</strong>
                  <span>scored</span>
                </div>
              )}
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}
