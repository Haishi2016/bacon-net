"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DiagnosisTree, { type DiagnosisTreeNode } from "./DiagnosisTree";
import PredictionCurve from "./PredictionCurve";
import TreeEditor from "./TreeEditor";
import ModelsPage from "./ModelsPage";
import DataSetsPage, { type DataSet } from "./DataSetsPage";
import DataBrowserPage from "./DataBrowserPage";
import DistillationPage from "./DistillationPage";

function formatMoney(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }
  return `$${value.toLocaleString()}`;
}

type Scenario = {
  id: string;
  name: string;
  category: string;
  dataset: string;
  summary: string;
  features: string[];
  testPackages: Array<{
    name: string;
    selected: boolean;
    cost: number;
    features: Array<{ name: string; used: boolean }>;
  }>;
  metrics: {
    accuracy: number;
    auc: number;
    recall: number;
    precision: number;
    f1: number;
    costReduction: number;
    totalSavings: number;
    savingsPerPatient: number;
  };
  thresholdPoints: Array<{
    threshold: number;
    recall: number;
    precision: number;
    specificity: number;
  }>;
  tree: Array<{
    label: string;
    count: number;
    children?: Array<{
      label: string;
      count: number;
      children?: Array<{
        label: string;
        count: number;
      }>;
    }>;
  }>;
};

type ScenarioOption = {
  id: string;
  name: string;
  category: string;
  dataset: string;
  description?: string;
  url?: string;
};

const scenarios: Scenario[] = [
  {
    id: "breast-cancer",
    name: "Breast Cancer",
    category: "Diagnosis",
    dataset: "Wisconsin Breast Cancer Diagnostic",
    summary: "Risk stratification for malignant vs benign lesions with interpretable screening signals.",
    features: ["Radius mean", "Texture mean", "Concavity mean", "Fractal dimension"],
    testPackages: [
      {
        name: "Morphometry Panel",
        selected: true,
        cost: 145,
        features: [
          { name: "Radius mean", used: true },
          { name: "Texture mean", used: true },
          { name: "Perimeter mean", used: false },
          { name: "Area mean", used: false }
        ]
      },
      {
        name: "Contour Panel",
        selected: true,
        cost: 210,
        features: [
          { name: "Concavity mean", used: true },
          { name: "Fractal dimension", used: true },
          { name: "Symmetry mean", used: false }
        ]
      },
      {
        name: "Density Panel",
        selected: false,
        cost: 320,
        features: [
          { name: "Smoothness mean", used: false },
          { name: "Compactness mean", used: false }
        ]
      },
      {
        name: "Genomic Panel",
        selected: false,
        cost: 890,
        features: [
          { name: "BRCA score", used: false },
          { name: "Ki-67 index", used: false }
        ]
      }
    ],
    metrics: {
      accuracy: 0.958,
      auc: 0.982,
      recall: 0.941,
      precision: 0.924,
      f1: 0.932,
      costReduction: 68,
      totalSavings: 1220000,
      savingsPerPatient: 978
    },
    thresholdPoints: [
      { threshold: 0.18, recall: 0.98, precision: 0.77, specificity: 0.61 },
      { threshold: 0.32, recall: 0.95, precision: 0.86, specificity: 0.74 },
      { threshold: 0.5, recall: 0.91, precision: 0.92, specificity: 0.84 },
      { threshold: 0.68, recall: 0.83, precision: 0.95, specificity: 0.91 },
      { threshold: 0.84, recall: 0.71, precision: 0.97, specificity: 0.95 }
    ],
    tree: [
      {
        label: "All patients",
        count: 1248,
        children: [
          {
            label: "High texture variance",
            count: 412,
            children: [
              { label: "Irregular shape", count: 276 },
              { label: "Regular shape", count: 136 }
            ]
          },
          {
            label: "Low texture variance",
            count: 836,
            children: [
              { label: "Dense margin", count: 214 },
              { label: "Smooth margin", count: 622 }
            ]
          }
        ]
      }
    ]
  },
  {
    id: "lung-nodule",
    name: "Lung Nodule",
    category: "Diagnosis",
    dataset: "LIDC-IDRI Nodule Cohort",
    summary: "Imaging-derived malignancy support for pulmonary nodule review.",
    features: ["Spiculation", "Diameter", "Lobulation", "Growth rate"],
    testPackages: [
      {
        name: "Imaging Panel",
        selected: true,
        cost: 380,
        features: [
          { name: "Spiculation", used: true },
          { name: "Diameter", used: true },
          { name: "Lobulation", used: true },
          { name: "Margin sharpness", used: false }
        ]
      },
      {
        name: "Temporal Panel",
        selected: true,
        cost: 260,
        features: [
          { name: "Growth rate", used: true },
          { name: "Doubling time", used: false }
        ]
      },
      {
        name: "Texture Panel",
        selected: false,
        cost: 175,
        features: [
          { name: "Calcification", used: false },
          { name: "Cavitation", used: false }
        ]
      },
      {
        name: "Serum Panel",
        selected: false,
        cost: 540,
        features: [
          { name: "CEA", used: false },
          { name: "CYFRA 21-1", used: false }
        ]
      }
    ],
    metrics: {
      accuracy: 0.917,
      auc: 0.961,
      recall: 0.889,
      precision: 0.903,
      f1: 0.896,
      costReduction: 54,
      totalSavings: 1840000,
      savingsPerPatient: 642
    },
    thresholdPoints: [
      { threshold: 0.2, recall: 0.96, precision: 0.71, specificity: 0.56 },
      { threshold: 0.38, recall: 0.92, precision: 0.82, specificity: 0.71 },
      { threshold: 0.55, recall: 0.88, precision: 0.9, specificity: 0.82 },
      { threshold: 0.72, recall: 0.79, precision: 0.94, specificity: 0.9 },
      { threshold: 0.88, recall: 0.66, precision: 0.96, specificity: 0.96 }
    ],
    tree: [
      {
        label: "All scans",
        count: 980,
        children: [
          {
            label: "Suspicious shape",
            count: 326,
            children: [
              { label: "Growth-confirmed", count: 188 },
              { label: "Stable", count: 138 }
            ]
          },
          {
            label: "Non-suspicious shape",
            count: 654,
            children: [
              { label: "Follow-up", count: 192 },
              { label: "Routine", count: 462 }
            ]
          }
        ]
      }
    ]
  },
  {
    id: "cardio-risk",
    name: "Cardio Risk",
    category: "Risk Scoring",
    dataset: "Framingham Heart Study",
    summary: "Preventive screening score for near-term cardiovascular intervention.",
    features: ["Systolic BP", "Cholesterol", "Age", "Smoking status"],
    testPackages: [
      {
        name: "Vitals Panel",
        selected: true,
        cost: 60,
        features: [
          { name: "Systolic BP", used: true },
          { name: "Diastolic BP", used: false },
          { name: "Heart rate", used: false }
        ]
      },
      {
        name: "Lipid Panel",
        selected: true,
        cost: 95,
        features: [
          { name: "Cholesterol", used: true },
          { name: "HDL", used: false },
          { name: "LDL", used: false }
        ]
      },
      {
        name: "Demographics",
        selected: true,
        cost: 15,
        features: [
          { name: "Age", used: true },
          { name: "Smoking status", used: true },
          { name: "Sex", used: false }
        ]
      },
      {
        name: "Cardiac Markers",
        selected: false,
        cost: 430,
        features: [
          { name: "Troponin", used: false },
          { name: "BNP", used: false }
        ]
      }
    ],
    metrics: {
      accuracy: 0.904,
      auc: 0.947,
      recall: 0.872,
      precision: 0.881,
      f1: 0.876,
      costReduction: 61,
      totalSavings: 1460000,
      savingsPerPatient: 815
    },
    thresholdPoints: [
      { threshold: 0.16, recall: 0.97, precision: 0.73, specificity: 0.58 },
      { threshold: 0.34, recall: 0.9, precision: 0.84, specificity: 0.74 },
      { threshold: 0.5, recall: 0.86, precision: 0.88, specificity: 0.84 },
      { threshold: 0.67, recall: 0.77, precision: 0.93, specificity: 0.9 },
      { threshold: 0.82, recall: 0.62, precision: 0.96, specificity: 0.95 }
    ],
    tree: [
      {
        label: "Population",
        count: 2104,
        children: [
          {
            label: "Elevated BP",
            count: 684,
            children: [
              { label: "Smoker", count: 248 },
              { label: "Non-smoker", count: 436 }
            ]
          },
          {
            label: "Normal BP",
            count: 1420,
            children: [
              { label: "High LDL", count: 318 },
              { label: "Stable profile", count: 1102 }
            ]
          }
        ]
      }
    ]
  }
];

const views = ["Dashboard", "Models", "DataSets", "Model Editor", "Data Browser", "Distillation"] as const;

type ViewName = (typeof views)[number];

// Aggregator families (bacon.baconNet._aggregator_registry). The first is the
// hello-world default.
const AGGREGATOR_FAMILIES: Array<{ value: string; label: string }> = [
  { value: "bool.min_max", label: "Boolean · min/max (AND / OR)" },
  { value: "lsp.full_weight", label: "LSP · full weight (graded)" },
  { value: "lsp.half_weight", label: "LSP · half weight (graded)" },
  { value: "lsp.softmax", label: "LSP · softmax (graded)" },
  { value: "gl.generic", label: "Graded Logic · generic" },
  { value: "math.operator_set.logic", label: "Operator set · logic (AND / OR)" },
  { value: "math.operator_set.logic_identity", label: "Operator set · logic + identity" },
  { value: "math.operator_set.arith", label: "Operator set · arithmetic" }
];

// Auto-scrolling epoch/log console for the training modal.
function TrainLog({ lines, running }: { lines: string[]; running: boolean }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [lines]);
  return (
    <div className="train-log">
      {lines.length === 0 && running ? <div className="train-log-line dim">Starting training…</div> : null}
      {lines.map((line, idx) => (
        <div key={idx} className="train-log-line">
          {line}
        </div>
      ))}
      {running ? <div className="train-log-cursor" /> : null}
      <div ref={endRef} />
    </div>
  );
}

export default function Page() {
  const [activeView, setActiveView] = useState<ViewName>("Dashboard");
  const [browserDataset, setBrowserDataset] = useState<DataSet | null>(null);
  const [selectedScenarioId, setSelectedScenarioId] = useState(scenarios[0].id);
  const [threshold, setThreshold] = useState(50);
  const [modelName, setModelName] = useState("");
  const [editorDatasetId, setEditorDatasetId] = useState("");
  const [editorFeatures, setEditorFeatures] = useState<string[]>([]);
  const [datasetRefs, setDatasetRefs] = useState<DataSet[]>([]);
  const [learnedTree, setLearnedTree] = useState<DiagnosisTreeNode[] | null>(null);
  const [treeKey, setTreeKey] = useState(0);
  const [training, setTraining] = useState(false);
  const [trainSettingsOpen, setTrainSettingsOpen] = useState(false);
  const [trainAggregator, setTrainAggregator] = useState("bool.min_max");
  const [trainedAggregator, setTrainedAggregator] = useState<string | null>(null);
  const [trainLogs, setTrainLogs] = useState<string[]>([]);
  const [trainStatus, setTrainStatus] = useState<"running" | "done" | "error">("running");
  const [trainSummary, setTrainSummary] = useState<{ accuracy?: number; bestAccuracy?: number } | null>(null);
  const [trainModelStaging, setTrainModelStaging] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const completedRef = useRef(false);
  const [scenarioOptions, setScenarioOptions] = useState<ScenarioOption[]>([]);
  const [, setScenarioError] = useState<string | null>(null);
  const [prunedTree, setPrunedTree] = useState<Scenario["tree"] | null>(null);

  // Dataset references for the Model Editor dropdown, from the catalog (datasets.yaml).
  useEffect(() => {
    let active = true;
    fetch("/catalog/datasets")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((payload: { datasets?: DataSet[] }) => {
        if (active && Array.isArray(payload.datasets)) {
          setDatasetRefs(payload.datasets);
        }
      })
      .catch(() => {
        /* dropdown stays empty if the catalog can't be read */
      });
    return () => {
      active = false;
    };
  }, []);

  // Resolve the feature names for a selected dataset from its CSV header (via the
  // preview endpoint), dropping the label column. Datasets without a readable CSV
  // yield an empty feature list.
  const selectEditorDataset = async (id: string) => {
    setEditorDatasetId(id);
    if (!id) {
      setEditorFeatures([]);
      return;
    }
    try {
      const res = await fetch(`/catalog/preview?id=${encodeURIComponent(id)}`);
      const data = (await res.json()) as { kind?: string; header?: string[] };
      if (data.kind === "table" && Array.isArray(data.header)) {
        const labelNames = new Set(["label", "target", "y", "class", "output"]);
        setEditorFeatures(data.header.filter((h) => !labelNames.has(String(h).toLowerCase())));
      } else {
        setEditorFeatures([]);
      }
    } catch {
      setEditorFeatures([]);
    }
  };

  // Train a real BACON model on the selected dataset via the SSE train route,
  // streaming the library's epoch logs into a modal. On completion the learned
  // tree is loaded into the editor.
  const applyLearnedTree = (tree: DiagnosisTreeNode[]) => {
    setLearnedTree(tree);
    setTreeKey((k) => k + 1);
  };

  // Save the current editor model to models.yaml. The trained bacon checkpoint
  // (.pth) staged during training is finalized server-side; the tree is stored
  // for display/reload.
  const saveModel = async (tree: DiagnosisTreeNode[], stats: { nodes: number; links: number }) => {
    const name = modelName.trim();
    if (!name) {
      throw new Error("Enter a model name first.");
    }
    const ref = datasetRefs.find((d) => d.id === editorDatasetId);
    const res = await fetch("/catalog/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        dataset: editorDatasetId || undefined,
        datasetName: ref?.name,
        aggregator: trainedAggregator ?? undefined,
        accuracy: trainSummary?.accuracy ?? null,
        features: editorFeatures,
        nodes: stats.nodes,
        links: stats.links,
        tree,
        modelStaging: trainModelStaging ?? undefined
      })
    });
    const data = (await res.json().catch(() => ({}))) as { error?: string };
    if (!res.ok) {
      throw new Error(data.error ?? `HTTP ${res.status}`);
    }
  };

  // Open a saved model from the Models page in the editor. The tree is read
  // from the model's single .pth checkpoint (embedded metadata).
  const openModelInEditor = async (model: {
    id: string;
    name: string;
    dataset?: string;
    aggregator?: string;
    features?: string[];
  }) => {
    setActiveView("Model Editor");
    setModelName(model.name);
    setTrainedAggregator(model.aggregator ?? null);
    setEditorDatasetId(model.dataset ?? "");
    setEditorFeatures(model.features ?? []);
    setLearnedTree(null);
    setTreeKey((k) => k + 1);
    try {
      const res = await fetch(`/catalog/models/detail?id=${encodeURIComponent(model.id)}`);
      const data = (await res.json()) as {
        metadata?: { tree?: DiagnosisTreeNode[]; feature_names?: string[]; aggregator?: string };
      };
      const meta = data.metadata;
      if (meta?.tree) {
        if (Array.isArray(meta.feature_names)) setEditorFeatures(meta.feature_names);
        if (meta.aggregator) setTrainedAggregator(meta.aggregator);
        setLearnedTree(meta.tree);
        setTreeKey((k) => k + 1);
      }
    } catch {
      /* leave editor empty if the checkpoint can't be read */
    }
  };

  const closeTraining = () => {
    esRef.current?.close();
    esRef.current = null;
    setTraining(false);
  };

  const startTraining = () => {
    if (!editorDatasetId) {
      return;
    }
    esRef.current?.close();
    completedRef.current = false;
    setTrainSettingsOpen(false);
    setTrainLogs([]);
    setTrainSummary(null);
    setTrainModelStaging(null);
    setTrainStatus("running");
    setTraining(true);

    const aggregator = trainAggregator;
    const es = new EventSource(
      `/catalog/train?id=${encodeURIComponent(editorDatasetId)}&aggregator=${encodeURIComponent(aggregator)}`
    );
    esRef.current = es;

    es.addEventListener("log", (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as { message: string };
        setTrainLogs((logs) => [...logs, data.message]);
      } catch {
        /* ignore malformed line */
      }
    });

    es.addEventListener("tree", (event) => {
      try {
        const tree = JSON.parse((event as MessageEvent).data) as DiagnosisTreeNode[];
        applyLearnedTree(tree);
        setTrainedAggregator(aggregator);
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("model", (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as { staging: string };
        setTrainModelStaging(data.staging);
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("done", (event) => {
      try {
        setTrainSummary(JSON.parse((event as MessageEvent).data));
      } catch {
        /* ignore */
      }
    });

    es.addEventListener("error", (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as { message: string };
        setTrainLogs((logs) => [...logs, `⚠️ ${data.message}`]);
        setTrainStatus("error");
        completedRef.current = true;
      } catch {
        /* native transport error handled by onerror */
      }
    });

    es.addEventListener("close", () => {
      completedRef.current = true;
      es.close();
      esRef.current = null;
      setTrainStatus((status) => (status === "error" ? status : "done"));
    });

    es.onerror = () => {
      if (completedRef.current) {
        return;
      }
      completedRef.current = true;
      es.close();
      esRef.current = null;
      setTrainStatus("error");
      setTrainLogs((logs) => [...logs, "⚠️ Lost connection to the training service."]);
    };
  };

  // Populate the scenario dropdown from the backend, which serves the catalog
  // from scenarios.json through the state capability. The dropdown is driven
  // entirely by this data — there is no hard-coded fallback.
  useEffect(() => {
    let active = true;

    const slugify = (value: string) =>
      value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");

    // The HTTP binding wraps the agent's Fulfillment, so the body arrives as
    // { text | content: "<json array>" }. Unwrap, then accept either camelCase
    // or PascalCase fields.
    const normalize = (raw: unknown): ScenarioOption[] => {
      let value: unknown = raw;

      if (value && typeof value === "object" && !Array.isArray(value)) {
        const wrapper = value as Record<string, unknown>;
        const payload = (wrapper.text ?? wrapper.Text ?? wrapper.content ?? wrapper.Content) as unknown;
        if (typeof payload === "string") {
          try {
            value = JSON.parse(payload);
          } catch {
            value = [];
          }
        }
      }

      if (!Array.isArray(value)) {
        return [];
      }

      return value
        .flatMap((item): ScenarioOption[] => {
          const entry = item as Record<string, unknown>;
          const name = (entry.name ?? entry.Name ?? entry.title ?? entry.Title) as string | undefined;
          if (!name) {
            return [];
          }
          const id = (entry.id ?? entry.Id) as string | undefined;
          return [
            {
              id: id ?? slugify(name),
              name,
              category: ((entry.category ?? entry.Category) as string | undefined) ?? "Uncategorized",
              dataset: ((entry.dataset ?? entry.Dataset) as string | undefined) ?? "",
              description: (entry.description ?? entry.Description) as string | undefined,
              url: (entry.url ?? entry.Url) as string | undefined
            } satisfies ScenarioOption
          ];
        });
    };

    fetch("/api/scenarios", { method: "POST" })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((raw) => {
        if (!active) {
          return;
        }
        const options = normalize(raw);
        setScenarioOptions(options);
        setScenarioError(options.length === 0 ? "The scenarios API returned no scenarios." : null);
        if (options.length > 0) {
          setSelectedScenarioId((current) =>
            options.some((option) => option.id === current) ? current : options[0].id
          );
        }
      })
      .catch((err: unknown) => {
        if (!active) {
          return;
        }
        setScenarioOptions([]);
        setScenarioError(
          `Could not reach the scenarios API on http://localhost:5080. Is the backend running? (${
            err instanceof Error ? err.message : String(err)
          })`
        );
      });

    return () => {
      active = false;
    };
  }, []);

  // Fetch the critical (pruned) decision tree for the selected scenario from
  // the AI service (ai/tree, proxied to http://localhost:5090). The pruning
  // analysis exported this tree from BACON; we visualize it when available and
  // fall back to the built-in scenario tree otherwise.
  useEffect(() => {
    let active = true;
    setPrunedTree(null);

    const unwrap = (raw: unknown): unknown => {
      let value: unknown = raw;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        const wrapper = value as Record<string, unknown>;
        const payload = (wrapper.text ?? wrapper.Text ?? wrapper.content ?? wrapper.Content) as unknown;
        if (typeof payload === "string") {
          try {
            value = JSON.parse(payload);
          } catch {
            return null;
          }
        }
      }
      return value;
    };

    fetch("/ai/tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedScenarioId })
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((raw) => {
        if (!active) {
          return;
        }
        const value = unwrap(raw);
        const tree =
          value && typeof value === "object" && !Array.isArray(value)
            ? (value as Record<string, unknown>).tree
            : undefined;
        setPrunedTree(Array.isArray(tree) && tree.length > 0 ? (tree as Scenario["tree"]) : null);
      })
      .catch(() => {
        if (active) {
          setPrunedTree(null);
        }
      });

    return () => {
      active = false;
    };
  }, [selectedScenarioId]);

  const selectedScenario = useMemo(
    () => scenarios.find((scenario) => scenario.id === selectedScenarioId) ?? scenarios[0],
    [selectedScenarioId]
  );

  // The catalog entry driving the top-bar labels. Falls back to the
  // built-in scenario data before the catalog has loaded.
  const score = selectedScenario.thresholdPoints.reduce((closest, point) =>
    Math.abs(point.threshold * 100 - threshold) < Math.abs(closest.threshold * 100 - threshold) ? point : closest
  );

  const optimalThreshold = useMemo(() => {
    const f1 = (p: { recall: number; precision: number }) =>
      (2 * p.recall * p.precision) / (p.recall + p.precision);
    const best = selectedScenario.thresholdPoints.reduce((top, point) =>
      f1(point) > f1(top) ? point : top
    );
    return Math.round(best.threshold * 100);
  }, [selectedScenario]);

  const costSummary = useMemo(() => {
    const packages = selectedScenario.testPackages;
    const totalPackages = packages.length;
    const selectedPackages = packages.filter((pkg) => pkg.selected).length;
    const allFeatures = packages.flatMap((pkg) => pkg.features);
    const totalFeatures = allFeatures.length;
    const usedFeatures = allFeatures.filter((feature) => feature.used).length;
    return { totalPackages, selectedPackages, totalFeatures, usedFeatures };
  }, [selectedScenario]);

  const clinicalStages = useMemo(() => {
    const points = selectedScenario.thresholdPoints;
    const pick = (index: number) => points[Math.min(index, points.length - 1)];
    return [
      { name: "Screening", point: pick(0) },
      { name: "Diagnosis", point: pick(Math.floor(points.length / 2)) },
      { name: "Treatment", point: pick(points.length - 1) }
    ];
  }, [selectedScenario]);

  return (
    <main className="portal-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">BA</span>
          <div>
            <strong>BACON-NET</strong>
          </div>
        </div>
      </header>

      {trainSettingsOpen ? (
        <div className="train-backdrop" onClick={() => setTrainSettingsOpen(false)}>
          <div className="train-modal train-settings" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="train-modal-head">
              <h3>Training settings</h3>
            </div>
            <label className="train-field">
              <span>Aggregator family</span>
              <select value={trainAggregator} onChange={(event) => setTrainAggregator(event.target.value)}>
                {AGGREGATOR_FAMILIES.map((family) => (
                  <option key={family.value} value={family.value}>
                    {family.label}
                  </option>
                ))}
              </select>
              <small>The graded-logic operator family BACON uses at each tree node. More parameters coming soon.</small>
            </label>
            <div className="train-modal-actions train-settings-actions">
              <button type="button" className="ghost-button" onClick={() => setTrainSettingsOpen(false)}>
                Cancel
              </button>
              <button type="button" className="train-button" onClick={startTraining}>
                <span className="train-spark" aria-hidden="true" />
                Start training
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {training ? (
        <div className="train-backdrop" onClick={trainStatus === "running" ? undefined : closeTraining}>
          <div className="train-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="train-modal-head">
              <div className={`train-status-dot ${trainStatus}`} />
              <h3>
                {trainStatus === "running"
                  ? "Training BACON model…"
                  : trainStatus === "done"
                    ? "Training complete"
                    : "Training stopped"}
              </h3>
              {trainSummary?.accuracy != null ? (
                <span className="train-acc">{(trainSummary.accuracy * 100).toFixed(1)}% acc</span>
              ) : null}
            </div>

            <TrainLog lines={trainLogs} running={trainStatus === "running"} />

            <div className="train-modal-actions">
              {trainStatus === "running" ? (
                <button type="button" className="ghost-button" onClick={closeTraining}>
                  Cancel
                </button>
              ) : (
                <button type="button" className="primary-button" onClick={closeTraining}>
                  {learnedTree ? "View learned model" : "Close"}
                </button>
              )}
            </div>
          </div>
        </div>
      ) : null}

      <div className="portal-body">
        <aside className="sidebar">
          <nav className="nav-list" aria-label="Primary">
            {views.map((view) => (
              <button
                key={view}
                className={view === activeView ? "nav-item active" : "nav-item"}
                onClick={() => setActiveView(view)}
              >
                {view}
              </button>
            ))}
          </nav>
        </aside>

        <section className="workspace">
          <div className="content-grid">
            <section className="main-panel">
              {activeView === "Dashboard" ? (
                <div className="dashboard-grid">
                  <article className="panel viewport tree-view">
                    <div className="panel-header">
                      <div>
                        <p className="eyebrow">Diagnosis tree</p>
                        <h2>BACON aggregation tree</h2>
                      </div>
                      <button
                        className="reset-button"
                        onClick={() => setActiveView("Model Editor")}
                        title="Open the model editor"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                          <path d="M12 20h9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Edit
                      </button>
                    </div>
                    <div className="tile-body tile-body-flow">
                      <DiagnosisTree tree={prunedTree ?? selectedScenario.tree} />
                    </div>
                  </article>

                  <article className="panel viewport cost-view">
                    <div className="panel-header">
                      <div>
                        <p className="eyebrow">Cost view</p>
                        <h2>Cost reduction versus all-features baseline</h2>
                      </div>
                    </div>
                    <div className="tile-body">
                      <div className="cost-figures">
                        <div className="cost-figure">
                          <strong>{selectedScenario.metrics.costReduction}%</strong>
                          <span>cost reduction</span>
                        </div>
                        <div className="cost-figure">
                          <strong>{formatMoney(selectedScenario.metrics.totalSavings)}</strong>
                          <span>total saved</span>
                        </div>
                      </div>
                      <div className="package-summary">
                        <span>${selectedScenario.metrics.savingsPerPatient.toLocaleString()} saving per patient</span>
                        <span>{costSummary.selectedPackages}/{costSummary.totalPackages} test packages required</span>
                        <span>{costSummary.usedFeatures}/{costSummary.totalFeatures} features needed</span>
                      </div>
                      <div className="package-table">
                        {selectedScenario.testPackages.map((pkg) => (
                          <div key={pkg.name} className={pkg.selected ? "pkg-row selected" : "pkg-row"}>
                            <div className="pkg-name">
                              {pkg.name}
                              <span className="pkg-cost">${pkg.cost.toLocaleString()} / patient</span>
                            </div>
                            <div className="pkg-features">
                              {pkg.features.map((feature) => (
                                <span
                                  key={feature.name}
                                  className={`feat${feature.used ? " used" : pkg.selected ? "" : " inactive"}`}
                                >
                                  {feature.name}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </article>

                  <article className="panel viewport threshold-view">
                    <div className="panel-header">
                      <div>
                        <p className="eyebrow">Threshold tuning</p>
                        <h2>Trade off recall and precision</h2>
                      </div>
                      <button
                        className="reset-button"
                        onClick={() => setThreshold(optimalThreshold)}
                        disabled={threshold === optimalThreshold}
                        title={`Reset to optimum threshold (${optimalThreshold}%)`}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                          <path d="M3 12a9 9 0 1 0 3-6.7L3 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          <path d="M3 4v4h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Reset
                      </button>
                    </div>
                    <div className="tile-body">
                      <div className="threshold-split">
                        <div className="threshold-plot">
                          <PredictionCurve threshold={threshold} />
                        </div>
                        <div className="threshold-controls">
                          <div className="slider-head">
                            <span>Threshold</span>
                            <strong>{threshold}%</strong>
                          </div>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            value={threshold}
                            onChange={(event) => setThreshold(Number(event.target.value))}
                          />
                          <div className="scorecard-row">
                            <div>
                              <span>Recall</span>
                              <strong>{(score.recall * 100).toFixed(1)}%</strong>
                            </div>
                            <div>
                              <span>Precision</span>
                              <strong>{(score.precision * 100).toFixed(1)}%</strong>
                            </div>
                            <div>
                              <span>Specificity</span>
                              <strong>{(score.specificity * 100).toFixed(1)}%</strong>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="threshold-chart">
                        {clinicalStages.map((stage) => (
                          <div key={stage.name} className="threshold-row">
                            <div className="stage-head">
                              <span className="stage-name">{stage.name}</span>
                              <span className="stage-threshold">{Math.round(stage.point.threshold * 100)}%</span>
                            </div>
                            <div className="threshold-metrics">
                              <i style={{ width: `${stage.point.recall * 100}%` }} className="recall" />
                              <i style={{ width: `${stage.point.precision * 100}%` }} className="precision" />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </article>
                </div>
              ) : activeView === "Model Editor" ? (
                <article className="panel viewport editor-view">
                  <div className="panel-header">
                    <div>
                      <p className="eyebrow">Model editor</p>
                      <div className="editor-title-row">
                        <input
                          type="text"
                          className="model-name-input"
                          placeholder="Enter model name…"
                          value={modelName}
                          onChange={(event) => setModelName(event.target.value)}
                          aria-label="Model name"
                        />
                        <select
                          className="dataset-ref-select"
                          value={editorDatasetId}
                          onChange={(event) => selectEditorDataset(event.target.value)}
                          aria-label="Dataset reference"
                          title="Select a dataset reference"
                        >
                          <option value="">Select dataset…</option>
                          {datasetRefs.map((ref) => (
                            <option key={ref.id} value={ref.id}>
                              {ref.name}
                            </option>
                          ))}
                        </select>
                        <button
                          type="button"
                          className="train-button"
                          onClick={() => setTrainSettingsOpen(true)}
                          disabled={!editorDatasetId}
                          title={editorDatasetId ? "Configure and train a BACON model on this dataset" : "Select a dataset first"}
                        >
                          <span className="train-spark" aria-hidden="true" />
                          Train
                        </button>
                      </div>
                    </div>
                    <button className="reset-button" onClick={() => setActiveView("Dashboard")} title="Back to dashboard">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Back
                    </button>
                  </div>
                  <div className="tile-body tile-body-flow">
                    <TreeEditor
                      key={treeKey}
                      tree={learnedTree ?? prunedTree ?? []}
                      features={editorFeatures}
                      aggregatorFamily={trainedAggregator ?? undefined}
                      onSave={saveModel}
                    />
                  </div>
                </article>
              ) : activeView === "Data Browser" ? (
                <article className="panel viewport editor-view">
                  <div className="panel-header">
                    <div>
                      <p className="eyebrow">Data Browser</p>
                      <h2>{browserDataset ? browserDataset.name : "Browse dataset contents"}</h2>
                    </div>
                    <button className="reset-button" onClick={() => setActiveView("DataSets")} title="Back to datasets">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M12 19l-7-7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Back
                    </button>
                  </div>
                  <div className="tile-body">
                    <DataBrowserPage dataset={browserDataset} />
                  </div>
                </article>
              ) : activeView === "Models" ? (
                <article className="panel viewport editor-view">
                  <div className="panel-header">
                    <div>
                      <p className="eyebrow">Models</p>
                      <h2>Disease models and versions</h2>
                    </div>
                  </div>
                  <div className="tile-body">
                    <ModelsPage onEdit={openModelInEditor} />
                  </div>
                </article>
              ) : activeView === "DataSets" ? (
                <article className="panel viewport editor-view">
                  <div className="panel-header">
                    <div>
                      <p className="eyebrow">DataSets</p>
                      <h2>Training and inference datasets</h2>
                    </div>
                  </div>
                  <div className="tile-body">
                    <DataSetsPage
                      onOpenDataset={(dataset) => {
                        setBrowserDataset(dataset);
                        setActiveView("Data Browser");
                      }}
                    />
                  </div>
                </article>
              ) : activeView === "Distillation" ? (
                <article className="panel viewport editor-view">
                  <div className="panel-header">
                    <div>
                      <p className="eyebrow">Distillation</p>
                      <h2>Compress teacher models into deployable students</h2>
                    </div>
                  </div>
                  <div className="tile-body">
                    <DistillationPage />
                  </div>
                </article>
              ) : (
                <article className="panel empty-state">
                  <p className="eyebrow">{activeView}</p>
                  <h2>Connect external data sources</h2>
                  <p>
                    This area is ready for the next screen. The persistent chat rail stays visible so the agent can help
                    with integration steps, feature selection, and model iteration without losing context.
                  </p>
                </article>
              )}
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}