"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import {
  addEdge,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { DiagnosisTreeNode } from "./DiagnosisTree";

type FlowNodeData = {
  label: string;
  count: number;
  operator?: string;
  andness?: number;
  orphan?: boolean;
};

type PaletteKind = "aggregator" | "leaf";

type DropPayload = {
  kind: PaletteKind;
  label: string;
  operator?: string;
  andness?: number;
};

const X_GAP = 170;
const Y_GAP = 104;

type PaletteOperator = { label: string; andness: number; name?: string };

// The graded Conjunction/Disjunction (GCD) operator set (Dujmović, Graded
// Logic). andness spans -1 (drastic disjunction) to 2 (drastic conjunction),
// matching bacon's andness = sigmoid(bias)*3 - 1. Range operators (HHC, LHC,
// LHD, HHD) use a representative andness within their interval.
const AGGREGATORS: PaletteOperator[] = [
  { label: "CC", andness: 2, name: "Drastic conjunction" },
  { label: "HHC", andness: 1.625, name: "High hyper-conjunction" },
  { label: "CP", andness: 1.25, name: "Product t-norm" },
  { label: "LHC", andness: 1.125, name: "Low hyper-conjunction" },
  { label: "C", andness: 1, name: "Pure conjunction" },
  { label: "HC+", andness: 0.929, name: "High hard conjunction" },
  { label: "HC", andness: 0.857, name: "Medium hard conjunction" },
  { label: "HC-", andness: 0.786, name: "Low hard conjunction" },
  { label: "SC+", andness: 0.714, name: "High soft conjunction" },
  { label: "SC", andness: 0.643, name: "Medium soft conjunction" },
  { label: "SC-", andness: 0.571, name: "Low soft conjunction" },
  { label: "A", andness: 0.5, name: "Arithmetic mean (neutral)" },
  { label: "SD-", andness: 0.429, name: "Low soft disjunction" },
  { label: "SD", andness: 0.357, name: "Medium soft disjunction" },
  { label: "SD+", andness: 0.286, name: "High soft disjunction" },
  { label: "HD-", andness: 0.214, name: "Low hard disjunction" },
  { label: "HD", andness: 0.143, name: "Medium hard disjunction" },
  { label: "HD+", andness: 0.071, name: "High hard disjunction" },
  { label: "D", andness: 0, name: "Pure disjunction" },
  { label: "LHD", andness: -0.125, name: "Low hyper-disjunction" },
  { label: "DP", andness: -0.25, name: "Product t-conorm" },
  { label: "HHD", andness: -0.625, name: "High hyper-disjunction" },
  { label: "DD", andness: -1, name: "Drastic disjunction" }
];

// The palette reflects the aggregator family used to train the model. Discrete
// families (bool.min_max, math operator sets) expose named operators; LSP / GL
// families are graded and use the GCD andness spectrum above.
function paletteForFamily(family?: string): PaletteOperator[] {
  const AND: PaletteOperator = { label: "AND", andness: 2 };
  const OR: PaletteOperator = { label: "OR", andness: -1 };
  switch (family) {
    case "bool.min_max":
    case "math.operator_set.logic":
      return [AND, OR];
    case "math.operator_set.logic_identity":
      return [AND, OR, { label: "IDENTITY", andness: 0.5 }];
    case "math.operator_set.arith":
      return [
        { label: "ADD", andness: 0.5 },
        { label: "SUB", andness: 0.5 },
        { label: "MUL", andness: 0.5 },
        { label: "DIV", andness: 0.5 },
        { label: "IDENTITY", andness: 0.5 }
      ];
    default:
      // lsp.full_weight / lsp.half_weight / lsp.softmax / gl.generic / unset
      return AGGREGATORS;
  }
}

function isGradedFamily(family?: string): boolean {
  return (
    family == null ||
    family.startsWith("lsp.") ||
    family === "gl.generic"
  );
}

// map andness [-1, 2] -> hue (warm/orange = disjunctive, cool/cyan = conjunctive)
function andnessColor(andness: number, alpha = 1) {
  const norm = Math.min(1, Math.max(0, (andness + 1) / 3));
  const hue = 25 + norm * 175;
  return `hsla(${hue.toFixed(0)}, 78%, 62%, ${alpha})`;
}

function buildGraph(roots: DiagnosisTreeNode[]) {
  const nodes: Node<FlowNodeData>[] = [];
  const edges: Edge[] = [];
  let leafCursor = 0;
  let idCounter = 0;

  const walk = (node: DiagnosisTreeNode, depth: number, parentId?: string): number => {
    const id = `node-${idCounter++}`;
    const isAggregator = Boolean(node.children && node.children.length > 0);

    let x: number;
    if (isAggregator) {
      const childXs = node.children!.map((child) => walk(child, depth + 1, id));
      x = (childXs[0] + childXs[childXs.length - 1]) / 2;
    } else {
      x = leafCursor * X_GAP;
      leafCursor += 1;
    }

    nodes.push({
      id,
      type: isAggregator ? "aggregator" : "leaf",
      position: { x, y: depth * Y_GAP },
      data: { label: node.label, count: node.count, operator: node.operator, andness: node.andness }
    });

    if (parentId) {
      edges.push({
        id: `${parentId}->${id}`,
        source: parentId,
        target: id,
        type: "smoothstep",
        style: { stroke: "rgba(105, 210, 255, 0.45)", strokeWidth: 2 }
      });
    }

    return x;
  };

  roots.forEach((root) => walk(root, 0));
  return { nodes, edges: normalizeAll(edges) };
}

// Serialize the current editor graph (nodes + edges) back into the nested
// DiagnosisTreeNode form for saving / display. Edges point parent -> child.
function graphToTree(nodes: Node<FlowNodeData>[], edges: Edge[]): DiagnosisTreeNode[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const childIds = new Map<string, string[]>();
  const hasParent = new Set<string>();
  edges.forEach((e) => {
    if (!childIds.has(e.source)) childIds.set(e.source, []);
    childIds.get(e.source)!.push(e.target);
    hasParent.add(e.target);
  });

  const build = (id: string): DiagnosisTreeNode => {
    const node = byId.get(id)!;
    const kids = childIds.get(id) ?? [];
    if (kids.length === 0) {
      return { label: String(node.data.label), count: 1 };
    }
    const children = kids.map(build);
    const result: DiagnosisTreeNode = {
      label: String(node.data.label),
      count: children.reduce((sum, c) => sum + c.count, 0),
      children
    };
    if (typeof node.data.operator === "string") result.operator = node.data.operator;
    if (typeof node.data.andness === "number") result.andness = node.data.andness;
    return result;
  };

  return nodes.filter((n) => !hasParent.has(n.id)).map((n) => build(n.id));
}

// tidy top-down tree layout derived from the current edges. Parents are centered
// over their children; disconnected (orphan) nodes are lined up underneath.
function layoutNodes(nodes: Node<FlowNodeData>[], edges: Edge[]): Node<FlowNodeData>[] {
  if (nodes.length === 0) return nodes;
  const childrenOf = new Map<string, string[]>();
  const hasParent = new Set<string>();
  for (const e of edges) {
    const list = childrenOf.get(e.source) ?? [];
    list.push(e.target);
    childrenOf.set(e.source, list);
    hasParent.add(e.target);
  }

  const positions = new Map<string, { x: number; y: number }>();
  const visited = new Set<string>();
  let leafCursor = 0;

  const place = (id: string, depth: number): number => {
    if (visited.has(id)) {
      return positions.get(id)?.x ?? leafCursor * X_GAP;
    }
    visited.add(id);
    const children = childrenOf.get(id)?.filter((c) => c !== id) ?? [];
    let x: number;
    if (children.length > 0) {
      const childXs = children.map((c) => place(c, depth + 1));
      x = (Math.min(...childXs) + Math.max(...childXs)) / 2;
    } else {
      x = leafCursor * X_GAP;
      leafCursor += 1;
    }
    positions.set(id, { x, y: depth * Y_GAP });
    return x;
  };

  // roots = connected nodes with no incoming edge
  const roots = nodes
    .filter((n) => !hasParent.has(n.id) && (childrenOf.get(n.id)?.length ?? 0) > 0)
    .map((n) => n.id);
  roots.forEach((id) => place(id, 0));

  // any remaining connected nodes (e.g. cycles or unreached) get placed too
  nodes.forEach((n) => {
    if (!positions.has(n.id) && (hasParent.has(n.id) || (childrenOf.get(n.id)?.length ?? 0) > 0)) {
      place(n.id, 0);
    }
  });

  // orphans: nodes with no edges at all, lined up on a row below the tree
  const maxDepth = Math.max(0, ...Array.from(positions.values()).map((p) => p.y / Y_GAP));
  let orphanCursor = 0;
  nodes.forEach((n) => {
    if (!positions.has(n.id)) {
      positions.set(n.id, { x: orphanCursor * X_GAP, y: (maxDepth + 1.5) * Y_GAP });
      orphanCursor += 1;
    }
  });

  return nodes.map((n) => {
    const pos = positions.get(n.id);
    return pos ? { ...n, position: pos } : n;
  });
}

const EDGE_BASE = {
  type: "smoothstep" as const,
  style: { stroke: "rgba(105, 210, 255, 0.45)", strokeWidth: 2 },
  labelBgPadding: [5, 2] as [number, number],
  labelBgBorderRadius: 4,
  labelBgStyle: { fill: "rgba(8, 14, 24, 0.92)", stroke: "rgba(105, 210, 255, 0.4)" },
  labelStyle: { fill: "#dce8f7", fontSize: 11, fontWeight: 700 }
};

function withWeight(edge: Edge, weight: number): Edge {
  const w = Math.max(0, Math.min(1, weight));
  return { ...edge, ...EDGE_BASE, data: { ...edge.data, weight: w }, label: w.toFixed(2) };
}

// rescale each aggregator's child edges so they sum to 1.
// edges flow parent(aggregator) -> child, so siblings share the same source.
function normalizeAll(edges: Edge[]): Edge[] {
  const bySource = new Map<string, Edge[]>();
  for (const e of edges) {
    const list = bySource.get(e.source) ?? [];
    list.push(e);
    bySource.set(e.source, list);
  }
  const out: Edge[] = [];
  for (const list of bySource.values()) {
    const total = list.reduce((sum, e) => sum + (typeof e.data?.weight === "number" ? (e.data.weight as number) : 0), 0);
    list.forEach((e) => {
      const current = typeof e.data?.weight === "number" ? (e.data.weight as number) : 0;
      const weight = total > 0 ? current / total : 1 / list.length;
      out.push(withWeight(e, weight));
    });
  }
  return out;
}

// renormalize only the given aggregators (sources), e.g. after a deletion
function rebalanceSubset(edges: Edge[], sources: Set<string>): Edge[] {
  const grouped = new Map<string, Edge[]>();
  for (const e of edges) {
    if (!sources.has(e.source)) continue;
    const list = grouped.get(e.source) ?? [];
    list.push(e);
    grouped.set(e.source, list);
  }
  const replaced = new Map<string, Edge>();
  for (const list of grouped.values()) {
    const total = list.reduce((sum, e) => sum + (typeof e.data?.weight === "number" ? (e.data.weight as number) : 0), 0);
    list.forEach((e) => {
      const current = typeof e.data?.weight === "number" ? (e.data.weight as number) : 0;
      const weight = total > 0 ? current / total : 1 / list.length;
      replaced.set(e.id, withWeight(e, weight));
    });
  }
  return edges.map((e) => replaced.get(e.id) ?? e);
}

// set one edge's weight and proportionally rebalance its siblings so the
// aggregator's child weights still sum to 1
function applyWeight(edges: Edge[], edgeId: string, weight: number): Edge[] {
  const edge = edges.find((e) => e.id === edgeId);
  if (!edge) return edges;
  const siblings = edges.filter((e) => e.source === edge.source && e.id !== edgeId);
  const w = Math.max(0, Math.min(1, weight));
  if (siblings.length === 0) {
    return edges.map((e) => (e.id === edgeId ? withWeight(e, 1) : e));
  }
  const remaining = 1 - w;
  const sibTotal = siblings.reduce((sum, e) => sum + (typeof e.data?.weight === "number" ? (e.data.weight as number) : 0), 0);
  return edges.map((e) => {
    if (e.id === edgeId) return withWeight(e, w);
    if (e.source !== edge.source) return e;
    const current = typeof e.data?.weight === "number" ? (e.data.weight as number) : 0;
    const share = sibTotal > 0 ? (current / sibTotal) * remaining : remaining / siblings.length;
    return withWeight(e, share);
  });
}

function AggregatorNode({ data, selected }: NodeProps<Node<FlowNodeData>>) {
  const tinted = typeof data.andness === "number";
  const style = tinted
    ? {
        background: `radial-gradient(circle at 30% 30%, ${andnessColor(data.andness!, 0.4)}, ${andnessColor(
          data.andness!,
          0.18
        )})`,
        borderColor: andnessColor(data.andness!, 0.7)
      }
    : undefined;
  return (
    <div
      className={`flow-aggregator${selected ? " selected" : ""}${data.orphan ? " orphan" : ""}`}
      style={style}
      title={
        data.orphan
          ? "Not connected to the tree yet"
          : typeof data.andness === "number"
            ? `andness ${data.andness}`
            : undefined
      }
    >
      <Handle type="target" position={Position.Top} />
      {data.operator ? (
        <span className="flow-aggregator-op">{data.operator}</span>
      ) : (
        <>
          <span className="flow-aggregator-count">{data.count.toLocaleString()}</span>
          <span className="flow-aggregator-label">{data.label}</span>
        </>
      )}
      {data.orphan ? <span className="orphan-badge">unlinked</span> : null}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function LeafNode({ data, selected }: NodeProps<Node<FlowNodeData>>) {
  return (
    <div
      className={`flow-leaf${selected ? " selected" : ""}${data.orphan ? " orphan" : ""}`}
      title={data.orphan ? "Not connected to an aggregator yet" : undefined}
    >
      <Handle type="target" position={Position.Top} />
      <strong>{data.label}</strong>
      {data.orphan ? <span className="orphan-badge">unlinked</span> : null}
    </div>
  );
}

const nodeTypes: NodeTypes = {
  aggregator: AggregatorNode,
  leaf: LeafNode
};

function EditorCanvas({
  tree,
  features,
  aggregatorFamily,
  onSave
}: {
  tree: DiagnosisTreeNode[];
  features: string[];
  aggregatorFamily?: string;
  onSave?: (tree: DiagnosisTreeNode[], stats: { nodes: number; links: number }) => Promise<void>;
}) {
  const initial = useMemo(() => buildGraph(tree), [tree]);
  const paletteOperators = useMemo(() => paletteForFamily(aggregatorFamily), [aggregatorFamily]);
  const gradedPalette = isGradedFamily(aggregatorFamily);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [status, setStatus] = useState<string>("Drag items from the palette or connect nodes to build the tree.");
  const idRef = useRef(initial.nodes.length);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  // Keep the latest nodes/edges available to the feature-prune effect without
  // making it depend on (and re-run for) every graph change.
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  // When the selected dataset's features change, drop any leaf nodes whose
  // feature no longer exists in the dataset (and the edges touching them).
  // Skip the initial mount so the starting diagram is left intact.
  const featuresReady = useRef(false);
  useEffect(() => {
    if (!featuresReady.current) {
      featuresReady.current = true;
      return;
    }
    const allowed = new Set(features);
    const removedIds = new Set(
      nodesRef.current
        .filter((n) => n.type === "leaf" && !allowed.has(String(n.data.label)))
        .map((n) => n.id)
    );
    if (removedIds.size === 0) {
      return;
    }
    const sources = new Set(
      edgesRef.current.filter((e) => removedIds.has(e.target)).map((e) => e.source)
    );
    setNodes((nds) => nds.filter((n) => !removedIds.has(n.id)));
    setEdges((eds) =>
      rebalanceSubset(eds.filter((e) => !removedIds.has(e.source) && !removedIds.has(e.target)), sources)
    );
    setStatus(`Removed ${removedIds.size} feature node${removedIds.size === 1 ? "" : "s"} not in the selected dataset`);
  }, [features, setNodes, setEdges]);

  const onConnect = useCallback(
    (connection: Connection) =>
      setEdges((eds) => {
        const next = addEdge({ ...connection, ...EDGE_BASE }, eds);
        if (connection.source) {
          return rebalanceSubset(next, new Set([connection.source]));
        }
        return next;
      }),
    [setEdges]
  );

  const onEdgesDelete = useCallback(
    (deleted: Edge[]) => {
      const deletedIds = new Set(deleted.map((e) => e.id));
      const sources = new Set(deleted.map((e) => e.source));
      setEdges((eds) => rebalanceSubset(eds.filter((e) => !deletedIds.has(e.id)), sources));
    },
    [setEdges]
  );

  const spawnNode = useCallback(
    (payload: DropPayload, position: { x: number; y: number }) => {
      const id = `new-${idRef.current++}`;
      setNodes((nds) =>
        nds.concat({
          id,
          type: payload.kind,
          position,
          data:
            payload.kind === "aggregator"
              ? { label: `Operator ${payload.operator}`, count: 0, operator: payload.operator, andness: payload.andness }
              : { label: payload.label, count: 0 }
        })
      );
    },
    [setNodes]
  );

  const onDragStart = (event: DragEvent, payload: DropPayload) => {
    event.dataTransfer.setData("application/screenwise-node", JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "move";
  };

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      const raw = event.dataTransfer.getData("application/screenwise-node");
      if (!raw) return;
      let payload: DropPayload;
      try {
        payload = JSON.parse(raw) as DropPayload;
      } catch {
        return;
      }
      if (payload.kind !== "aggregator" && payload.kind !== "leaf") return;

      // if an aggregator is dropped onto an existing aggregator, replace its
      // operator in place and keep all connected edges
      if (payload.kind === "aggregator") {
        const target = (event.target as HTMLElement | null)?.closest(".react-flow__node");
        const targetId = target?.getAttribute("data-id");
        if (targetId) {
          let replaced = false;
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id !== targetId || n.type !== "aggregator") return n;
              replaced = true;
              return {
                ...n,
                data: {
                  ...n.data,
                  label: `Operator ${payload.operator}`,
                  operator: payload.operator,
                  andness: payload.andness
                }
              };
            })
          );
          if (replaced) {
            setStatus(`Replaced operator with ${payload.operator} (andness ${payload.andness})`);
            return;
          }
        }
      }

      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      spawnNode(payload, position);
    },
    [screenToFlowPosition, spawnNode, setNodes]
  );

  const addToCenter = (payload: DropPayload) => {
    const rect = wrapperRef.current?.getBoundingClientRect();
    const position = screenToFlowPosition({
      x: (rect?.left ?? 0) + (rect?.width ?? 600) / 2,
      y: (rect?.top ?? 0) + (rect?.height ?? 400) / 2
    });
    spawnNode(payload, position);
  };

  const deleteSelected = () => {
    const removedSources = new Set(edges.filter((e) => e.selected).map((e) => e.source));
    const removedNodeIds = new Set(nodes.filter((n) => n.selected).map((n) => n.id));
    setNodes((nds) => nds.filter((n) => !n.selected));
    setEdges((eds) => {
      const kept = eds.filter(
        (e) => !e.selected && !removedNodeIds.has(e.source) && !removedNodeIds.has(e.target)
      );
      return rebalanceSubset(kept, removedSources);
    });
  };

  const clearAll = () => {
    setNodes([]);
    setEdges([]);
    setStatus("Cleared all nodes and links");
  };

  const setWeight = (edgeId: string, weight: number) => {
    setEdges((eds) => applyWeight(eds, edgeId, weight));
  };

  const rearrange = () => {
    setNodes((nds) => layoutNodes(nds, edges));
    setStatus("Re-arranged layout");
    window.setTimeout(() => fitView({ padding: 0.12, duration: 400 }), 0);
  };

  const reEvaluate = () => {
    setStatus(`Re-evaluated · ${nodes.length} nodes, ${edges.length} links`);
  };

  const save = async () => {
    const treeData = graphToTree(nodes, edges);
    if (!onSave) {
      setStatus(`Saved tree · ${nodes.length} nodes, ${edges.length} links`);
      return;
    }
    if (treeData.length === 0) {
      setStatus("Nothing to save — build or train a model first.");
      return;
    }
    setStatus("Saving model…");
    try {
      await onSave(treeData, { nodes: nodes.length, links: edges.length });
      setStatus(`Saved model · ${nodes.length} nodes, ${edges.length} links`);
    } catch (error) {
      setStatus(`Save failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const selectedEdge = edges.find((e) => e.selected);
  const hasSelection = nodes.some((n) => n.selected) || edges.some((e) => e.selected);

  // a node is an orphan until at least one edge touches it
  const displayNodes = useMemo(() => {
    const connected = new Set<string>();
    edges.forEach((e) => {
      connected.add(e.source);
      connected.add(e.target);
    });
    return nodes.map((n) =>
      n.data.orphan === !connected.has(n.id)
        ? n
        : { ...n, data: { ...n.data, orphan: !connected.has(n.id) } }
    );
  }, [nodes, edges]);

  return (
    <div className="editor-layout">
      <div className="editor-stage">
        <div className="editor-canvas" ref={wrapperRef} onDrop={onDrop} onDragOver={onDragOver}>
          <ReactFlow
            nodes={displayNodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onEdgesDelete={onEdgesDelete}
            fitView
            fitViewOptions={{ padding: 0.12 }}
            minZoom={0.3}
            maxZoom={1.8}
            nodesDraggable
            nodesConnectable
            elementsSelectable
            deleteKeyCode={["Backspace", "Delete"]}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="rgba(161, 182, 214, 0.18)" />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>

        {selectedEdge ? (
          <div className="canvas-overlay top-left weight-editor">
            <span>Edge weight</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={typeof selectedEdge.data?.weight === "number" ? (selectedEdge.data.weight as number) : 0}
              onChange={(event) => setWeight(selectedEdge.id, Number(event.target.value))}
            />
            <strong>
              {(typeof selectedEdge.data?.weight === "number" ? (selectedEdge.data.weight as number) : 0).toFixed(2)}
            </strong>
            <span className="weight-hint">siblings sum to 1</span>
          </div>
        ) : null}

        <div className="canvas-overlay top-right">
          <button type="button" className="ghost-button" onClick={deleteSelected} disabled={!hasSelection}>
            Delete selected
          </button>
          <button type="button" className="ghost-button" onClick={clearAll} disabled={nodes.length === 0 && edges.length === 0}>
            Clear all
          </button>
        </div>

        <div className="editor-toolbar">
          <div className="editor-toolbar-status">
            <span>{status}</span>
          </div>
          <div className="editor-toolbar-actions">
            <button type="button" className="ghost-button" onClick={rearrange}>
              Re-arrange
            </button>
            <button type="button" className="ghost-button" onClick={reEvaluate}>
              Re-evaluate
            </button>
            <button type="button" className="primary-button" onClick={save}>
              Save
            </button>
          </div>
        </div>
      </div>

      <aside className="editor-palette">
        <div className="palette-section">
          <p className="eyebrow">Aggregator operators</p>
          <div className="andness-legend">
            <span>disjunctive</span>
            <span className="andness-bar" />
            <span>conjunctive</span>
          </div>
          <div className={`palette-aggregators${gradedPalette ? "" : " palette-aggregators-named"}`}>
            {paletteOperators.map((agg) => (
              <button
                key={agg.label}
                type="button"
                className={`agg-chip${gradedPalette ? "" : " agg-chip-named"}`}
                style={{
                  background: andnessColor(agg.andness, 0.22),
                  borderColor: andnessColor(agg.andness, 0.7),
                  color: andnessColor(agg.andness, 1)
                }}
                draggable
                onDragStart={(event) =>
                  onDragStart(event, { kind: "aggregator", label: agg.label, operator: agg.label, andness: agg.andness })
                }
                onDoubleClick={() =>
                  addToCenter({ kind: "aggregator", label: agg.label, operator: agg.label, andness: agg.andness })
                }
                title={
                  agg.name
                    ? `${agg.label} · ${agg.name} · andness ${agg.andness}`
                    : `Operator ${agg.label}`
                }
              >
                {agg.label}
              </button>
            ))}
          </div>
        </div>

        <div className="palette-section palette-section-features">
          <p className="eyebrow">Dataset features</p>
          <div className="palette-features">
            {features.map((feature) => (
              <div
                key={feature}
                className="palette-feature"
                draggable
                onDragStart={(event) => onDragStart(event, { kind: "leaf", label: feature })}
                onDoubleClick={() => addToCenter({ kind: "leaf", label: feature })}
                role="button"
                tabIndex={0}
                title={`Add "${feature}" node`}
              >
                <span className="palette-glyph square" />
                <span>{feature}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}

export default function TreeEditor({
  tree,
  features,
  aggregatorFamily,
  onSave
}: {
  tree: DiagnosisTreeNode[];
  features: string[];
  aggregatorFamily?: string;
  onSave?: (tree: DiagnosisTreeNode[], stats: { nodes: number; links: number }) => Promise<void>;
}) {
  return (
    <ReactFlowProvider>
      <EditorCanvas tree={tree} features={features} aggregatorFamily={aggregatorFamily} onSave={onSave} />
    </ReactFlowProvider>
  );
}
