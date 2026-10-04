"use client";

import { useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

export type DiagnosisTreeNode = {
  label: string;
  count: number;
  children?: DiagnosisTreeNode[];
  operator?: string;
  andness?: number;
};

type FlowNodeData = {
  label: string;
  count: number;
};

const X_GAP = 170;
const Y_GAP = 104;

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
      data: { label: node.label, count: node.count }
    });

    if (parentId) {
      edges.push({
        id: `${parentId}->${id}`,
        source: parentId,
        target: id,
        type: "smoothstep",
        animated: false,
        style: { stroke: "rgba(105, 210, 255, 0.45)", strokeWidth: 2 }
      });
    }

    return x;
  };

  roots.forEach((root) => walk(root, 0));
  return { nodes, edges };
}

function AggregatorNode({ data }: NodeProps<Node<FlowNodeData>>) {
  return (
    <div className="flow-aggregator">
      <Handle type="target" position={Position.Top} />
      <span className="flow-aggregator-count">{data.count.toLocaleString()}</span>
      <span className="flow-aggregator-label">{data.label}</span>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function LeafNode({ data }: NodeProps<Node<FlowNodeData>>) {
  return (
    <div className="flow-leaf">
      <Handle type="target" position={Position.Top} />
      <strong>{data.label}</strong>
      <span>{data.count.toLocaleString()} records</span>
    </div>
  );
}

const nodeTypes: NodeTypes = {
  aggregator: AggregatorNode,
  leaf: LeafNode
};

export default function DiagnosisTree({ tree }: { tree: DiagnosisTreeNode[] }) {
  const { nodes, edges } = useMemo(() => buildGraph(tree), [tree]);

  return (
    <div className="flow-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.12 }}
        minZoom={0.3}
        maxZoom={1.8}
        nodesConnectable={false}
        nodesDraggable={false}
        elementsSelectable={false}
        panOnScroll
        zoomOnScroll
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="rgba(161, 182, 214, 0.18)" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
