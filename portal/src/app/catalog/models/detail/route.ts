import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { load } from "js-yaml";
import { NextResponse } from "next/server";

// Server-side only: read a saved model's embedded display metadata (feature
// names, aggregator, accuracy, tree) directly from its single .pth checkpoint
// via portal/scripts/read_model.py. The .pth is the source of truth for display.

export const dynamic = "force-dynamic";

type ModelEntry = { id: string; location?: string };

async function readCatalog(): Promise<ModelEntry[]> {
  const raw = await readFile(path.join(process.cwd(), "data", "models.yaml"), "utf8");
  const parsed = load(raw) as { models?: ModelEntry[] } | null;
  return Array.isArray(parsed?.models) ? parsed.models : [];
}

export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id");
  if (!id) {
    return NextResponse.json({ error: "Missing model id." }, { status: 400 });
  }

  const entry = (await readCatalog()).find((m) => m.id === id);
  if (!entry) {
    return NextResponse.json({ error: `Model "${id}" not found.` }, { status: 404 });
  }
  if (!entry.location) {
    return NextResponse.json({ error: "This model has no saved checkpoint." }, { status: 404 });
  }

  const repoRoot = path.resolve(process.cwd(), "..");
  const pthPath = path.resolve(repoRoot, entry.location);
  if (pthPath !== repoRoot && !pthPath.startsWith(repoRoot + path.sep)) {
    return NextResponse.json({ error: "Checkpoint path is outside the repository." }, { status: 400 });
  }

  const scriptPath = path.join(process.cwd(), "scripts", "read_model.py");
  const pythonCmd = process.env.PYTHON ?? "python";

  const result = await new Promise<{ ok: boolean; body: string }>((resolve) => {
    const child = spawn(pythonCmd, [scriptPath, "--pth", pthPath, "--repo", repoRoot], {
      cwd: repoRoot,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" }
    });
    let out = "";
    let err = "";
    child.stdout.on("data", (chunk: Buffer) => (out += chunk.toString("utf8")));
    child.stderr.on("data", (chunk: Buffer) => (err += chunk.toString("utf8")));
    child.on("error", (e) => resolve({ ok: false, body: JSON.stringify({ error: e.message }) }));
    child.on("close", () => {
      // The script prints a single JSON line; fall back to stderr on failure.
      const line = out.trim().split(/\r?\n/).filter(Boolean).pop();
      resolve({ ok: Boolean(line), body: line ?? JSON.stringify({ error: err.trim() || "No output" }) });
    });
  });

  try {
    const parsed = JSON.parse(result.body) as { metadata?: unknown; error?: string };
    if (parsed.error) {
      return NextResponse.json({ error: parsed.error }, { status: 500 });
    }
    return NextResponse.json({ metadata: parsed.metadata ?? null });
  } catch {
    return NextResponse.json({ error: "Could not parse model metadata." }, { status: 500 });
  }
}
