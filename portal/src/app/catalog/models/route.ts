import { readFile, writeFile, rename, rm, mkdir } from "node:fs/promises";
import path from "node:path";
import { load, dump } from "js-yaml";
import { NextResponse } from "next/server";

// Server-side only: models.yaml is the single source of truth for the Models
// screen. GET returns it as JSON; POST adds a saved model (tree + metadata);
// DELETE removes one. No external backend service is required.

export const dynamic = "force-dynamic";

type ModelEntry = {
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

type ModelCatalog = {
  models: ModelEntry[];
};

const CATALOG_HEADER = `# BACON-NET portal model catalog.
#
# This file is the source of truth for the Models screen. The portal reads and
# writes it server-side (see src/app/catalog/models/route.ts); there is no
# external backend.
#
# Models are created by the Model Editor's "Save" button. The trained bacon
# checkpoint (.pth, pointed to by \`location\`) is the single source of truth for
# display/train/inference; the learned tree and feature names are embedded in
# the checkpoint's metadata. This file is a lightweight catalog index.
#
# Fields:
#   id          unique slug
#   name        display name
#   dataset     dataset id this model references
#   datasetName dataset display name
#   aggregator  aggregator family used (bacon.baconNet._aggregator_registry key)
#   accuracy    final accuracy (0..1) when trained, else null
#   features    list of feature names
#   nodes       node count in the tree
#   links       edge count in the tree
#   status      draft | staging | production
#   updated     YYYY-MM-DD
#   location    repo-relative path to the .pth checkpoint (tree embedded inside)
`;

function catalogPath(): string {
  return path.join(process.cwd(), "data", "models.yaml");
}

async function readCatalog(): Promise<ModelEntry[]> {
  const raw = await readFile(catalogPath(), "utf8");
  const parsed = load(raw) as ModelCatalog | null;
  return Array.isArray(parsed?.models) ? parsed.models : [];
}

function toDateString(value: unknown): string {
  const dt = new Date(value as string | number | Date);
  return Number.isNaN(dt.getTime()) ? String(value) : dt.toISOString().slice(0, 10);
}

async function writeCatalog(models: ModelEntry[]): Promise<void> {
  const normalized = models.map((m) => ({ ...m, updated: toDateString(m.updated) }));
  const body = dump({ models: normalized }, { lineWidth: 120, noRefs: true });
  await writeFile(catalogPath(), `${CATALOG_HEADER}\n${body}`, "utf8");
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function uniqueId(base: string, taken: Set<string>): string {
  const seed = base || "model";
  if (!taken.has(seed)) {
    return seed;
  }
  let n = 2;
  while (taken.has(`${seed}-${n}`)) {
    n += 1;
  }
  return `${seed}-${n}`;
}

export async function GET() {
  try {
    const models = await readCatalog();
    return NextResponse.json({ models });
  } catch (error) {
    return NextResponse.json(
      { models: [], error: `Could not read models.yaml: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Partial<ModelEntry> & { modelStaging?: string };
    const name = typeof body.name === "string" ? body.name.trim() : "";
    if (!name) {
      return NextResponse.json({ error: "Missing model name." }, { status: 400 });
    }

    const models = await readCatalog();
    const id = uniqueId(slugify(name), new Set(models.map((m) => m.id)));

    // Finalize the staged bacon checkpoint (.pth) saved during training into
    // data/models/<id>.pth and point `location` at it. The .pth is the single
    // source of truth for display/train/inference (tree is embedded in it).
    let location: string | undefined;
    const staging = typeof body.modelStaging === "string" ? body.modelStaging : "";
    if (staging && /^[A-Za-z0-9._-]+\.pth$/.test(staging)) {
      const modelsDir = path.join(process.cwd(), "data", "models");
      const source = path.join(modelsDir, ".staging", staging);
      const dest = path.join(modelsDir, `${id}.pth`);
      try {
        await mkdir(modelsDir, { recursive: true });
        await rename(source, dest);
        location = `portal/data/models/${id}.pth`;
      } catch {
        location = undefined;
      }
    }

    if (!location) {
      return NextResponse.json(
        { error: "No trained checkpoint to save. Train the model before saving." },
        { status: 400 }
      );
    }

    const entry: ModelEntry = {
      id,
      name,
      dataset: typeof body.dataset === "string" ? body.dataset : undefined,
      datasetName: typeof body.datasetName === "string" ? body.datasetName : undefined,
      aggregator: typeof body.aggregator === "string" ? body.aggregator : undefined,
      accuracy: typeof body.accuracy === "number" ? body.accuracy : null,
      features: Array.isArray(body.features) ? body.features : [],
      nodes: typeof body.nodes === "number" ? body.nodes : undefined,
      links: typeof body.links === "number" ? body.links : undefined,
      status: "draft",
      updated: today(),
      location
    };

    await writeCatalog([entry, ...models]);
    return NextResponse.json({ ok: true, model: entry });
  } catch (error) {
    return NextResponse.json(
      { error: `Could not save model: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { id } = (await request.json()) as { id?: string };
    if (!id || typeof id !== "string") {
      return NextResponse.json({ error: "Missing model id." }, { status: 400 });
    }
    const models = await readCatalog();
    const entry = models.find((m) => m.id === id);
    if (!entry) {
      return NextResponse.json({ error: `Model "${id}" not found.` }, { status: 404 });
    }
    await writeCatalog(models.filter((m) => m.id !== id));

    // Remove the model's own checkpoint file (not shared data).
    if (entry.location && !/^https?:\/\//i.test(entry.location)) {
      const repoRoot = path.resolve(process.cwd(), "..");
      const targetPath = path.resolve(repoRoot, entry.location);
      if (targetPath.startsWith(repoRoot + path.sep)) {
        await rm(targetPath, { force: true }).catch(() => {});
      }
    }
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: `Could not remove model: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}
