import { readFile, writeFile, rm, mkdir } from "node:fs/promises";
import path from "node:path";
import { load, dump } from "js-yaml";
import { NextResponse } from "next/server";

// Server-side only: datasets.yaml is the single source of truth for the Datasets
// screen. GET returns it as JSON; POST adds an entry (writing its CSV to a data
// file and pointing `location` at it); DELETE removes an entry (and, only when
// explicitly requested, the pointed-to local data folder/file). No external
// backend service is required.

export const dynamic = "force-dynamic";

type DatasetEntry = {
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
};

type DatasetCatalog = {
  datasets: DatasetEntry[];
};

const CATALOG_HEADER = `# BACON-NET portal dataset catalog.
#
# This file is the source of truth for the Datasets screen. The portal reads it
# server-side (see src/app/catalog/datasets/route.ts) and renders the table from
# it, so there is no external backend to run.
#
# Each entry describes one dataset and points to where the actual data lives
# via \`location\` (a repo-relative path or a URL).
#
# Fields:
#   id        unique slug
#   name      display name
#   source    short provenance label
#   modality  Tabular | Boolean | Imaging | FHIR R4 | ...
#   samples   number of rows / samples
#   features  number of input columns
#   labeled   number of labeled rows
#   updated   YYYY-MM-DD
#   status    ready | syncing | review
#   location  repo-relative path or URL to the actual dataset
#   notes     optional free text
`;

function catalogPath(): string {
  return path.join(process.cwd(), "data", "datasets.yaml");
}

async function readCatalog(): Promise<DatasetEntry[]> {
  const raw = await readFile(catalogPath(), "utf8");
  const parsed = load(raw) as DatasetCatalog | null;
  return Array.isArray(parsed?.datasets) ? parsed.datasets : [];
}

async function writeCatalog(datasets: DatasetEntry[]): Promise<void> {
  // Keep `updated` as a plain YYYY-MM-DD string so the YAML stays tidy.
  const normalized = datasets.map((d) => ({ ...d, updated: toDateString(d.updated) }));
  const body = dump({ datasets: normalized }, { lineWidth: 100, noRefs: true });
  await writeFile(catalogPath(), `${CATALOG_HEADER}\n${body}`, "utf8");
}

function toDateString(value: unknown): string {
  const dt = new Date(value as string | number | Date);
  return Number.isNaN(dt.getTime()) ? String(value) : dt.toISOString().slice(0, 10);
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
  const seed = base || "dataset";
  if (!taken.has(seed)) {
    return seed;
  }
  let n = 2;
  while (taken.has(`${seed}-${n}`)) {
    n += 1;
  }
  return `${seed}-${n}`;
}

function isUrl(value: string | undefined): value is string {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

export async function GET() {
  try {
    const datasets = await readCatalog();
    return NextResponse.json({ datasets });
  } catch (error) {
    return NextResponse.json(
      {
        datasets: [],
        error: `Could not read datasets.yaml: ${error instanceof Error ? error.message : String(error)}`
      },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Partial<DatasetEntry> & { csv?: string };
    const name = typeof body.name === "string" ? body.name.trim() : "";
    const csv = typeof body.csv === "string" ? body.csv : "";
    if (!name) {
      return NextResponse.json({ error: "Missing dataset name." }, { status: 400 });
    }
    if (!csv) {
      return NextResponse.json({ error: "Missing CSV data." }, { status: 400 });
    }

    const datasets = await readCatalog();
    const id = uniqueId(slugify(name), new Set(datasets.map((d) => d.id)));

    // Persist the actual data as a CSV file and point `location` at it.
    const dataDir = path.join(process.cwd(), "data", "datasets");
    await mkdir(dataDir, { recursive: true });
    await writeFile(path.join(dataDir, `${id}.csv`), csv, "utf8");

    const entry: DatasetEntry = {
      id,
      name,
      source: typeof body.source === "string" ? body.source : "Portal",
      modality: typeof body.modality === "string" ? body.modality : "Tabular",
      samples: Number(body.samples) || 0,
      features: Number(body.features) || 0,
      labeled: Number(body.labeled) || 0,
      updated: today(),
      status: "ready",
      location: `portal/data/datasets/${id}.csv`
    };

    await writeCatalog([entry, ...datasets]);
    return NextResponse.json({ ok: true, dataset: entry });
  } catch (error) {
    return NextResponse.json(
      { error: `Could not add dataset: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { id, deleteData } = (await request.json()) as { id?: string; deleteData?: boolean };
    if (!id || typeof id !== "string") {
      return NextResponse.json({ error: "Missing dataset id." }, { status: 400 });
    }

    const datasets = await readCatalog();
    const entry = datasets.find((d) => d.id === id);
    if (!entry) {
      return NextResponse.json({ error: `Dataset "${id}" not found.` }, { status: 404 });
    }

    // Remove the catalog entry first (kept in sync with the UI).
    await writeCatalog(datasets.filter((d) => d.id !== id));

    // Optionally delete the pointed-to local data. Off by default: the caller
    // must opt in, and URLs / external sources are never touched.
    let dataDeleted = false;
    if (deleteData === true && entry.location && !isUrl(entry.location)) {
      const repoRoot = path.resolve(process.cwd(), "..");
      const target = path.resolve(repoRoot, entry.location);
      // Guard against path traversal: never delete the repo root or anything
      // outside it.
      if (target === repoRoot || !target.startsWith(repoRoot + path.sep)) {
        return NextResponse.json(
          {
            ok: true,
            dataDeleted: false,
            error: "Catalog entry removed, but the data path is outside the repository and was not deleted."
          },
          { status: 200 }
        );
      }
      await rm(target, { recursive: true, force: true });
      dataDeleted = true;
    }

    return NextResponse.json({ ok: true, dataDeleted });
  } catch (error) {
    return NextResponse.json(
      { error: `Could not remove dataset: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}
