import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { load } from "js-yaml";
import { NextResponse } from "next/server";

// Server-side only: given a dataset id, resolve its `location` from
// datasets.yaml and, when it points to a local CSV (a .csv file or a folder
// containing one), return a parsed preview. URLs and non-tabular locations are
// reported so the client can render an appropriate message.

export const dynamic = "force-dynamic";

const MAX_ROWS = 500;

type DatasetEntry = {
  id: string;
  location?: string;
  modality?: string;
};

type DatasetCatalog = {
  datasets: DatasetEntry[];
};

function isUrl(value: string | undefined): value is string {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

async function readCatalog(): Promise<DatasetEntry[]> {
  const raw = await readFile(path.join(process.cwd(), "data", "datasets.yaml"), "utf8");
  const parsed = load(raw) as DatasetCatalog | null;
  return Array.isArray(parsed?.datasets) ? parsed.datasets : [];
}

function parseCsv(text: string) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length === 0) {
    return { header: [] as string[], rows: [] as string[][], total: 0, truncated: false };
  }
  const header = lines[0].split(",").map((cell) => cell.trim());
  const dataLines = lines.slice(1);
  const rows = dataLines.slice(0, MAX_ROWS).map((line) => line.split(",").map((cell) => cell.trim()));
  return { header, rows, total: dataLines.length, truncated: dataLines.length > rows.length };
}

// Resolve the first CSV to preview for a local location: the file itself if it
// is a .csv, otherwise a `data.csv` or the first *.csv inside the folder.
async function findCsv(target: string): Promise<string | null> {
  const info = await stat(target);
  if (info.isFile()) {
    return target.toLowerCase().endsWith(".csv") ? target : null;
  }
  if (info.isDirectory()) {
    const entries = await readdir(target);
    const csvs = entries.filter((name) => name.toLowerCase().endsWith(".csv"));
    if (csvs.length === 0) {
      return null;
    }
    const preferred = csvs.find((name) => name.toLowerCase() === "data.csv") ?? csvs.sort()[0];
    return path.join(target, preferred);
  }
  return null;
}

export async function GET(request: Request) {
  try {
    const id = new URL(request.url).searchParams.get("id");
    if (!id) {
      return NextResponse.json({ kind: "none", message: "Missing dataset id." }, { status: 400 });
    }

    const entry = (await readCatalog()).find((d) => d.id === id);
    if (!entry) {
      return NextResponse.json({ kind: "none", message: `Dataset "${id}" not found.` }, { status: 404 });
    }
    if (!entry.location) {
      return NextResponse.json({ kind: "none", message: "This dataset has no location." });
    }
    if (isUrl(entry.location)) {
      return NextResponse.json({ kind: "url", url: entry.location });
    }

    const repoRoot = path.resolve(process.cwd(), "..");
    const target = path.resolve(repoRoot, entry.location);
    if (target !== repoRoot && !target.startsWith(repoRoot + path.sep)) {
      return NextResponse.json({ kind: "none", message: "Location is outside the repository." }, { status: 400 });
    }

    const csvPath = await findCsv(target).catch(() => null);
    if (!csvPath) {
      return NextResponse.json({
        kind: "none",
        message: `No CSV file found at ${entry.location}. This dataset loads its data programmatically.`
      });
    }

    const text = await readFile(csvPath, "utf8");
    const parsed = parseCsv(text);
    return NextResponse.json({ kind: "table", ...parsed });
  } catch (error) {
    return NextResponse.json(
      { kind: "none", message: `Could not read dataset: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}
