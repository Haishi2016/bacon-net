import { spawn } from "node:child_process";
import { readFile, readdir, stat, mkdir } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { load } from "js-yaml";

// Server-side only: train a real BACON model on the selected dataset's CSV by
// running portal/scripts/train_bacon.py (which uses the local `bacon` library,
// mirroring samples/hello-world/main.py). Progress and the learned tree are
// streamed to the client as Server-Sent Events.

export const dynamic = "force-dynamic";

// Valid aggregator families (bacon.baconNet._aggregator_registry). Guarded so
// only known values reach the Python process.
const ALLOWED_AGGREGATORS = new Set([
  "bool.min_max",
  "lsp.full_weight",
  "lsp.half_weight",
  "lsp.softmax",
  "gl.generic",
  "math.operator_set.logic",
  "math.operator_set.logic_identity",
  "math.operator_set.arith"
]);

type DatasetEntry = { id: string; location?: string };

function isUrl(value: string | undefined): value is string {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

async function readCatalog(): Promise<DatasetEntry[]> {
  const raw = await readFile(path.join(process.cwd(), "data", "datasets.yaml"), "utf8");
  const parsed = load(raw) as { datasets?: DatasetEntry[] } | null;
  return Array.isArray(parsed?.datasets) ? parsed.datasets : [];
}

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
  const url = new URL(request.url);
  const id = url.searchParams.get("id");
  const requestedAggregator = url.searchParams.get("aggregator") ?? "bool.min_max";
  const aggregator = ALLOWED_AGGREGATORS.has(requestedAggregator) ? requestedAggregator : "bool.min_max";
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: string, data: string) => {
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${data}\n\n`));
      };
      const fail = (message: string) => {
        send("error", JSON.stringify({ message }));
        controller.close();
      };

      try {
        if (!id) {
          return fail("Missing dataset id.");
        }
        const entry = (await readCatalog()).find((d) => d.id === id);
        if (!entry) {
          return fail(`Dataset "${id}" not found.`);
        }
        if (!entry.location || isUrl(entry.location)) {
          return fail("Training needs a local CSV dataset. This dataset has no local data file.");
        }

        const repoRoot = path.resolve(process.cwd(), "..");
        const target = path.resolve(repoRoot, entry.location);
        if (target !== repoRoot && !target.startsWith(repoRoot + path.sep)) {
          return fail("Dataset location is outside the repository.");
        }
        const csvPath = await findCsv(target).catch(() => null);
        if (!csvPath) {
          return fail(`No CSV file found at ${entry.location}. This dataset loads its data programmatically.`);
        }

        const scriptPath = path.join(process.cwd(), "scripts", "train_bacon.py");
        const pythonCmd = process.env.PYTHON ?? "python";

        // Stage the trained model (.pth) so "Save" can finalize it into
        // data/models/<id>.pth. bacon's own save_model writes the checkpoint.
        const stagingDir = path.join(process.cwd(), "data", "models", ".staging");
        await mkdir(stagingDir, { recursive: true });
        const stagingName = `${randomUUID()}.pth`;
        const stagingPath = path.join(stagingDir, stagingName);

        const child = spawn(
          pythonCmd,
          [scriptPath, "--csv", csvPath, "--repo", repoRoot, "--aggregator", aggregator, "--save-model", stagingPath],
          {
            cwd: repoRoot,
            env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1" }
          }
        );

        let buffer = "";
        const handleLine = (line: string) => {
          const idx = line.indexOf("::");
          if (idx === -1) {
            if (line.trim()) send("log", JSON.stringify({ message: line }));
            return;
          }
          const channel = line.slice(0, idx);
          const payload = line.slice(idx + 2);
          if (channel === "LOG") {
            send("log", JSON.stringify({ message: payload }));
          } else if (channel === "TREE") {
            send("tree", payload); // already JSON
          } else if (channel === "MODEL") {
            send("model", JSON.stringify({ staging: payload }));
          } else if (channel === "DONE") {
            send("done", payload);
          } else if (channel === "ERROR") {
            send("error", JSON.stringify({ message: payload }));
          } else {
            send("log", JSON.stringify({ message: line }));
          }
        };

        child.stdout.on("data", (chunk: Buffer) => {
          buffer += chunk.toString("utf8");
          let nl: number;
          while ((nl = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, nl).replace(/\r$/, "");
            buffer = buffer.slice(nl + 1);
            handleLine(line);
          }
        });

        child.stderr.on("data", (chunk: Buffer) => {
          const text = chunk.toString("utf8").trim();
          if (text) send("log", JSON.stringify({ message: text, stderr: true }));
        });

        child.on("error", (err) => {
          send("error", JSON.stringify({ message: `Failed to start Python: ${err.message}` }));
          controller.close();
        });

        child.on("close", (code) => {
          if (buffer.trim()) handleLine(buffer.trim());
          send("close", JSON.stringify({ code }));
          controller.close();
        });

        // Abort the Python process if the client disconnects.
        request.signal.addEventListener("abort", () => {
          child.kill();
          try {
            controller.close();
          } catch {
            /* already closed */
          }
        });
      } catch (error) {
        fail(error instanceof Error ? error.message : String(error));
      }
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive"
    }
  });
}
