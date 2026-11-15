// free-model-aggregator-01: one server, two client APIs, ten free-tier providers.
//
// The server reads the trial brief (provider directory, quality table, workload manifest), probes
// each provider's own model list, registers one catalog with the env, then answers the client's 20
// requests one at a time. Each request is routed to the highest-rated free model that meets its
// requirements; a refusal (402, 429, 5xx) moves the request to the next provider, never to a second
// model on the provider that just refused — budgets, outages and failure schedules are all
// provider-level, so a refusal says something about the provider, not the model.
//
// Routing order is by the quality table's rating, best first. At an exact rating tie the provider
// with the deeper free-tier budget goes first (the table's own call ranges: Cloudflare's 7-8 calls
// over Z.ai's 4-5 at the 1470 tie), because a spent provider is the one failure mode failover
// cannot route around twice.
//
// Before the first request arrives, the whole manifest is planned at once: candidate
// (request, provider) pairs are taken in weight-times-rating order and each request is given a
// planned primary provider, with a per-provider allowance of primaries estimated from the kit's
// published call ranges minus the failure schedule's one or two calls. Scarce capacity is thereby
// reserved for the requests that can only go there (vision lives on four providers, 200k context
// on five) and for the weight-2 requests, instead of being spent by whichever request arrives
// first. The plan only orders the attempts: at run time each request tries its planned primary,
// then the plain rating order, so a provider with budget left is never walled off.
//
// A provider that refuses three calls in a row is dead for the rest of the trial — a failure
// schedule holds at most two refusals, so three in a row can only be a spent budget or an outage —
// and is demoted behind every live provider (still tried before giving up, never first again).
//
// Everything is a function of the brief and the providers' answers: no wall clock, no randomness,
// so a replay of the trial walks the same routes and leaves the same state.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesOf,
  isRecord,
  joinBase,
  meets,
  probeAll,
  readBrief,
  type Brief,
  type Candidate,
  type Catalog,
  type Json,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  canonFromAnthropic,
  canonFromOpenAI,
  openAIError,
  openAIModelList,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropic,
  renderOpenAI,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CanonAnswer,
  type CanonRequest,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const UPSTREAM_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/** Provider preference at an exact rating tie: deeper free-tier budget first. */
const TIE_PREFERENCE = ['openrouter', 'cloudflare', 'zai', 'sambanova', 'gemini', 'ovh', 'groq', 'cerebras', 'kilol', 'pollinations'];

/**
 * How many primaries the plan grants each provider: the kit's published per-trial call range,
 * less the failure schedule's one or two refused calls, rounded down. Unknown providers get 6.
 */
const PRIMARY_ALLOWANCE: Record<string, number> = {
  openrouter: 9,
  cloudflare: 6,
  gemini: 5,
  zai: 3,
  sambanova: 3,
  groq: 8,
  cerebras: 4,
  ovh: 4,
  kilol: 10,
  pollinations: 10,
};
const DEFAULT_ALLOWANCE = 6;
/** Consecutive refusals after which a provider counts as dead (spent budget or outage). */
const DEAD_AFTER = 3;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

let brief: Brief;
let catalog: Catalog = { providers: [] };
let candidates: Candidate[] = [];
let manifestIndex = 0;
/** The planned primary provider per manifest request id, computed once from the whole manifest. */
let plan = new Map<string, string>();
/** Consecutive refusals per provider id, across the whole trial. */
const refusalStreak = new Map<string, number>();
/** The manifest id of the request being served, and its index (set by requirementsFor). */
let currentRequestId: string | null = null;

const tieRank = (id: string): number => {
  const at = TIE_PREFERENCE.indexOf(id);
  return at === -1 ? TIE_PREFERENCE.length : at;
};

async function load(): Promise<void> {
  brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  const probes = await probeAll(ENV_URL, brief.providers);
  for (const probe of probes) if (!probe.ok) log(`probe failed for ${probe.provider.id}: it is out of the catalog and gets no traffic`);
  catalog = buildCatalog(brief, probes);
  candidates = candidatesOf(brief, probes);
  plan = computePlan();
}

/**
 * Plan the whole manifest at once: (request, provider) pairs in weight-times-rating order each
 * claim their request's primary, subject to the provider's allowance of primaries. For planning,
 * an OpenAI-api request that requires JSON counts as JSON-mode, since its body will carry
 * response_format and an Anthropic-wire provider cannot echo it.
 */
function computePlan(): Map<string, string> {
  const out = new Map<string, string>();
  const allowanceLeft = new Map<string, number>();
  interface Pair {
    requestId: string;
    providerId: string;
    score: number;
    rating: number;
    index: number;
  }
  const pairs: Pair[] = [];
  brief.manifest.forEach((entry, index) => {
    const jsonMode = entry.api === 'openai' && entry.requires.json === true;
    for (const choice of providerOrder(entry.requires, jsonMode)) {
      pairs.push({ requestId: entry.id, providerId: choice.providerId, score: entry.weight * choice.best.rating, rating: choice.best.rating, index });
    }
  });
  pairs.sort((a, b) => b.score - a.score || b.rating - a.rating || a.index - b.index || tieRank(a.providerId) - tieRank(b.providerId));
  for (const pair of pairs) {
    if (out.has(pair.requestId)) continue;
    const left = allowanceLeft.get(pair.providerId) ?? PRIMARY_ALLOWANCE[pair.providerId] ?? DEFAULT_ALLOWANCE;
    if (left <= 0) continue;
    allowanceLeft.set(pair.providerId, left - 1);
    out.set(pair.requestId, pair.providerId);
  }
  return out;
}

/** The free models the server can actually serve, deduplicated by id, in brief provider order. */
function servedModels(): { id: string; provider: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string }[] = [];
  for (const c of candidates) {
    if (!c.free || seen.has(c.modelId)) continue;
    seen.add(c.modelId);
    out.push({ id: c.modelId, provider: c.provider.id });
  }
  return out;
}

interface ProviderChoice {
  providerId: string;
  best: Candidate;
}

/**
 * The providers that can serve this request, best first: each contributes its highest-rated
 * eligible model. A JSON-mode OpenAI request never routes to an Anthropic-wire provider, whose
 * wire cannot carry response_format and so cannot echo it back.
 */
function providerOrder(requires: Requirement, jsonMode: boolean): ProviderChoice[] {
  const byProvider = new Map<string, Candidate>();
  for (const c of candidates) {
    if (!c.free) continue;
    if (!meets(c, requires)) continue;
    if (jsonMode && c.provider.wire === 'anthropic') continue;
    const held = byProvider.get(c.provider.id);
    if (held === undefined || c.rating > held.rating) byProvider.set(c.provider.id, c);
  }
  return [...byProvider.values()]
    .map((best) => ({ providerId: best.provider.id, best }))
    .sort((a, b) => b.best.rating - a.best.rating || tieRank(a.providerId) - tieRank(b.providerId) || (a.providerId < b.providerId ? -1 : 1));
}

/** Requirements derived from the request body itself, when the manifest has no row for it. */
function requirementsFromCanon(canon: CanonRequest): Requirement {
  let vision = false;
  for (const turn of canon.turns) for (const block of turn.blocks) if (block.kind === 'image' || block.kind === 'imageUrl') vision = true;
  return { tools: canon.tools.length > 0, vision, json: canon.jsonMode };
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(canon: CanonRequest): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  currentRequestId = entry !== undefined ? entry.id : null;
  if (entry !== undefined) return entry.requires;
  return requirementsFromCanon(canon);
}

/** One upstream call, in the provider's own wire, never streamed. Null on any failure. */
async function callProvider(choice: Candidate, canon: CanonRequest): Promise<CanonAnswer | null> {
  const provider = choice.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(canon, choice.modelId);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = toAnthropic(canon, choice.modelId);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(choice.modelId)}:generateContent`;
    body = toGemini(canon, choice.modelId);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return null;
  }
  if (!res.ok) {
    // A refusal is spent either way; drain the body so the connection closes cleanly.
    await res.text().catch(() => '');
    return null;
  }
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    return null;
  }
  if (provider.wire === 'openai') return parseOpenAIAnswer(parsed);
  if (provider.wire === 'anthropic') return parseAnthropicAnswer(parsed);
  return parseGeminiAnswer(parsed);
}

/**
 * Serve one canonical request. The attempt order is the planned primary first, then the plain
 * rating order, with dead providers (three refusals in a row: a spent budget or an outage, never
 * a failure schedule, which holds at most two) demoted behind every live one. One call per
 * provider per round; a second round runs only when every provider refused — a scheduled refusal
 * may have passed by then, a spent budget will refuse again at no cost to the other columns.
 */
async function serve(canon: CanonRequest, requires: Requirement): Promise<{ answer: CanonAnswer; modelId: string } | null> {
  let order = providerOrder(requires, canon.jsonMode);
  const planned = currentRequestId !== null ? plan.get(currentRequestId) : undefined;
  if (planned !== undefined) {
    const at = order.findIndex((choice) => choice.providerId === planned);
    if (at > 0) order = [order[at], ...order.slice(0, at), ...order.slice(at + 1)];
  }
  const dead = order.filter((choice) => (refusalStreak.get(choice.providerId) ?? 0) >= DEAD_AFTER);
  if (dead.length > 0 && dead.length < order.length) {
    const deadIds = new Set(dead.map((choice) => choice.providerId));
    order = [...order.filter((choice) => !deadIds.has(choice.providerId)), ...dead];
  }
  for (let round = 0; round < 2; round++) {
    for (const choice of order) {
      const answer = await callProvider(choice.best, canon);
      if (answer !== null) {
        refusalStreak.set(choice.providerId, 0);
        return { answer, modelId: choice.best.modelId };
      }
      refusalStreak.set(choice.providerId, (refusalStreak.get(choice.providerId) ?? 0) + 1);
    }
    if (order.length === 0) break;
  }
  return null;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  send(res, status, 'application/json', JSON.stringify(body));
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendJson(res, 400, openAIError('the request body is not one JSON object', 'invalid_request_error', null));
  const canon = canonFromOpenAI(body);
  const served = await serve(canon, requirementsFor(canon));
  if (served === null) return sendJson(res, 503, openAIError('no free model on the directory could serve this request', 'no_free_model', null));
  const rendered = renderOpenAI(served.answer, served.modelId, body.stream === true);
  send(res, 200, rendered.contentType, rendered.body);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendJson(res, 400, anthropicError('invalid_request_error', 'the request body is not one JSON object'));
  const canon = canonFromAnthropic(body);
  const served = await serve(canon, requirementsFor(canon));
  if (served === null) return sendJson(res, 503, anthropicError('overloaded_error', 'no free model on the directory could serve this request'));
  const rendered = renderAnthropic(served.answer, served.modelId, body.stream === true);
  send(res, 200, rendered.contentType, rendered.body);
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return models(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return chat(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return messages(req, res);
  if (url.pathname === '/v1/messages') return sendJson(res, 404, anthropicError('not_found_error', `no route ${req.method ?? ''} ${url.pathname}`));
  return sendJson(res, 404, openAIError(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found', null));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment it says the workload is done. */
function poll(server: ReturnType<typeof createServer>): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendJson(res, 500, openAIError('the server failed to answer', 'internal_error', null));
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
