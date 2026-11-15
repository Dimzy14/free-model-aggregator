// The trial brief: three JSON documents inside one file, found by their `schema` member.
//
//   providers/v1  the directory: each provider's id, wire, auth scheme, key env var, key and base path
//   ratings/v1    the quality table: per model its rating, context length, capabilities and free flag,
//                 plus each workload request's best attainable rating
//   workload/v1   the manifest: the requests in arrival order, with api, stream, weight, requirements
//
// The catalog is probed from the providers themselves (each speaks its own wire's model-list route)
// and joined with the quality table: the table is the authority on whether a model is free and what
// it can do; a listing fills in only where the table has no row for the model.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

export interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

/** One servable model: a probed listing joined with its quality-table row where one exists. */
export interface Candidate {
  provider: ProviderEntry;
  modelId: string;
  rating: number;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** A base path (or absolute URL) joined to the env's origin. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key, in its own scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON object: keep scanning.
    }
  }
  return out;
}

/** The index of the `}` that closes the object opening at `start`, or -1. String contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = str(v.path) !== '' ? str(v.path) : str(v.base_url) !== '' ? str(v.base_url) : str(v.base_path) !== '' ? str(v.base_path) : str(v.base);
  return { id, name: str(v.name) !== '' ? str(v.name) : id, wire, auth, base, key_env: str(v.key_env), key: str(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: typeof v.weight === 'number' && Number.isFinite(v.weight) ? v.weight : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** Read and parse the brief at `path`. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read the brief: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directoryDoc === undefined) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings = ratingsDoc === undefined ? [] : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
    const r = ratingOf(v);
    return r === null ? [] : [r];
  });
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best as unknown[]) {
      if (isRecord(v) && str(v.id) !== '') best.set(str(v.id), numOrNull(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest = workloadDoc === undefined ? [] : (workloadDoc.manifest as unknown[]).flatMap((v) => {
    const w = workloadOf(v);
    return w === null ? [] : [w];
  });
  return { providers, ratings, best, manifest, error: null };
}

/** One model as a provider's listing spelled it, whichever wire the listing used. */
function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: str(v.display_name) !== '' ? str(v.display_name) : str(v.displayName) !== '' ? str(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Probe one provider's model-list route in its own wire. Any failure leaves the provider unprobed. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((v) => {
      const m = listedModel(provider.wire, v);
      return m === null ? [] : [m];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/** Probe every provider, with one retry each: a provider missed here is lost for the whole trial. */
export async function probeAll(envUrl: string, providers: ProviderEntry[]): Promise<Probe[]> {
  const first = await Promise.all(providers.map((p) => probeProvider(envUrl, p)));
  return Promise.all(
    first.map(async (probe) => {
      if (probe.ok) return probe;
      const retry = await probeProvider(envUrl, probe.provider);
      return retry.ok ? retry : probe;
    }),
  );
}

/** Join probes with the quality table into the servable candidate set. */
export function candidatesOf(brief: Brief, probes: Probe[]): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = brief.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      out.push({
        provider: probe.provider,
        modelId: model.id,
        rating: row?.rating_milli ?? 0,
        free: row !== null ? row.free : (model.free ?? false),
        context_length: row?.context_length ?? model.context_length ?? null,
        tools: row !== null ? row.tools : (model.tools ?? false),
        vision: row !== null ? row.vision : (model.vision ?? false),
        json: row !== null ? row.json : (model.json ?? false),
      });
    }
  }
  return out;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/**
 * The catalog document: every provider that answered its probe, every model it listed, with the
 * quality table's word on free and context length. Its base_url is the directory's own path, never
 * joined to the env's origin, so the document is identical in a live trial and its replay.
 */
export function buildCatalog(brief: Brief, probes: Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = brief.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      models.push({ id: model.id, free: row !== null ? row.free : (model.free ?? false), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

/** Whether a candidate meets one request's requirements. An unknown capability never meets one. */
export function meets(c: Candidate, requires: Requirement): boolean {
  if (requires.tools === true && !c.tools) return false;
  if (requires.vision === true && !c.vision) return false;
  if (requires.json === true && !c.json) return false;
  if (requires.min_context !== undefined && (c.context_length === null || c.context_length < requires.min_context)) return false;
  return true;
}
