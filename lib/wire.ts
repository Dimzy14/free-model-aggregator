// Wire translation between the two client APIs (OpenAI Chat Completions, Anthropic Messages) and the
// three provider wires (OpenAI, Anthropic, Gemini).
//
// A provider answers with an echo of what it parsed from the request it received, and the client
// compares that echo with what it sent: system text, the last user text, tool names, max_tokens, stop
// sequences, the image hash and the JSON-mode flag. Every translation here is built to keep those
// fields identical under each wire's own parsing rules:
//
//   - system text is carried whole (wires join system parts with "\n", so parts are pre-joined)
//   - a turn's text blocks are pre-joined with "\n" into one string wherever a wire would do the same
//   - an image travels as base64 bytes or as its URL, the two forms every wire hashes the same way
//   - max_tokens and stops are copied only when the client sent them: an invented value would echo
//     back a number the client never sent
//   - JSON mode exists on the OpenAI wire (response_format) and the Gemini wire (responseMimeType);
//     the Anthropic wire cannot carry it, and the router never sends it there
//
// Upstream calls are never streamed: the provider's answer is one JSON document however it was
// asked, and the client's stream, when it asked for one, is synthesized from the whole answer.

import { isRecord, type Json } from './brief.ts';

export interface ImageBlock {
  kind: 'image';
  mediaType: string;
  data: string;
}
export interface ImageUrlBlock {
  kind: 'imageUrl';
  url: string;
}
export interface TextBlock {
  kind: 'text';
  text: string;
}
export type Block = TextBlock | ImageBlock | ImageUrlBlock;

export interface Turn {
  role: 'user' | 'assistant';
  blocks: Block[];
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export interface CanonRequest {
  system: string;
  turns: Turn[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
}

export interface CanonAnswer {
  text: string;
  usageIn: number;
  usageOut: number;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** The text of a content value: a string as it stands, or its text blocks joined with "\n". */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Split a data URL into its media type and base64 payload; null when it is not one. */
function dataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]*)(?:;[^,]*)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** The blocks of one OpenAI-wire message's content. */
function openAIBlocks(content: unknown): Block[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Block[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') blocks.push({ kind: 'text', text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const parsed = dataUrl(block.image_url.url);
      if (parsed !== null) blocks.push({ kind: 'image', mediaType: parsed.mediaType, data: parsed.data });
      else blocks.push({ kind: 'imageUrl', url: block.image_url.url });
    }
  }
  return blocks;
}

/** The blocks of one Anthropic-wire message's content. */
function anthropicBlocks(content: unknown): Block[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Block[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') blocks.push({ kind: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        blocks.push({ kind: 'image', mediaType: str(block.source.media_type) !== '' ? str(block.source.media_type) : 'image/png', data: block.source.data });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        blocks.push({ kind: 'imageUrl', url: block.source.url });
      }
    }
  }
  return blocks;
}

/** A client OpenAI request, canonicalized. */
export function canonFromOpenAI(body: Json): CanonRequest {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (message.role === 'system') {
      const text = contentText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user' || message.role === 'assistant') {
      turns.push({ role: message.role, blocks: openAIBlocks(message.content) });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
        tools.push({ name: tool.function.name, description: str(tool.function.description), parameters: isRecord(tool.function.parameters) ? tool.function.parameters : {} });
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** A client Anthropic request, canonicalized. */
export function canonFromAnthropic(body: Json): CanonRequest {
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (message.role === 'user' || message.role === 'assistant') turns.push({ role: message.role, blocks: anthropicBlocks(message.content) });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') {
        tools.push({ name: tool.name, description: str(tool.description), parameters: isRecord(tool.input_schema) ? tool.input_schema : {} });
      }
    }
  }
  return {
    system: contentText(body.system),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: false,
  };
}

/** A turn's content for the OpenAI wire: one string when it is all text, blocks when it holds an image. */
function openAIContent(blocks: Block[]): unknown {
  if (blocks.every((b) => b.kind === 'text')) return blocks.map((b) => (b as TextBlock).text).join('\n');
  return blocks.map((b) => {
    if (b.kind === 'text') return { type: 'text', text: b.text };
    if (b.kind === 'image') return { type: 'image_url', image_url: { url: `data:${b.mediaType};base64,${b.data}` } };
    return { type: 'image_url', image_url: { url: b.url } };
  });
}

/** A turn's content for the Anthropic wire. */
function anthropicContent(blocks: Block[]): unknown {
  if (blocks.every((b) => b.kind === 'text')) return blocks.map((b) => (b as TextBlock).text).join('\n');
  return blocks.map((b) => {
    if (b.kind === 'text') return { type: 'text', text: b.text };
    if (b.kind === 'image') return { type: 'image', source: { type: 'base64', media_type: b.mediaType, data: b.data } };
    return { type: 'image', source: { type: 'url', url: b.url } };
  });
}

/** A turn's parts for the Gemini wire. */
function geminiParts(blocks: Block[]): Json[] {
  return blocks.map((b) => {
    if (b.kind === 'text') return { text: b.text };
    if (b.kind === 'image') return { inlineData: { mimeType: b.mediaType, data: b.data } };
    return { fileData: { fileUri: b.url } };
  });
}

/** Build the upstream request body for an OpenAI-wire provider. */
export function toOpenAI(canon: CanonRequest, modelId: string): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const turn of canon.turns) messages.push({ role: turn.role, content: openAIContent(turn.blocks) });
  const body: Json = { model: modelId, messages };
  if (canon.tools.length > 0) body.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

/** Build the upstream request body for an Anthropic-wire provider. Never carries JSON mode. */
export function toAnthropic(canon: CanonRequest, modelId: string): Json {
  const body: Json = { model: modelId, messages: canon.turns.map((turn) => ({ role: turn.role, content: anthropicContent(turn.blocks) })) };
  if (canon.system !== '') body.system = canon.system;
  if (canon.tools.length > 0) body.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop_sequences = canon.stops;
  return body;
}

/** Build the upstream request body for a Gemini-wire provider. */
export function toGemini(canon: CanonRequest, modelId: string): Json {
  const body: Json = { contents: canon.turns.map((turn) => ({ role: turn.role === 'assistant' ? 'model' : 'user', parts: geminiParts(turn.blocks) })) };
  if (canon.system !== '') body.systemInstruction = { parts: [{ text: canon.system }] };
  if (canon.tools.length > 0) {
    body.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/** Parse an OpenAI-wire provider's JSON answer. */
export function parseOpenAIAnswer(body: unknown): CanonAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (first === null || !isRecord(first.message)) return null;
  const content = first.message.content;
  const text = typeof content === 'string' ? content : contentText(content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, usageIn: numOrNull(usage.prompt_tokens) ?? 0, usageOut: numOrNull(usage.completion_tokens) ?? 0 };
}

/** Parse an Anthropic-wire provider's JSON answer. */
export function parseAnthropicAnswer(body: unknown): CanonAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.content)) return null;
  const parts: string[] = [];
  for (const block of body.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: parts.join(''), usageIn: numOrNull(usage.input_tokens) ?? 0, usageOut: numOrNull(usage.output_tokens) ?? 0 };
}

/** Parse a Gemini-wire provider's JSON answer. */
export function parseGeminiAnswer(body: unknown): CanonAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const first = body.candidates[0];
  if (!isRecord(first) || !isRecord(first.content) || !Array.isArray(first.content.parts)) return null;
  const parts: string[] = [];
  for (const part of first.content.parts) {
    if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text: parts.join(''), usageIn: numOrNull(usage.promptTokenCount) ?? 0, usageOut: numOrNull(usage.candidatesTokenCount) ?? 0 };
}

const FIXED_CREATED = 1767225600;

/** Split a text into stream pieces of at most `size` characters. */
function pieces(text: string, size = 64): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [''] : out;
}

/** Render the answer on the OpenAI wire, as JSON or as a synthesized event stream. */
export function renderOpenAI(answer: CanonAnswer, modelId: string, stream: boolean): { contentType: string; body: string } {
  const usage = { prompt_tokens: answer.usageIn, completion_tokens: answer.usageOut, total_tokens: answer.usageIn + answer.usageOut };
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model: modelId,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`];
  for (const piece of pieces(answer.text)) frames.push(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  frames.push(`data: ${JSON.stringify(chunk({}, 'stop'))}\n\n`);
  frames.push(`data: ${JSON.stringify({ id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId, choices: [], usage })}\n\n`);
  frames.push('data: [DONE]\n\n');
  return { contentType: 'text/event-stream', body: frames.join('') };
}

/** Render the answer on the Anthropic wire, as JSON or as a synthesized event stream. */
export function renderAnthropic(answer: CanonAnswer, modelId: string, stream: boolean): { contentType: string; body: string } {
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: answer.usageIn, output_tokens: answer.usageOut },
      }),
    };
  }
  const frame = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregator', type: 'message', role: 'assistant', model: modelId, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usageIn, output_tokens: 0 } },
    }),
    frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of pieces(answer.text)) {
    frames.push(frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(frame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(frame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.usageOut } }));
  frames.push(frame('message_stop', { type: 'message_stop' }));
  return { contentType: 'text/event-stream', body: frames.join('') };
}

/** An error body in the OpenAI wire's envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** An error body in the Anthropic wire's envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** The model list in the OpenAI shape: a `data` array, one row per free model id. */
export function openAIModelList(models: { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider, free: true })),
  };
}

/** The model list in the Anthropic shape: also a `data` array, as that wire spells it. */
export function anthropicModelList(models: { id: string; provider: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z' })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}
