import init, { ModelLoader } from './pkg/valendra_candle_wasm.js';
import { headerLength, verifyRange } from './ranges.js';

const REVISION = '62a73ef2a5a216b0e0d0d5d2aac4e869847f3983';
const BASE = `https://huggingface.co/valendra/privacy-filter-mini/resolve/${REVISION}`;
const WEIGHT_SIZE = 2384403244;
const CHUNK = 8 * 1024 * 1024;
const CONFIG_SHA = '64a52e2ad92b6824d4b2daad520e3f318392bedca3e9887ab3a5881a1a784101';
const TOKENIZER_SHA = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4';
let detector = null;

async function fetchRange(start, end) {
  const response = await fetch(`${BASE}/model.safetensors`, {
    headers: { Range: `bytes=${start}-${end}` }, cache: 'no-store', credentials: 'omit',
  });
  if (response.status !== 206) throw new Error(`La descarga por bloques no está disponible (HTTP ${response.status}).`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  verifyRange(response.status, response.headers.get('content-range'), bytes.byteLength, start, end, WEIGHT_SIZE);
  return bytes;
}

async function fetchVerified(name, expected) {
  const response = await fetch(`${BASE}/${name}`, { credentials: 'omit' });
  if (!response.ok) throw new Error(`No se pudo descargar ${name} (HTTP ${response.status}).`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map(value => value.toString(16).padStart(2, '0')).join('');
  if (hash !== expected) throw new Error(`La comprobación de ${name} ha fallado.`);
  return bytes;
}

async function load() {
  if (detector) return;
  postMessage({ type: 'progress', stage: 'Preparando WebAssembly…', received: 0, total: WEIGHT_SIZE });
  await init();
  const [config, tokenizer, first] = await Promise.all([
    fetchVerified('config.json', CONFIG_SHA), fetchVerified('tokenizer.json', TOKENIZER_SHA), fetchRange(0, 7),
  ]);
  const length = headerLength(first, WEIGHT_SIZE);
  const header = await fetchRange(8, 7 + length);
  const prefix = new Uint8Array(8 + length);
  prefix.set(first);
  prefix.set(header, 8);
  const loader = new ModelLoader(prefix);
  const plan = JSON.parse(loader.plan());
  if (!plan.length || prefix.byteLength + plan[plan.length - 1].end !== WEIGHT_SIZE) {
    throw new Error('El índice de pesos no coincide con el tamaño publicado.');
  }
  let received = prefix.byteLength;
  for (const tensor of plan) {
    loader.begin_tensor(tensor.name);
    for (let position = tensor.start; position < tensor.end; position += CHUNK) {
      const end = Math.min(position + CHUNK, tensor.end) - 1;
      const bytes = await fetchRange(prefix.byteLength + position, prefix.byteLength + end);
      loader.push_chunk(bytes);
      received += bytes.byteLength;
      postMessage({ type: 'progress', stage: 'Descargando y verificando pesos…', received, total: WEIGHT_SIZE });
    }
    loader.end_tensor();
  }
  postMessage({ type: 'progress', stage: 'Inicializando el modelo…', received, total: WEIGHT_SIZE });
  detector = loader.finish(new TextDecoder('utf-8', { fatal: true }).decode(config), tokenizer);
  loader.free();
  postMessage({ type: 'ready' });
}

self.onmessage = async event => {
  try {
    if (event.data?.type === 'load') await load();
    else if (event.data?.type === 'detect') {
      if (!detector) throw new Error('Carga primero el modelo.');
      const result = JSON.parse(detector.detect(event.data.text));
      postMessage({ type: 'result', result });
    }
  } catch (error) {
    postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Error en el modelo.' });
  }
};
