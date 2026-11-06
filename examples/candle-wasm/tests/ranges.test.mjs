import assert from 'node:assert/strict';
import test from 'node:test';
import { headerLength, verifyRange } from '../web/ranges.js';

test('reads a valid little-endian Safetensors header length', () => {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigUint64(0, 128n, true);
  assert.equal(headerLength(bytes, 1000), 128);
  assert.throws(() => headerLength(bytes.subarray(0, 7), 1000));
  assert.throws(() => headerLength(bytes, 100));
});

test('refuses full-file, incorrect and truncated range responses', () => {
  assert.doesNotThrow(() => verifyRange(206, 'bytes 8-15/100', 8, 8, 15, 100));
  assert.throws(() => verifyRange(200, 'bytes 8-15/100', 8, 8, 15, 100));
  assert.throws(() => verifyRange(206, 'bytes 0-7/100', 8, 8, 15, 100));
  assert.throws(() => verifyRange(206, 'bytes 8-15/100', 7, 8, 15, 100));
});
