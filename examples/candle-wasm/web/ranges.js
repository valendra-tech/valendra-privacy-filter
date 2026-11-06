export function headerLength(bytes, totalSize) {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== 8 || !Number.isSafeInteger(totalSize)) {
    throw new Error('Invalid Safetensors prefix.');
  }
  const length = new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true);
  if (length === 0n || length > 67108864n || length + 8n > BigInt(totalSize)) {
    throw new Error('Invalid Safetensors header length.');
  }
  return Number(length);
}

export function verifyRange(status, contentRange, actualSize, start, end, totalSize) {
  if (![actualSize, start, end, totalSize].every(Number.isSafeInteger) || start < 0 || end < start || end >= totalSize ||
      status !== 206 || contentRange !== `bytes ${start}-${end}/${totalSize}` || actualSize !== end - start + 1) {
    throw new Error('The model host did not return the requested byte range.');
  }
}
