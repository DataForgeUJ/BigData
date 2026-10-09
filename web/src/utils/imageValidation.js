const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const MAX_BYTES = 10 * 1024 * 1024;

function decodeImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error('This file could not be read as an image.'));
    image.src = url;
  });
}

export async function validateImage(file) {
  if (!file) throw new Error('Choose an image to continue.');
  if (!ALLOWED_TYPES.has(file.type)) throw new Error('Use a JPEG, PNG or WebP image.');
  if (file.size > MAX_BYTES) throw new Error('The image must be smaller than 10 MB.');

  const previewUrl = URL.createObjectURL(file);
  try {
    const dimensions = await decodeImage(previewUrl);
    if (!dimensions.width || !dimensions.height) throw new Error('The image has invalid dimensions.');
    return { file, previewUrl, ...dimensions };
  } catch (error) {
    URL.revokeObjectURL(previewUrl);
    throw error;
  }
}
