import RNFS from 'react-native-fs';

/** Resolve a DB filename or URI to a file:// path that exists on disk. */
export async function resolveAudioUri(filePath: string): Promise<string | null> {
  const strip = (p: string) => p.replace(/^file:\/\//, '');

  if (filePath.startsWith('file://') || filePath.startsWith('/')) {
    const abs = strip(filePath);
    if (await RNFS.exists(abs)) {
      return filePath.startsWith('file://') ? filePath : `file://${abs}`;
    }
  }

  const name = filePath.split('/').pop() || filePath;
  const candidates = [
    `${RNFS.CachesDirectoryPath}/${name}`,
    `${RNFS.DocumentDirectoryPath}/${name}`,
  ];
  for (const c of candidates) {
    if (await RNFS.exists(c)) return `file://${c}`;
  }
  return null;
}
