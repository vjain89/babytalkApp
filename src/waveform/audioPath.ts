import RNFS from 'react-native-fs';

/** Resolve a DB filename or URI to a file:// path that exists on disk. */
export async function resolveAudioUri(filePath: string): Promise<string | null> {
  const strip = (p: string) => p.replace(/^file:\/\//, '');
  const withPrivateVariants = (abs: string): string[] => {
    const out = [abs];
    if (abs.startsWith('/private/')) out.push(abs.replace(/^\/private/, ''));
    else if (abs.startsWith('/var/')) out.push(`/private${abs}`);
    return out;
  };

  if (filePath.startsWith('file://') || filePath.startsWith('/')) {
    let abs = strip(filePath);
    try {
      abs = decodeURIComponent(abs);
    } catch {
      // keep abs
    }
    for (const c of withPrivateVariants(abs)) {
      if (await RNFS.exists(c)) {
        return `file://${c}`;
      }
    }
  }

  const name = filePath.split('/').pop() || filePath;
  const candidates = [
    `${RNFS.DocumentDirectoryPath}/${name}`,
    `${RNFS.CachesDirectoryPath}/${name}`,
    `/private${RNFS.DocumentDirectoryPath}/${name}`,
    `/private${RNFS.CachesDirectoryPath}/${name}`,
  ];
  for (const c of candidates) {
    if (await RNFS.exists(c)) return `file://${c}`;
  }
  return null;
}
