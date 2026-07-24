import {
  AVEncoderAudioQualityIOSType,
  AVEncodingOption,
  AVLinearPCMBitDepthKeyIOSType,
  type AudioSet,
} from 'react-native-audio-recorder-player';

/** Locked capture params: ALAC master, 44.1 kHz, 16-bit, mono. */
export const CAPTURE_SAMPLE_RATE = 44100;
export const CAPTURE_CHANNELS = 1;
export const CAPTURE_BIT_DEPTH = 16;
export const CAPTURE_CODEC = 'alac' as const;
export const CAPTURE_CONTAINER_EXT = 'm4a';

export const IOS_ALAC_AUDIO_SET: AudioSet = {
  AVSampleRateKeyIOS: CAPTURE_SAMPLE_RATE,
  AVFormatIDKeyIOS: AVEncodingOption.alac,
  AVNumberOfChannelsKeyIOS: CAPTURE_CHANNELS,
  AVLinearPCMBitDepthKeyIOS: AVLinearPCMBitDepthKeyIOSType.bit16,
  AVEncoderAudioQualityKeyIOS: AVEncoderAudioQualityIOSType.max,
};

/** Unique ALAC master filename (avoids overwriting Caches/recording.m4a). */
export function newRecordingFilename(): string {
  return `recording_${Date.now()}.${CAPTURE_CONTAINER_EXT}`;
}

/** Default session display name: YY_MM_DD__HH:MM:SS (local time). */
export function formatSessionStampName(date: Date = new Date()): string {
  const yy = String(date.getFullYear()).slice(-2);
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  const hh = String(date.getHours()).padStart(2, '0');
  const mi = String(date.getMinutes()).padStart(2, '0');
  const ss = String(date.getSeconds()).padStart(2, '0');
  return `${yy}_${mm}_${dd}__${hh}:${mi}:${ss}`;
}
