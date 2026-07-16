import React, { useMemo, useState } from 'react';
import { View, StyleSheet, LayoutChangeEvent } from 'react-native';
import {
  BAR_GAP_PX,
  BAR_MS,
  MIN_BAR_PX,
  TAG_MARKER_MS,
  WAVEFORM_HEIGHT,
} from '../waveform/config';
import { dbToHeight01, type DbRange } from '../waveform/scale';

type Props = {
  /** Dense peak dB array: one entry per barDurationMs in the visible window. */
  peaksDb: number[];
  progressMs: number;
  durationMs?: number;

  barDurationMs?: number;
  /** Visible time window length (rolling modes). */
  windowMs: number;
  mode?: 'rolling' | 'full';
  cursorMode?: 'pinned' | 'follow';
  showCursor?: boolean;

  tagTimestamps?: number[];
  tagWidthMs?: number;
  showTagMarkers?: boolean;

  /** Fixed Y mapping for this session (no live autoscale). */
  dbRange: DbRange;

  height?: number;
  minBarPx?: number;
};

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/**
 * Display-only waveform. Does not capture or store audio.
 * Expects peaksDb already densified for the visible window (or full clip).
 * Renders a level envelope (metering), not a PCM oscilloscope.
 */
export default function Waveform({
  peaksDb,
  progressMs,
  durationMs,
  barDurationMs = BAR_MS,
  windowMs,
  mode = 'rolling',
  cursorMode = 'pinned',
  showCursor = true,
  tagTimestamps = [],
  tagWidthMs = TAG_MARKER_MS,
  showTagMarkers = true,
  dbRange,
  height = WAVEFORM_HEIGHT,
  minBarPx = MIN_BAR_PX,
}: Props) {
  const [width, setWidth] = useState(1);

  const onLayout = (e: LayoutChangeEvent) => {
    const w = e.nativeEvent.layout.width;
    if (w > 0) setWidth(w);
  };

  // For rolling mode, peaksDb is already the densified trailing window ending at progressMs.
  // visibleStart may be negative early in a session (silence padding on the left).
  const { visibleStartMs, visibleEndMs } = useMemo(() => {
    if (mode === 'full') {
      const totalMs = Math.max(durationMs ?? 0, peaksDb.length * barDurationMs);
      return { visibleStartMs: 0, visibleEndMs: totalMs };
    }
    const end = progressMs;
    return { visibleStartMs: end - windowMs, visibleEndMs: end };
  }, [mode, durationMs, peaksDb.length, barDurationMs, progressMs, windowMs]);

  // One pixel bar per source bar when they fit; only downsample if screen is too narrow.
  const { barsToRender, barPx } = useMemo(() => {
    const minPx = Math.max(1, Math.floor(minBarPx));
    const maxCount = Math.max(1, Math.floor(width / minPx));
    const source = peaksDb;

    if (source.length === 0) {
      return { barsToRender: [] as number[], barPx: minPx };
    }

    if (source.length <= maxCount) {
      return {
        barsToRender: source.map((db) => dbToHeight01(db, dbRange)),
        barPx: width / source.length,
      };
    }

    const bucketSize = Math.ceil(source.length / maxCount);
    const out: number[] = [];
    for (let i = 0; i < source.length; i += bucketSize) {
      let m = -160;
      for (let j = 0; j < bucketSize && i + j < source.length; j++) {
        m = Math.max(m, source[i + j]);
      }
      out.push(dbToHeight01(m, dbRange));
    }
    return { barsToRender: out, barPx: width / Math.max(1, out.length) };
  }, [peaksDb, width, minBarPx, dbRange]);

  const drawnBarWidth = Math.max(1, barPx - BAR_GAP_PX);

  const cursorX = useMemo(() => {
    if (!showCursor) return -100;
    if (mode === 'full') {
      const totalMs = Math.max(1, durationMs ?? peaksDb.length * barDurationMs);
      return clamp01(progressMs / totalMs) * width;
    }
    if (cursorMode === 'pinned') return width - 1;
    return clamp01((progressMs - visibleStartMs) / windowMs) * width;
  }, [
    showCursor,
    mode,
    cursorMode,
    width,
    progressMs,
    durationMs,
    peaksDb.length,
    barDurationMs,
    visibleStartMs,
    windowMs,
  ]);

  const tagRects = useMemo(() => {
    if (!showTagMarkers || tagTimestamps.length === 0) return [];

    if (mode === 'full') {
      const totalMs = Math.max(1, durationMs ?? peaksDb.length * barDurationMs);
      const msPerPixel = totalMs / Math.max(1, width);
      const wPx = Math.max(2, Math.round(tagWidthMs / msPerPixel));
      return tagTimestamps.map((ts) => {
        const x = (ts / totalMs) * width;
        return { left: Math.max(0, Math.min(width, x - wPx / 2)), width: wPx };
      });
    }

    const msPerPixel = windowMs / Math.max(1, width);
    const wPx = Math.max(2, Math.round(tagWidthMs / msPerPixel));
    const rects: { left: number; width: number }[] = [];
    for (const ts of tagTimestamps) {
      if (ts < visibleStartMs || ts > visibleEndMs) continue;
      const x = ((ts - visibleStartMs) / windowMs) * width;
      rects.push({ left: Math.max(0, Math.min(width, x - wPx / 2)), width: wPx });
    }
    return rects;
  }, [
    showTagMarkers,
    tagTimestamps,
    mode,
    durationMs,
    peaksDb.length,
    barDurationMs,
    visibleStartMs,
    visibleEndMs,
    windowMs,
    width,
    tagWidthMs,
  ]);

  return (
    <View style={[styles.container, { height }]} onLayout={onLayout}>
      <View style={styles.barsRow}>
        {barsToRender.map((h01, i) => {
          // Silence: hairline only. Active audio: proportional height from center.
          const h = h01 <= 0 ? 1 : Math.max(2, h01 * height);
          return (
            <View
              key={i}
              style={{
                width: drawnBarWidth,
                marginRight: BAR_GAP_PX,
                height: h,
                backgroundColor: h01 <= 0 ? '#E8E8E8' : '#8A8A8A',
                borderRadius: 1,
              }}
            />
          );
        })}
      </View>

      {showTagMarkers && (
        <View pointerEvents="none" style={styles.overlay}>
          {tagRects.map((r, idx) => (
            <View
              key={idx}
              style={[styles.tagMarker, { left: r.left, width: r.width, height }]}
            />
          ))}
        </View>
      )}

      {showCursor && <View style={[styles.cursor, { left: cursorX }]} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    backgroundColor: '#F4F4F4',
    borderRadius: 6,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#E0E0E0',
    position: 'relative',
  },
  barsRow: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    top: 0,
    flexDirection: 'row',
    alignItems: 'center',
  },
  overlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
  },
  tagMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    backgroundColor: '#2F80ED',
    opacity: 0.55,
  },
  cursor: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: 'red',
  },
});
