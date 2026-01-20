import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, StyleSheet, LayoutChangeEvent } from 'react-native';

type WaveformMode = 'rolling' | 'full';
type CursorMode = 'pinned' | 'follow';

type Props = {
  peaks: number[];          // fixed-grid samples, each bin represents barDurationMs
  progressMs: number;
  durationMs?: number;

  barDurationMs?: number;  // default 50
  windowMs?: number;       // default 30s
  mode?: WaveformMode;

  cursorMode?: CursorMode;
  showCursor?: boolean;

  tagTimestamps?: number[];
  tagWidthMs?: number;
  showTagMarkers?: boolean;

  height?: number;
  minBarPx?: number;
  amplitudeBoost?: number;

  autoScale?: boolean;
  autoScaleIntervalMs?: number;
  autoScalePercentile?: number;
};

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

export default function Waveform({
  peaks,
  progressMs,
  durationMs,

  barDurationMs = 50,
  windowMs = 30_000,
  mode = 'rolling',

  cursorMode = 'pinned',
  showCursor = true,

  tagTimestamps = [],
  tagWidthMs = 500,
  showTagMarkers = true,

  height = 64,
  minBarPx = 2,
  amplitudeBoost = 1.6,

  autoScale = true,
  autoScaleIntervalMs = 500,
  autoScalePercentile = 0.98,
}: Props) {
  const [width, setWidth] = useState(1);

  const [yScale, setYScale] = useState(1);
  const lastScaleUpdateRef = useRef(0);

  const onLayout = (e: LayoutChangeEvent) => {
    const w = e.nativeEvent.layout.width;
    if (w > 0) setWidth(w);
  };

  const windowBars = Math.max(1, Math.round(windowMs / barDurationMs)); // 600 for 30s@50ms

  // Visible range in ms
  const { visibleStartMs, visibleEndMs } = useMemo(() => {
    if (mode === 'full') {
      const totalMs = Math.max(durationMs ?? 0, peaks.length * barDurationMs);
      return { visibleStartMs: 0, visibleEndMs: totalMs };
    }
    const end = Math.max(0, progressMs);
    const start = Math.max(0, end - windowMs);
    return { visibleStartMs: start, visibleEndMs: start + windowMs };
  }, [mode, durationMs, peaks.length, barDurationMs, progressMs, windowMs]);

  // Extract fixed-grid window for rolling mode (always windowBars long, padded)
  const baseBars = useMemo(() => {
    if (mode === 'full') return peaks.map(clamp01);

    const startIndex = Math.floor(visibleStartMs / barDurationMs);
    const bars = new Array<number>(windowBars).fill(0);

    for (let i = 0; i < windowBars; i++) {
      const srcIdx = startIndex + i;
      bars[i] = srcIdx >= 0 && srcIdx < peaks.length ? clamp01(peaks[srcIdx]) : 0;
    }
    return bars;
  }, [mode, peaks, visibleStartMs, barDurationMs, windowBars]);

  // Downsample to pixel bars
  const { barsToRender, barPx, barCount } = useMemo(() => {
    const px = Math.max(1, Math.floor(minBarPx));
    const count = Math.max(1, Math.floor(width / px));

    const bucketSize = Math.max(1, Math.ceil(baseBars.length / count));
    const out: number[] = [];

    for (let i = 0; i < baseBars.length; i += bucketSize) {
      let m = 0;
      for (let j = 0; j < bucketSize && i + j < baseBars.length; j++) {
        m = Math.max(m, baseBars[i + j]);
      }
      out.push(m);
    }

    // exact count via trim/pad
    let trimmed = out.length > count ? out.slice(0, count) : out;
    if (trimmed.length < count) trimmed = [...trimmed, ...new Array(count - trimmed.length).fill(0)];

    return { barsToRender: trimmed, barPx: px, barCount: trimmed.length };
  }, [baseBars, width, minBarPx]);

  // Autoscale based on percentile of non-zero bars (post-downsample)
  useEffect(() => {
    if (!autoScale) return;
    const now = Date.now();
    if (now - lastScaleUpdateRef.current < autoScaleIntervalMs) return;
    lastScaleUpdateRef.current = now;

    const vals = barsToRender.filter((v) => v > 0).slice();
    if (vals.length === 0) {
      setYScale(1);
      return;
    }
    vals.sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(vals.length - 1, Math.floor(vals.length * autoScalePercentile)));
    const p = vals[idx];

    // allow stronger boosting during quiet periods
    const target = Math.max(0.04, Math.min(1, p));
    setYScale(target);
  }, [autoScale, autoScaleIntervalMs, autoScalePercentile, barsToRender]);

  // Cursor position
  const cursorX = useMemo(() => {
    if (!showCursor) return -100;

    if (mode === 'full') {
      const totalMs = Math.max(1, durationMs ?? peaks.length * barDurationMs);
      return clamp01(progressMs / totalMs) * width;
    }

    if (cursorMode === 'pinned') return width - 1;

    const within = clamp01((progressMs - visibleStartMs) / windowMs);
    return within * width;
  }, [showCursor, mode, cursorMode, width, progressMs, durationMs, peaks.length, barDurationMs, visibleStartMs, windowMs]);

  // Tag overlay rectangles mapped to the same axis
  const tagRects = useMemo(() => {
    if (!showTagMarkers || tagTimestamps.length === 0) return [];

    if (mode === 'full') {
      const totalMs = Math.max(1, durationMs ?? peaks.length * barDurationMs);
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
    peaks.length,
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
        {barsToRender.map((p, i) => {
          const scaled = clamp01((p * amplitudeBoost) / Math.max(1e-6, yScale));
          const h = Math.max(1, scaled * height);
          return (
            <View
              key={i}
              style={{
                width: barPx,
                height: h,
                backgroundColor: '#CFCFCF',
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
              style={[
                styles.tagMarker,
                { left: r.left, width: r.width, height },
              ]}
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
    alignItems: 'flex-end',
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
    opacity: 0.85,
  },
  cursor: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: 'red',
  },
});
