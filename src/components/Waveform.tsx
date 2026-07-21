import React, { useMemo, useRef, useState } from 'react';
import { View, StyleSheet, LayoutChangeEvent, PanResponder } from 'react-native';
import {
  BAR_GAP_PX,
  BAR_MS,
  MIN_BAR_PX,
  TAG_MARKER_MS,
  WAVEFORM_HEIGHT,
} from '../waveform/config';
import { dbToHeight01, type DbRange } from '../waveform/scale';
import type { BipolarColumn } from '../waveform/types';

type Props = {
  /** Dense peak dB array: one entry per barDurationMs in the visible window. */
  peaksDb?: number[];
  /** A1 bipolar columns (min/max amp already densified for the window). */
  bipolarColumns?: BipolarColumn[];
  /** Divisor for bipolar amp → height (from computeBipolarAmpScale). */
  bipolarScale?: number;
  progressMs: number;
  /** Preview position while scrubbing (cursor only). */
  scrubPreviewMs?: number | null;
  /** Rolling view: time at the right edge of the visible window. */
  rollingWindowEndMs?: number;
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

  /** Fixed Y mapping for this session (no live autoscale). Envelope mode only. */
  dbRange: DbRange;

  height?: number;
  minBarPx?: number;

  /** Tap/drag to seek (playback). Maps touch X → time via onSeek. */
  seekable?: boolean;
  onSeek?: (ms: number) => void;
  /** Fires during scrub with preview ms, or null when scrub ends. */
  onScrubChange?: (ms: number | null) => void;
};

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/**
 * Display-only waveform. Does not capture or store audio.
 * Envelope mode: peaksDb densified for the visible window.
 * Bipolar mode (A1): bipolarColumns with min/max amp around centerline.
 */
export default function Waveform({
  peaksDb = [],
  bipolarColumns,
  bipolarScale = 1,
  progressMs,
  scrubPreviewMs = null,
  rollingWindowEndMs,
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
  seekable = false,
  onSeek,
  onScrubChange,
}: Props) {
  const [width, setWidth] = useState(1);
  const [scrubMs, setScrubMs] = useState<number | null>(null);
  const widthRef = useRef(1);
  const mapXToMsRef = useRef<(x: number) => number>(() => 0);
  const bipolar = Boolean(bipolarColumns && bipolarColumns.length > 0);
  const columnCount = bipolar ? bipolarColumns!.length : peaksDb.length;

  const onLayout = (e: LayoutChangeEvent) => {
    const w = e.nativeEvent.layout.width;
    if (w > 0) {
      setWidth(w);
      widthRef.current = w;
    }
  };

  const displayProgressMs = scrubPreviewMs ?? scrubMs ?? progressMs;
  const rollingEndMs = rollingWindowEndMs ?? progressMs;

  const { visibleStartMs, visibleEndMs } = useMemo(() => {
    if (mode === 'full') {
      const totalMs = Math.max(durationMs ?? 0, columnCount * barDurationMs);
      return { visibleStartMs: 0, visibleEndMs: totalMs };
    }
    return { visibleStartMs: rollingEndMs - windowMs, visibleEndMs: rollingEndMs };
  }, [mode, durationMs, columnCount, barDurationMs, rollingEndMs, windowMs]);

  const totalMsFull = useMemo(
    () => Math.max(1, durationMs ?? columnCount * barDurationMs),
    [durationMs, columnCount, barDurationMs],
  );

  mapXToMsRef.current = (x: number) => {
    const w = Math.max(1, widthRef.current);
    const clampedX = clamp01(x / w);
    if (mode === 'full') {
      return clampedX * totalMsFull;
    }
    const start = rollingEndMs - windowMs;
    const t = start + clampedX * windowMs;
    return Math.max(0, Math.min(totalMsFull, t));
  };

  const endScrub = () => {
    setScrubMs(null);
    onScrubChange?.(null);
  };

  const panResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => seekable,
        onMoveShouldSetPanResponder: () => seekable,
        onPanResponderGrant: (evt) => {
          if (!seekable) return;
          const ms = mapXToMsRef.current(evt.nativeEvent.locationX);
          setScrubMs(ms);
          onScrubChange?.(ms);
        },
        onPanResponderMove: (evt) => {
          if (!seekable) return;
          const ms = mapXToMsRef.current(evt.nativeEvent.locationX);
          setScrubMs(ms);
          onScrubChange?.(ms);
        },
        onPanResponderRelease: (evt) => {
          if (!seekable) return;
          const ms = mapXToMsRef.current(evt.nativeEvent.locationX);
          onSeek?.(ms);
          endScrub();
        },
        onPanResponderTerminate: () => {
          endScrub();
        },
      }),
    [seekable, onSeek, onScrubChange, mode, rollingEndMs, windowMs, totalMsFull],
  );

  const { envelopeBars, bipolarBars, barPx } = useMemo(() => {
    const minPx = Math.max(1, Math.floor(minBarPx));
    const maxCount = Math.max(1, Math.floor(width / minPx));

    if (bipolar && bipolarColumns) {
      const source = bipolarColumns;
      if (source.length === 0) {
        return { envelopeBars: [] as number[], bipolarBars: [] as BipolarColumn[], barPx: minPx };
      }
      const downsample = (cols: BipolarColumn[]): BipolarColumn[] => {
        if (cols.length <= maxCount) return cols;
        const bucketSize = Math.ceil(cols.length / maxCount);
        const out: BipolarColumn[] = [];
        for (let i = 0; i < cols.length; i += bucketSize) {
          let min = 0;
          let max = 0;
          for (let j = 0; j < bucketSize && i + j < cols.length; j++) {
            min = Math.min(min, cols[i + j].min);
            max = Math.max(max, cols[i + j].max);
          }
          out.push({ min, max });
        }
        return out;
      };
      const cols = downsample(source);
      return {
        envelopeBars: [] as number[],
        bipolarBars: cols,
        barPx: width / Math.max(1, cols.length),
      };
    }

    const source = peaksDb;
    if (source.length === 0) {
      return { envelopeBars: [] as number[], bipolarBars: [] as BipolarColumn[], barPx: minPx };
    }
    if (source.length <= maxCount) {
      return {
        envelopeBars: source.map((db) => dbToHeight01(db, dbRange)),
        bipolarBars: [] as BipolarColumn[],
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
    return {
      envelopeBars: out,
      bipolarBars: [] as BipolarColumn[],
      barPx: width / Math.max(1, out.length),
    };
  }, [peaksDb, bipolarColumns, bipolar, width, minBarPx, dbRange]);

  const drawnBarWidth = Math.max(1, barPx - BAR_GAP_PX);
  const midY = height / 2;

  const cursorX = useMemo(() => {
    if (!showCursor) return -100;
    if (mode === 'full') {
      return clamp01(displayProgressMs / totalMsFull) * width;
    }
    if (cursorMode === 'pinned' && scrubPreviewMs == null && scrubMs == null) {
      return width - 1;
    }
    const start = rollingEndMs - windowMs;
    return clamp01((displayProgressMs - start) / windowMs) * width;
  }, [
    showCursor,
    mode,
    cursorMode,
    scrubPreviewMs,
    scrubMs,
    width,
    displayProgressMs,
    rollingEndMs,
    totalMsFull,
    windowMs,
  ]);

  const tagRects = useMemo(() => {
    if (!showTagMarkers || tagTimestamps.length === 0) return [];

    if (mode === 'full') {
      const totalMs = Math.max(1, durationMs ?? columnCount * barDurationMs);
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
    columnCount,
    barDurationMs,
    visibleStartMs,
    visibleEndMs,
    windowMs,
    width,
    tagWidthMs,
  ]);

  return (
    <View style={[styles.container, { height }]} onLayout={onLayout}>
      {bipolar ? (
        <View style={styles.barsRow}>
          {bipolarBars.map((col, i) => {
            const maxN = clamp01(Math.abs(col.max) / bipolarScale);
            const minN = clamp01(Math.abs(col.min) / bipolarScale);
            const topH = Math.max(col.max === 0 && col.min === 0 ? 0 : 1, maxN * midY);
            const botH = Math.max(col.max === 0 && col.min === 0 ? 0 : 1, minN * midY);
            const silent = col.max === 0 && col.min === 0;
            return (
              <View
                key={i}
                style={{
                  width: drawnBarWidth,
                  marginRight: BAR_GAP_PX,
                  height,
                  justifyContent: 'center',
                }}
              >
                <View
                  style={{
                    position: 'absolute',
                    left: 0,
                    width: drawnBarWidth,
                    bottom: midY,
                    height: silent ? 1 : topH,
                    backgroundColor: silent ? '#E8E8E8' : '#6B6B6B',
                    borderTopLeftRadius: 1,
                    borderTopRightRadius: 1,
                  }}
                />
                <View
                  style={{
                    position: 'absolute',
                    left: 0,
                    width: drawnBarWidth,
                    top: midY,
                    height: silent ? 0 : botH,
                    backgroundColor: '#6B6B6B',
                    borderBottomLeftRadius: 1,
                    borderBottomRightRadius: 1,
                  }}
                />
              </View>
            );
          })}
        </View>
      ) : (
        <View style={styles.barsRow}>
          {envelopeBars.map((h01, i) => {
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
      )}

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

      {showCursor && (
        <View
          style={[
            styles.cursor,
            { left: cursorX },
            scrubMs !== null && styles.cursorScrubbing,
          ]}
        />
      )}

      {seekable && (
        <View
          style={styles.seekOverlay}
          {...panResponder.panHandlers}
          accessibilityRole="adjustable"
          accessibilityLabel="Waveform seek"
        />
      )}
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
  cursorScrubbing: {
    width: 3,
    backgroundColor: '#D32F2F',
  },
  seekOverlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
  },
});
