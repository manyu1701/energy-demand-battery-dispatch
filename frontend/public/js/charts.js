/**
 * Minimal SVG charting, written from scratch.
 *
 * There is no charting library here on purpose: the dashboard has to run from
 * a fresh clone with no CDN and no build step, and the four chart types it
 * needs are less code than the wiring a library would require.
 *
 * Every chart re-renders on container resize, so it stays sharp instead of
 * being scaled by a viewBox.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
const MARGIN = { top: 16, right: 16, bottom: 30, left: 56 };

export const PALETTE = [
  '#4ea8de', // blue
  '#f4a261', // amber
  '#84cc8b', // green
  '#e07a7a', // red
  '#b79ced', // violet
  '#6fd0c4', // teal
];

function el(name, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
  if (min === max) return [min];

  const rawStep = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalised = rawStep / magnitude;
  const step = (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;

  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-9; t += step) {
    ticks.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return ticks;
}

export function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (abs >= 10_000) return `${(value / 1000).toFixed(1)}k`;
  return value.toFixed(abs >= 100 ? 0 : digits);
}

function formatTimeLabel(value, span) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  const opts =
    span > 60 * 24 * 3600 * 1000
      ? { month: 'short', timeZone: 'UTC' }
      : span > 3 * 24 * 3600 * 1000
        ? { day: 'numeric', month: 'short', timeZone: 'UTC' }
        : { hour: '2-digit', hour12: false, timeZone: 'UTC' };
  return new Intl.DateTimeFormat('en-GB', opts).format(date);
}

export function formatTimestamp(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'UTC',
  }).format(date);
}

/** Timestamps in this project carry no zone and mean UTC. */
export function parseTimestamp(value) {
  if (value instanceof Date) return value;
  const text = String(value);
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text);
  return new Date(hasZone ? text : `${text}Z`);
}

class ChartBase {
  constructor(container) {
    this.container = container;
    this.container.classList.add('chart');
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'chart-tooltip';
    this.tooltip.hidden = true;
    this.container.append(this.tooltip);

    // ResizeObserver fires once on observe, which performs the initial render.
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(this.container);
  }

  get width() {
    return Math.max(this.container.clientWidth, 260);
  }

  get height() {
    return Math.max(this.container.clientHeight, 180);
  }

  reset() {
    this.container.querySelector('svg')?.remove();
    const svg = el('svg', { width: this.width, height: this.height, class: 'chart-svg' });
    this.container.prepend(svg);
    this.svg = svg;
    this.plotWidth = this.width - MARGIN.left - MARGIN.right;
    this.plotHeight = this.height - MARGIN.top - MARGIN.bottom;
    this.plot = el('g', { transform: `translate(${MARGIN.left},${MARGIN.top})` });
    svg.append(this.plot);
    return svg;
  }

  drawYAxis(scale, ticks, formatter = (v) => formatNumber(v)) {
    for (const tick of ticks) {
      const y = scale(tick);
      this.plot.append(
        el('line', { x1: 0, x2: this.plotWidth, y1: y, y2: y, class: 'grid-line' }),
        el('text', { x: -10, y: y + 4, class: 'axis-label', 'text-anchor': 'end' }, formatter(tick)),
      );
    }
  }

  showTooltip(html, x, y) {
    this.tooltip.innerHTML = html;
    this.tooltip.hidden = false;
    const box = this.tooltip.getBoundingClientRect();
    const left = Math.min(Math.max(x - box.width / 2, 4), this.width - box.width - 4);
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${Math.max(y - box.height - 12, 4)}px`;
  }

  hideTooltip() {
    this.tooltip.hidden = true;
  }

  destroy() {
    this.observer.disconnect();
    this.container.replaceChildren();
  }
}

/**
 * Multi-series time-series line chart with a hover crosshair.
 *
 * config: {
 *   timestamps: string[] | Date[],
 *   series: [{ name, values: number[], color?, dashed?, axis?: 'left'|'right', area?: bool }],
 *   yLabel?, rightLabel?, formatValue?(v, series)
 * }
 */
export class TimeSeriesChart extends ChartBase {
  constructor(container, config) {
    super(container);
    this.update(config);
  }

  update(config) {
    this.config = { ...this.config, ...config };
    this.times = (this.config.timestamps ?? []).map(parseTimestamp);
    this.render();
  }

  render() {
    const { series = [], formatValue } = this.config ?? {};
    if (!this.times?.length || !series.length) return;

    this.reset();

    const leftSeries = series.filter((s) => s.axis !== 'right');
    const rightSeries = series.filter((s) => s.axis === 'right');

    const xMin = this.times[0].getTime();
    const xMax = this.times[this.times.length - 1].getTime();
    const xSpan = xMax - xMin || 1;
    const x = (time) => ((time - xMin) / xSpan) * this.plotWidth;

    const makeScale = (group) => {
      const values = group.flatMap((s) => s.values).filter((v) => typeof v === 'number' && !Number.isNaN(v));
      if (!values.length) return null;
      let min = Math.min(...values);
      let max = Math.max(...values);
      const pad = (max - min || Math.abs(max) || 1) * 0.08;
      min = min < 0 ? min - pad : Math.max(0, min - pad);
      max += pad;
      const scale = (v) => this.plotHeight - ((v - min) / (max - min || 1)) * this.plotHeight;
      scale.ticks = niceTicks(min, max, 5);
      return scale;
    };

    const yLeft = makeScale(leftSeries);
    const yRight = rightSeries.length ? makeScale(rightSeries) : null;
    if (!yLeft) return;

    this.drawYAxis(yLeft, yLeft.ticks, this.config.formatLeft);

    if (yRight) {
      for (const tick of yRight.ticks) {
        this.plot.append(
          el(
            'text',
            {
              x: this.plotWidth + 10,
              y: yRight(tick) + 4,
              class: 'axis-label',
              'text-anchor': 'start',
            },
            (this.config.formatRight ?? formatNumber)(tick),
          ),
        );
      }
    }

    // Zero line, drawn only when the data actually crosses it (negative
    // prices) — otherwise it duplicates the bottom axis.
    if (yLeft.ticks.some((t) => t < 0)) {
      this.plot.append(
        el('line', { x1: 0, x2: this.plotWidth, y1: yLeft(0), y2: yLeft(0), class: 'zero-line' }),
      );
    }

    const tickCount = Math.max(2, Math.min(8, Math.floor(this.plotWidth / 90)));
    let lastLabel = null;
    for (let i = 0; i <= tickCount; i += 1) {
      const time = xMin + (xSpan * i) / tickCount;
      const label = formatTimeLabel(new Date(time), xSpan);
      // Evenly spaced ticks can land twice inside the same day or month; a
      // repeated label reads as a rendering bug, so skip it.
      if (label === lastLabel) continue;
      lastLabel = label;
      this.plot.append(
        el(
          'text',
          { x: x(time), y: this.plotHeight + 20, class: 'axis-label', 'text-anchor': 'middle' },
          label,
        ),
      );
    }

    series.forEach((s, index) => {
      const scale = s.axis === 'right' ? yRight : yLeft;
      if (!scale) return;
      const color = s.color ?? PALETTE[index % PALETTE.length];

      const points = [];
      for (let i = 0; i < this.times.length; i += 1) {
        const value = s.values[i];
        if (typeof value !== 'number' || Number.isNaN(value)) continue;
        points.push(`${x(this.times[i].getTime()).toFixed(2)},${scale(value).toFixed(2)}`);
      }
      if (!points.length) return;

      if (s.area) {
        this.plot.append(
          el('polygon', {
            points: `${x(xMin)},${this.plotHeight} ${points.join(' ')} ${x(xMax)},${this.plotHeight}`,
            fill: color,
            'fill-opacity': 0.14,
            stroke: 'none',
          }),
        );
      }

      this.plot.append(
        el('polyline', {
          points: points.join(' '),
          fill: 'none',
          stroke: color,
          'stroke-width': s.width ?? 1.8,
          'stroke-dasharray': s.dashed ? '5 4' : null,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        }),
      );
    });

    this.attachCrosshair(x, xMin, xSpan, series, formatValue);
    this.renderLegend(series);
  }

  attachCrosshair(x, xMin, xSpan, series, formatValue) {
    const crosshair = el('line', {
      y1: 0,
      y2: this.plotHeight,
      class: 'crosshair',
      visibility: 'hidden',
    });
    const overlay = el('rect', {
      width: this.plotWidth,
      height: this.plotHeight,
      fill: 'transparent',
    });
    this.plot.append(crosshair, overlay);

    const onMove = (event) => {
      const bounds = this.svg.getBoundingClientRect();
      const offset = event.clientX - bounds.left - MARGIN.left;
      const ratio = Math.min(Math.max(offset / this.plotWidth, 0), 1);
      const targetTime = xMin + ratio * xSpan;

      // Nearest sample, not nearest index — series can be irregular.
      let index = 0;
      let best = Infinity;
      for (let i = 0; i < this.times.length; i += 1) {
        const distance = Math.abs(this.times[i].getTime() - targetTime);
        if (distance < best) {
          best = distance;
          index = i;
        }
      }

      const snapped = x(this.times[index].getTime());
      crosshair.setAttribute('x1', snapped);
      crosshair.setAttribute('x2', snapped);
      crosshair.setAttribute('visibility', 'visible');

      const rows = series
        .map((s, i) => {
          const value = s.values[index];
          if (typeof value !== 'number' || Number.isNaN(value)) return '';
          const color = s.color ?? PALETTE[i % PALETTE.length];
          const shown = formatValue ? formatValue(value, s) : formatNumber(value);
          return `<div class="tt-row"><span class="tt-dot" style="background:${color}"></span>
                  <span class="tt-name">${s.name}</span><span class="tt-value">${shown}</span></div>`;
        })
        .join('');

      this.showTooltip(
        `<div class="tt-title">${formatTimestamp(this.times[index])}</div>${rows}`,
        snapped + MARGIN.left,
        MARGIN.top + this.plotHeight / 3,
      );
    };

    overlay.addEventListener('mousemove', onMove);
    overlay.addEventListener('mouseleave', () => {
      crosshair.setAttribute('visibility', 'hidden');
      this.hideTooltip();
    });
  }

  renderLegend(series) {
    this.container.parentElement?.querySelector('[data-legend]')?.replaceChildren(
      ...series.map((s, i) => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const color = s.color ?? PALETTE[i % PALETTE.length];
        item.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${s.name}`;
        return item;
      }),
    );
  }
}

/**
 * Vertical bar chart, optionally with a line overlaid on a second axis.
 * Used for the dispatch view (charge/discharge bars + price line + SoC line).
 *
 * config: {
 *   timestamps, bars: [{name, values, color}], lines: [{name, values, color, axis}]
 * }
 */
export class DispatchChart extends ChartBase {
  constructor(container, config) {
    super(container);
    this.update(config);
  }

  update(config) {
    this.config = { ...this.config, ...config };
    this.times = (this.config.timestamps ?? []).map(parseTimestamp);
    this.render();
  }

  render() {
    const { bars = [], lines = [] } = this.config ?? {};
    if (!this.times?.length) return;

    this.reset();

    const n = this.times.length;
    const slot = this.plotWidth / n;
    const barWidth = Math.max(slot * 0.72, 1);

    // State of charge is in MWh just like the bars, so it shares the left
    // axis and its range has to be part of that scale's domain.
    const leftLines = lines.filter((l) => l.axis === 'left');
    const rightLines = lines.filter((l) => l.axis !== 'left');

    const leftValues = [...bars, ...leftLines].flatMap((s) => s.values).filter(Number.isFinite);
    const barMax = Math.max(...leftValues, 0);
    const barMin = Math.min(...leftValues, 0);
    const padded = (barMax - barMin) * 0.1 || 1;
    const yBar = (v) =>
      this.plotHeight -
      ((v - (barMin - padded)) / (barMax + padded - (barMin - padded))) * this.plotHeight;

    this.drawYAxis(yBar, niceTicks(barMin - padded, barMax + padded, 5), this.config.formatLeft);

    bars.forEach((bar, index) => {
      const color = bar.color ?? PALETTE[index % PALETTE.length];
      for (let i = 0; i < n; i += 1) {
        const value = bar.values[i];
        if (!Number.isFinite(value) || Math.abs(value) < 1e-9) continue;
        const top = Math.min(yBar(value), yBar(0));
        this.plot.append(
          el('rect', {
            x: i * slot + (slot - barWidth) / 2,
            y: top,
            width: barWidth,
            height: Math.max(Math.abs(yBar(value) - yBar(0)), 1),
            fill: color,
            'fill-opacity': 0.85,
          }),
        );
      }
    });

    for (const line of leftLines) {
      const points = line.values
        .map((v, i) => (Number.isFinite(v) ? `${(i + 0.5) * slot},${yBar(v)}` : null))
        .filter(Boolean)
        .join(' ');
      this.plot.append(
        el('polyline', {
          points,
          fill: 'none',
          stroke: line.color ?? PALETTE[4],
          'stroke-width': line.width ?? 1.6,
          'stroke-dasharray': line.dashed ? '5 4' : null,
        }),
      );
    }

    const lineValues = rightLines.flatMap((l) => l.values).filter(Number.isFinite);
    let yLine = null;
    if (lineValues.length) {
      let min = Math.min(...lineValues);
      let max = Math.max(...lineValues);
      const pad = (max - min || Math.abs(max) || 1) * 0.1;
      min -= pad;
      max += pad;
      yLine = (v) => this.plotHeight - ((v - min) / (max - min || 1)) * this.plotHeight;

      for (const tick of niceTicks(min, max, 5)) {
        this.plot.append(
          el(
            'text',
            {
              x: this.plotWidth + 10,
              y: yLine(tick) + 4,
              class: 'axis-label',
              'text-anchor': 'start',
            },
            (this.config.formatRight ?? formatNumber)(tick),
          ),
        );
      }

      rightLines.forEach((line, index) => {
        const color = line.color ?? PALETTE[(bars.length + index) % PALETTE.length];
        const points = line.values
          .map((v, i) => (Number.isFinite(v) ? `${(i + 0.5) * slot},${yLine(v)}` : null))
          .filter(Boolean)
          .join(' ');
        this.plot.append(
          el('polyline', {
            points,
            fill: 'none',
            stroke: color,
            'stroke-width': line.width ?? 1.8,
            'stroke-dasharray': line.dashed ? '5 4' : null,
          }),
        );
      });
    }

    this.plot.append(
      el('line', { x1: 0, x2: this.plotWidth, y1: yBar(0), y2: yBar(0), class: 'zero-line' }),
    );

    const tickEvery = Math.max(1, Math.round(n / Math.max(2, Math.floor(this.plotWidth / 80))));
    const span = this.times[n - 1] - this.times[0];
    let lastLabel = null;
    for (let i = 0; i < n; i += tickEvery) {
      const label = formatTimeLabel(this.times[i], span);
      if (label === lastLabel) continue; // see TimeSeriesChart.render
      lastLabel = label;
      this.plot.append(
        el(
          'text',
          {
            x: (i + 0.5) * slot,
            y: this.plotHeight + 20,
            class: 'axis-label',
            'text-anchor': 'middle',
          },
          label,
        ),
      );
    }

    this.attachHover(slot, bars, lines);
    this.renderLegend([...bars, ...lines]);
  }

  attachHover(slot, bars, lines) {
    const overlay = el('rect', {
      width: this.plotWidth,
      height: this.plotHeight,
      fill: 'transparent',
    });
    const highlight = el('rect', {
      width: slot,
      height: this.plotHeight,
      class: 'hover-band',
      visibility: 'hidden',
    });
    this.plot.append(highlight, overlay);

    overlay.addEventListener('mousemove', (event) => {
      const bounds = this.svg.getBoundingClientRect();
      const offset = event.clientX - bounds.left - MARGIN.left;
      const index = Math.min(Math.max(Math.floor(offset / slot), 0), this.times.length - 1);

      highlight.setAttribute('x', index * slot);
      highlight.setAttribute('visibility', 'visible');

      const rows = [...bars, ...lines]
        .map((s, i) => {
          const value = s.values[index];
          if (!Number.isFinite(value)) return '';
          const color = s.color ?? PALETTE[i % PALETTE.length];
          const shown = this.config.formatValue
            ? this.config.formatValue(value, s)
            : formatNumber(value, 2);
          return `<div class="tt-row"><span class="tt-dot" style="background:${color}"></span>
                  <span class="tt-name">${s.name}</span><span class="tt-value">${shown}</span></div>`;
        })
        .join('');

      this.showTooltip(
        `<div class="tt-title">${formatTimestamp(this.times[index])}</div>${rows}`,
        index * slot + slot / 2 + MARGIN.left,
        MARGIN.top + this.plotHeight / 3,
      );
    });

    overlay.addEventListener('mouseleave', () => {
      highlight.setAttribute('visibility', 'hidden');
      this.hideTooltip();
    });
  }

  renderLegend(series) {
    TimeSeriesChart.prototype.renderLegend.call(this, series);
  }
}

/**
 * Horizontal bar chart for ranked categories (feature importance) or a simple
 * vertical one for hour-of-day profiles.
 *
 * config: { labels: string[], values: number[], color?, horizontal?, formatValue? }
 */
export class BarChart extends ChartBase {
  constructor(container, config) {
    super(container);
    this.update(config);
  }

  update(config) {
    this.config = { ...this.config, ...config };
    this.render();
  }

  render() {
    const { labels = [], values = [], color = PALETTE[0], horizontal = false } = this.config ?? {};
    if (!labels.length) return;

    const format = this.config.formatValue ?? ((v) => formatNumber(v, 2));
    this.reset();

    const max = Math.max(...values.filter(Number.isFinite), 0) || 1;

    if (horizontal) {
      // Long feature names need the room; the shared left margin is too tight.
      const labelWidth = Math.min(150, this.width * 0.42);
      const plotWidth = this.plotWidth + MARGIN.left - labelWidth;
      const rowHeight = this.plotHeight / labels.length;

      labels.forEach((label, i) => {
        const y = i * rowHeight;
        const barHeight = Math.max(rowHeight * 0.62, 3);
        const width = Math.max((values[i] / max) * plotWidth, 1);
        const group = el('g', { transform: `translate(${labelWidth - MARGIN.left},${y})` });

        group.append(
          el(
            'text',
            { x: -8, y: barHeight / 2 + 4, class: 'axis-label', 'text-anchor': 'end' },
            label,
          ),
          el('rect', { x: 0, y: 0, width, height: barHeight, fill: color, rx: 2 }),
          el(
            'text',
            { x: width + 6, y: barHeight / 2 + 4, class: 'axis-label bar-value' },
            format(values[i]),
          ),
        );
        this.plot.append(group);
      });
      return;
    }

    const slot = this.plotWidth / labels.length;
    const barWidth = Math.max(slot * 0.68, 1);
    const y = (v) => this.plotHeight - (v / max) * this.plotHeight;

    this.drawYAxis(y, niceTicks(0, max, 4), this.config.formatLeft);

    labels.forEach((label, i) => {
      const value = values[i];
      if (!Number.isFinite(value)) return;
      const bar = el('rect', {
        x: i * slot + (slot - barWidth) / 2,
        y: y(value),
        width: barWidth,
        height: Math.max(this.plotHeight - y(value), 1),
        fill: color,
        rx: 2,
      });
      bar.addEventListener('mousemove', (event) => {
        const bounds = this.svg.getBoundingClientRect();
        this.showTooltip(
          `<div class="tt-title">${label}</div><div class="tt-row"><span class="tt-value">${format(value)}</span></div>`,
          event.clientX - bounds.left,
          event.clientY - bounds.top,
        );
      });
      bar.addEventListener('mouseleave', () => this.hideTooltip());
      this.plot.append(bar);

      const every = Math.max(1, Math.round(labels.length / Math.floor(this.plotWidth / 40)));
      if (i % every === 0) {
        this.plot.append(
          el(
            'text',
            {
              x: i * slot + slot / 2,
              y: this.plotHeight + 20,
              class: 'axis-label',
              'text-anchor': 'middle',
            },
            label,
          ),
        );
      }
    });
  }
}
