"use client";

import { useEffect, useRef } from "react";

import { anomalyColor, finite } from "@/lib/color";
import type { AtlasBundle } from "@/lib/bundle";

interface MapCanvasProps {
  bundle: AtlasBundle;
  field: Float32Array;
  coverage: Float32Array;
  year: number;
  globe: boolean;
  onSelect: (cell: { lat: number; lon: number; value: number | null; coverage: number | null }) => void;
}

export function MapCanvas({ bundle, field, coverage, year, globe, onSelect }: MapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const yearIndex = bundle.grid.years.indexOf(year);
  const available = yearIndex >= 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      paint(context, rect.width, rect.height);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [bundle, field, coverage, year, globe, yearIndex]);

  function paint(context: CanvasRenderingContext2D, width: number, height: number) {
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#e7e0d4";
    context.fillRect(0, 0, width, height);
    if (!available) {
      context.fillStyle = "#5c564c";
      context.font = "16px sans-serif";
      context.fillText("No gridded observations for this year.", 24, 40);
      context.font = "13px sans-serif";
      context.fillText("The global emulation can continue. The map does not invent a field.", 24, 64);
      return;
    }
    const { nlat, nlon, lat, lon } = bundle.grid;
    const offset = yearIndex * nlat * nlon;
    if (!globe) {
      for (let j = 0; j < nlat; j += 1) {
        const south = lat[j] - 2.5;
        const north = lat[j] + 2.5;
        const y0 = ((90 - north) / 180) * height;
        const y1 = ((90 - south) / 180) * height;
        for (let i = 0; i < nlon; i += 1) {
          const sample = field[offset + j * nlon + i];
          if (!finite(sample)) continue;
          const cover = coverage[offset + j * nlon + i];
          const west = lon[i] - 2.5;
          const east = lon[i] + 2.5;
          const x0 = ((west + 180) / 360) * width;
          const x1 = ((east + 180) / 360) * width;
          context.fillStyle = anomalyColor(sample, 0.35 + 0.65 * Math.max(0, Math.min(1, cover)));
          context.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
        }
      }
      return;
    }
    const radius = Math.min(width, height) * 0.46;
    const cx = width / 2;
    const cy = height / 2;
    context.beginPath();
    context.arc(cx, cy, radius, 0, Math.PI * 2);
    context.fillStyle = "#efe8dc";
    context.fill();
    for (let j = 0; j < nlat; j += 1) {
      for (let i = 0; i < nlon; i += 1) {
        const sample = field[offset + j * nlon + i];
        if (!finite(sample)) continue;
        const cover = coverage[offset + j * nlon + i];
        const corners = [
          project(lon[i] - 2.5, lat[j] - 2.5, cx, cy, radius),
          project(lon[i] + 2.5, lat[j] - 2.5, cx, cy, radius),
          project(lon[i] + 2.5, lat[j] + 2.5, cx, cy, radius),
          project(lon[i] - 2.5, lat[j] + 2.5, cx, cy, radius),
        ];
        if (corners.some((corner) => corner == null)) continue;
        context.beginPath();
        corners.forEach((corner, index) => {
          if (!corner) return;
          if (index === 0) context.moveTo(corner[0], corner[1]);
          else context.lineTo(corner[0], corner[1]);
        });
        context.closePath();
        context.fillStyle = anomalyColor(sample, 0.35 + 0.65 * Math.max(0, Math.min(1, cover)));
        context.fill();
      }
    }
    context.beginPath();
    context.arc(cx, cy, radius, 0, Math.PI * 2);
    context.strokeStyle = "#1c2430";
    context.lineWidth = 1;
    context.stroke();
  }

  function selectAt(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || yearIndex < 0 || globe) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const lon = (x / rect.width) * 360 - 180;
    const lat = 90 - (y / rect.height) * 180;
    const { nlat, nlon } = bundle.grid;
    const i = bundle.grid.lon.findIndex((center) => Math.abs(center - lon) <= 2.5);
    const j = bundle.grid.lat.findIndex((center) => Math.abs(center - lat) <= 2.5);
    if (i < 0 || j < 0) return;
    const index = yearIndex * nlat * nlon + j * nlon + i;
    const sample = field[index];
    const cover = coverage[index];
    onSelect({
      lat: bundle.grid.lat[j],
      lon: bundle.grid.lon[i],
      value: finite(sample) ? sample : null,
      coverage: finite(cover) ? cover : null,
    });
  }

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full cursor-crosshair rounded-md"
      onClick={selectAt}
      role="img"
      aria-label={
        available
          ? `HadCRUT non-infilled anomalies for ${year}. Empty cells were not observed.`
          : `No gridded field for ${year}.`
      }
    />
  );
}

function project(lon: number, lat: number, cx: number, cy: number, radius: number): [number, number] | null {
  const lam = (lon * Math.PI) / 180;
  const phi = (lat * Math.PI) / 180;
  const x = Math.cos(phi) * Math.sin(lam);
  const y = Math.sin(phi);
  const z = Math.cos(phi) * Math.cos(lam);
  if (z < 0) return null;
  return [cx + radius * x, cy - radius * y];
}
