"""Shared layer-based Matplotlib renderer for Cartest visualizations.

Scenario-specific code describes what to draw using plain dataclasses; this
module only knows how to render those layers onto an axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


Aspect = Literal["auto", "equal"]


@dataclass(frozen=True)
class RectLayer:
    xy: tuple[float, float]
    width: float
    height: float
    facecolor: str
    edgecolor: str = "none"
    linewidth: float = 0.0
    alpha: float = 1.0
    zorder: float = 1.0
    gid: str | None = None
    boxstyle: str | None = None


@dataclass(frozen=True)
class CircleLayer:
    center: tuple[float, float]
    radius: float
    facecolor: str
    edgecolor: str = "none"
    linewidth: float = 0.0
    alpha: float = 1.0
    zorder: float = 1.0
    gid: str | None = None


@dataclass(frozen=True)
class PolygonLayer:
    xy: tuple[tuple[float, float], ...]
    facecolor: str
    edgecolor: str = "none"
    linewidth: float = 0.0
    alpha: float = 1.0
    zorder: float = 1.0
    gid: str | None = None


@dataclass(frozen=True)
class LineLayer:
    x: tuple[float, ...]
    y: tuple[float, ...]
    color: str
    linewidth: float = 1.0
    linestyle: Any = "-"
    alpha: float = 1.0
    zorder: float = 1.0
    gid: str | None = None


@dataclass(frozen=True)
class TextLayer:
    x: float
    y: float
    text: str
    color: str
    fontsize: float = 8.0
    ha: str = "left"
    va: str = "top"
    transform: Literal["data", "axes"] = "data"
    bbox: dict[str, Any] | None = None
    zorder: float = 10.0
    gid: str | None = None


@dataclass(frozen=True)
class VehicleLayer:
    center: tuple[float, float]
    heading: float
    length: float
    width: float
    facecolor: str
    edgecolor: str = "white"
    linewidth: float = 0.8
    alpha: float = 1.0
    zorder: float = 5.0
    gid: str | None = None


@dataclass(frozen=True)
class SceneRenderSpec:
    facecolor: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    aspect: Aspect | None = None
    patches: tuple[RectLayer | CircleLayer | PolygonLayer, ...] = field(default_factory=tuple)
    lines: tuple[LineLayer, ...] = field(default_factory=tuple)
    vehicles: tuple[VehicleLayer, ...] = field(default_factory=tuple)
    texts: tuple[TextLayer, ...] = field(default_factory=tuple)
    hide_ticks: bool = False
    hide_spines: bool = False


def _set_gid(artist, gid):
    if gid:
        artist.set_gid(gid)
    return artist


def draw_vehicle_footprint(ax, layer: VehicleLayer):
    """Draw a top-down oriented vehicle footprint."""
    center_x, center_y = layer.center
    patch = mpatches.Rectangle(
        (center_x - 0.5 * layer.length, center_y - 0.5 * layer.width),
        layer.length,
        layer.width,
        facecolor=layer.facecolor,
        edgecolor=layer.edgecolor,
        linewidth=layer.linewidth,
        alpha=layer.alpha,
    )
    transform = (
        mtransforms.Affine2D().rotate_around(center_x, center_y, layer.heading)
        + ax.transData
    )
    patch.set_transform(transform)
    patch.set_zorder(layer.zorder)
    _set_gid(patch, layer.gid)
    ax.add_patch(patch)
    return patch


def _draw_rect(ax, layer: RectLayer):
    if layer.boxstyle is None:
        patch = mpatches.Rectangle(
            layer.xy,
            layer.width,
            layer.height,
            facecolor=layer.facecolor,
            edgecolor=layer.edgecolor,
            linewidth=layer.linewidth,
            alpha=layer.alpha,
        )
    else:
        patch = mpatches.FancyBboxPatch(
            layer.xy,
            layer.width,
            layer.height,
            boxstyle=layer.boxstyle,
            facecolor=layer.facecolor,
            edgecolor=layer.edgecolor,
            linewidth=layer.linewidth,
            alpha=layer.alpha,
        )
    patch.set_zorder(layer.zorder)
    _set_gid(patch, layer.gid)
    ax.add_patch(patch)
    return patch


def _draw_circle(ax, layer: CircleLayer):
    patch = mpatches.Circle(
        layer.center,
        layer.radius,
        facecolor=layer.facecolor,
        edgecolor=layer.edgecolor,
        linewidth=layer.linewidth,
        alpha=layer.alpha,
    )
    patch.set_zorder(layer.zorder)
    _set_gid(patch, layer.gid)
    ax.add_patch(patch)
    return patch


def _draw_polygon(ax, layer: PolygonLayer):
    patch = mpatches.Polygon(
        layer.xy,
        closed=True,
        facecolor=layer.facecolor,
        edgecolor=layer.edgecolor,
        linewidth=layer.linewidth,
        alpha=layer.alpha,
    )
    patch.set_zorder(layer.zorder)
    _set_gid(patch, layer.gid)
    ax.add_patch(patch)
    return patch


def render_scene(ax, spec: SceneRenderSpec):
    """Render a scene spec onto an existing Matplotlib axis."""
    if spec.facecolor is not None:
        ax.set_facecolor(spec.facecolor)
    if spec.xlim is not None:
        ax.set_xlim(*spec.xlim)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    if spec.aspect is not None:
        ax.set_aspect(spec.aspect, adjustable="box")

    for patch in spec.patches:
        if isinstance(patch, RectLayer):
            _draw_rect(ax, patch)
        elif isinstance(patch, CircleLayer):
            _draw_circle(ax, patch)
        elif isinstance(patch, PolygonLayer):
            _draw_polygon(ax, patch)
        else:
            raise TypeError(f"Unsupported patch layer: {patch!r}")

    for line in spec.lines:
        artists = ax.plot(
            line.x,
            line.y,
            color=line.color,
            lw=line.linewidth,
            ls=line.linestyle,
            alpha=line.alpha,
            zorder=line.zorder,
        )
        for artist in artists:
            _set_gid(artist, line.gid)

    for vehicle in spec.vehicles:
        draw_vehicle_footprint(ax, vehicle)

    for text in spec.texts:
        transform = ax.transAxes if text.transform == "axes" else ax.transData
        artist = ax.text(
            text.x,
            text.y,
            text.text,
            transform=transform,
            color=text.color,
            fontsize=text.fontsize,
            ha=text.ha,
            va=text.va,
            bbox=text.bbox,
            zorder=text.zorder,
        )
        _set_gid(artist, text.gid)

    if spec.hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if spec.hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    return ax
