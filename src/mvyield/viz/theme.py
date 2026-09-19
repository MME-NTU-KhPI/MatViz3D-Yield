"""Visual theme: colours, colormaps and global matplotlib settings.

Palette and colormaps are carried over from the pre-refactor ``config.THEME``
so figures already in the paper keep their appearance. What changes is where
the settings live: applying them was previously scattered across a private
``_style_ax`` helper called by hand in every plotting function, which meant a
function that forgot the call quietly produced an off-theme figure.

One rcParam here is not cosmetic. ``svg.fonttype = "none"`` keeps text as
text in vector output; the matplotlib default converts glyphs to paths,
which looks identical on screen but makes every label uneditable in
Inkscape or Illustrator and unsearchable in a PDF. A journal asking for a
label change would otherwise mean regenerating the figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class Theme:
    """Colours and typography for every figure.

    Parameters
    ----------
    background : str
        Figure canvas colour.
    card : str
        Axes face colour.
    border : str
        Spines and grid lines.
    text_heading, text_body, text_muted : str
        Title, label and annotation colours.
    blue, rose, green, amber : str
        Series colours. :attr:`method_colours` maps methods onto them so a
        method keeps its colour across every figure.
    """

    background: str = "#F7F9FC"
    card: str = "#FFFFFF"
    border: str = "#E4EAF2"
    grid: str = "#EEF2F6"

    text_heading: str = "#1A2035"
    text_body: str = "#4A5568"
    text_muted: str = "#8899AA"

    blue: str = "#3B82F6"
    blue_light: str = "#DBEAFE"
    rose: str = "#F43F5E"
    rose_light: str = "#FFE4E6"
    green: str = "#10B981"
    amber: str = "#F59E0B"

    font_size: float = 10.0
    title_size: float = 11.0
    label_size: float = 10.0
    tick_size: float = 9.0

    method_colours: dict[str, str] = field(
        default_factory=lambda: {
            "kde_slice": "#F43F5E",
            "kde_ray": "#10B981",
            "analytic": "#3B82F6",
            "pinn": "#F59E0B",
        }
    )

    def method_colour(self, method: str) -> str:
        """Colour reserved for a method, consistent across all figures."""
        return self.method_colours.get(method, self.text_body)

    def style_axes(self, ax, fig=None) -> None:
        """Apply the theme to one axes.

        Called by the rendering helpers rather than by individual plotting
        functions, so a figure cannot end up unstyled by omission.
        """
        if fig is not None:
            fig.patch.set_facecolor(self.background)
        ax.set_facecolor(self.card)
        ax.grid(color=self.border, linestyle="--", linewidth=0.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(self.border)
        ax.tick_params(colors=self.text_body, labelsize=self.tick_size)

    def title(self, ax, text: str, **kwargs) -> None:
        """Set a themed axes title."""
        ax.set_title(
            text,
            color=self.text_heading,
            fontweight="bold",
            fontsize=self.title_size,
            pad=kwargs.pop("pad", 12),
            **kwargs,
        )

    def axis_labels(self, ax, x: str | None = None, y: str | None = None) -> None:
        """Set themed axis labels."""
        if x is not None:
            ax.set_xlabel(x, color=self.text_body, fontsize=self.label_size)
        if y is not None:
            ax.set_ylabel(y, color=self.text_body, fontsize=self.label_size)

    def legend(self, ax, **kwargs):
        """Add a themed legend."""
        return ax.legend(
            facecolor=self.card,
            edgecolor=self.border,
            fontsize=kwargs.pop("fontsize", 9),
            **kwargs,
        )

    def colorbar(self, fig, mappable, ax, label: str | None = None, **kwargs):
        """Add a themed colourbar."""
        bar = fig.colorbar(mappable, ax=ax, **kwargs)
        bar.ax.tick_params(colors=self.text_body, labelsize=self.tick_size)
        bar.outline.set_edgecolor(self.border)
        if label:
            bar.set_label(label, color=self.text_heading, fontweight="bold")
        return bar


THEME = Theme()
"""The default theme, used unless a caller passes its own."""


CMAP_PROBABILITY = LinearSegmentedColormap.from_list(
    "mvy_probability", ["#EFF6FF", "#93C5FD", "#3B82F6", "#F59E0B", "#F43F5E"]
)
"""Yield probability, 0 to 1: cool where safe, warm where yielding."""

CMAP_COVERAGE = LinearSegmentedColormap.from_list(
    "mvy_coverage", ["#F87171", "#FAFAFA", "#34D399"]
)
"""Dataset coverage: red where the query leaves the supported region."""

CMAP_SPREAD = LinearSegmentedColormap.from_list(
    "mvy_spread", ["#EFF6FF", "#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A5F"]
)
"""Estimator uncertainty in MPa; darker means less certain."""

CMAP_STRESS = "viridis"
"""Stress magnitude, where a perceptually uniform ramp matters more than hue."""

CMAP_DIFFERENCE = "RdBu_r"
"""Signed difference between two methods, centred on zero."""


def apply_rcparams(theme: Theme = THEME) -> None:
    """Install global matplotlib settings.

    Notes
    -----
    ``svg.fonttype = "none"`` is the load-bearing entry: without it,
    matplotlib renders text as outlines in SVG and the labels can no longer
    be edited or searched.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.background,
            "axes.facecolor": theme.card,
            "axes.edgecolor": theme.border,
            "axes.labelcolor": theme.text_body,
            "axes.titlecolor": theme.text_heading,
            "text.color": theme.text_body,
            "xtick.color": theme.text_body,
            "ytick.color": theme.text_body,
            "grid.color": theme.border,
            "font.size": theme.font_size,
            "axes.titlesize": theme.title_size,
            "axes.labelsize": theme.label_size,
            "xtick.labelsize": theme.tick_size,
            "ytick.labelsize": theme.tick_size,
            "legend.fontsize": 9.0,
            "figure.autolayout": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )
