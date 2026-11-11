from __future__ import annotations

"""
Mini Variable Explorer
======================

Ein kleiner lokaler Variable Explorer fuer Python-Skripte.
Gedacht als leichter Ersatz fuer den Spyder Variable Explorer,
falls du in Cursor keine vernuenftige Tabellenansicht hast.

Features:
- zeigt Name, Typ, Shape, Laenge und Kurzvorschau aller Variablen
- DataFrames werden als HTML-Tabelle dargestellt
- NumPy Arrays bekommen Form, Statistik und optional Histogramm/Heatmap
- Listen, Dicts, Series, Strings und Skalare werden sinnvoll angezeigt
- laeuft lokal im Browser ueber Flask

Typische Nutzung in deinem Skript:

    from mini_variable_explorer import explore

    # ... dein Code ...
    explore(locals())

Oder nur ausgewaehlte Variablen:

    explore({
        "foo": foo,
        "bar": bar,
        "fubar": fubar,
    })

Dann im Browser oeffnen:
http://127.0.0.1:5001
"""

from dataclasses import dataclass
from html import escape
from io import BytesIO
import base64
import math
import threading
import webbrowser
from typing import Any, Mapping

import numpy as np
import pandas as pd
from flask import Flask, abort, render_template_string, request, url_for

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INDEX_TEMPLATE = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>Mini Variable Explorer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f8f9fb; color: #222; }
    h1, h2, h3 { margin-bottom: 0.4rem; }
    .muted { color: #666; }
    .toolbar { margin: 16px 0 20px; }
    input[type=text] { padding: 8px 10px; width: 320px; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
    th { background: #f0f2f5; text-align: left; }
    tr:hover { background: #fafafa; }
    .card { background: white; padding: 18px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 20px; }
    .mono { font-family: Menlo, Consolas, monospace; }
    .preview { max-width: 520px; white-space: pre-wrap; }
    .small { font-size: 0.92rem; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #eef2ff; color: #334; font-size: 0.85rem; }
    a { color: #1d4ed8; text-decoration: none; }
    a:hover { text-decoration: underline; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
    .kv { margin: 8px 0; }
    .kv b { display: inline-block; min-width: 130px; }
  </style>
</head>
<body>
  <h1>Mini Variable Explorer</h1>
  <div class="muted">{{ count }} Variablen im aktuellen Namespace</div>

  <form class="toolbar" method="get">
    <input type="text" name="q" placeholder="Name oder Typ filtern" value="{{ query }}">
    <button type="submit">Filtern</button>
    <a href="/">Zuruecksetzen</a>
  </form>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Typ</th>
          <th>Shape / Laenge</th>
          <th>Preview</th>
        </tr>
      </thead>
      <tbody>
        {% for item in items %}
        <tr>
          <td><a class="mono" href="{{ url_for('view_var', name=item.name) }}">{{ item.name }}</a></td>
          <td>{{ item.type_name }}</td>
          <td>{{ item.shape }}</td>
          <td class="preview small mono">{{ item.preview }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


DETAIL_TEMPLATE = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>{{ name }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f8f9fb; color: #222; }
    h1, h2, h3 { margin-bottom: 0.4rem; }
    .muted { color: #666; }
    .card { background: white; padding: 18px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 20px; }
    .mono { font-family: Menlo, Consolas, monospace; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
    th { background: #f0f2f5; text-align: left; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }
    a { color: #1d4ed8; text-decoration: none; }
    a:hover { text-decoration: underline; }
    pre { background: #f5f5f5; padding: 12px; border-radius: 8px; overflow: auto; }
    .kv { margin: 8px 0; }
    .kv b { display: inline-block; min-width: 130px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #eef2ff; color: #334; font-size: 0.85rem; }
  </style>
</head>
<body>
  <p><a href="/">← Zurueck zur Uebersicht</a></p>
  <h1 class="mono">{{ name }}</h1>
  <div class="muted">{{ type_name }}</div>

  <div class="card">
    <div class="kv"><b>Typ:</b> <span class="pill">{{ type_name }}</span></div>
    <div class="kv"><b>Shape / Laenge:</b> <span class="mono">{{ shape }}</span></div>
    {% if dtype %}<div class="kv"><b>Dtype:</b> <span class="mono">{{ dtype }}</span></div>{% endif %}
    {% if memory %}<div class="kv"><b>Memory:</b> <span class="mono">{{ memory }}</span></div>{% endif %}
  </div>

  {% if stats_html %}
  <div class="card">
    <h2>Statistik</h2>
    {{ stats_html | safe }}
  </div>
  {% endif %}

  {% if plot_b64 %}
  <div class="card">
    <h2>Plot</h2>
    <img src="data:image/png;base64,{{ plot_b64 }}" alt="plot">
  </div>
  {% endif %}

  {% if table_html %}
  <div class="card">
    <h2>Tabellenansicht</h2>
    {{ table_html | safe }}
  </div>
  {% endif %}

  {% if text_preview %}
  <div class="card">
    <h2>Preview</h2>
    <pre class="mono">{{ text_preview }}</pre>
  </div>
  {% endif %}
</body>
</html>
"""


@dataclass
class VarInfo:
    name: str
    type_name: str
    shape: str
    preview: str


class VariableExplorer:
    def __init__(self, namespace: Mapping[str, Any], title: str = "Mini Variable Explorer") -> None:
        self.namespace = dict(namespace)
        self.title = title
        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.route("/")
        def index() -> str:
            query = request.args.get("q", "").strip().lower()
            items = self._collect_items()
            if query:
                items = [
                    item for item in items
                    if query in item.name.lower() or query in item.type_name.lower()
                ]
            return render_template_string(
                INDEX_TEMPLATE,
                items=items,
                query=query,
                count=len(items),
            )

        @self.app.route("/var/<name>")
        def view_var(name: str) -> str:
            if name not in self.namespace:
                abort(404)
            value = self.namespace[name]
            details = self._build_details(value)
            return render_template_string(
                DETAIL_TEMPLATE,
                name=name,
                type_name=type(value).__name__,
                shape=self._shape_str(value),
                dtype=getattr(value, "dtype", None),
                memory=self._memory_str(value),
                **details,
            )

    def _collect_items(self) -> list[VarInfo]:
        items: list[VarInfo] = []
        for name, value in self.namespace.items():
            if name.startswith("__"):
                continue
            if callable(value):
                continue
            items.append(
                VarInfo(
                    name=name,
                    type_name=type(value).__name__,
                    shape=self._shape_str(value),
                    preview=self._preview_str(value),
                )
            )
        items.sort(key=lambda x: x.name.lower())
        return items

    def _shape_str(self, value: Any) -> str:
        if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
            return str(value.shape)
        if isinstance(value, (list, tuple, dict, set, str, bytes)):
            return f"len={len(value)}"
        return "-"

    def _preview_str(self, value: Any, max_len: int = 140) -> str:
        try:
            if isinstance(value, pd.DataFrame):
                preview = value.head(3).to_string()
            elif isinstance(value, pd.Series):
                preview = value.head(5).to_string()
            elif isinstance(value, np.ndarray):
                preview = np.array2string(value, threshold=12, edgeitems=3)
            elif isinstance(value, dict):
                preview = repr(dict(list(value.items())[:5]))
            elif isinstance(value, (list, tuple, set)):
                preview = repr(list(value)[:8])
            else:
                preview = repr(value)
        except Exception as exc:
            preview = f"<Preview error: {exc}>"
        if len(preview) > max_len:
            preview = preview[: max_len - 3] + "..."
        return preview

    def _memory_str(self, value: Any) -> str | None:
        try:
            if isinstance(value, pd.DataFrame):
                mem = int(value.memory_usage(deep=True).sum())
            elif isinstance(value, pd.Series):
                mem = int(value.memory_usage(deep=True))
            elif isinstance(value, np.ndarray):
                mem = int(value.nbytes)
            else:
                return None
            return self._human_bytes(mem)
        except Exception:
            return None

    def _build_details(self, value: Any) -> dict[str, Any]:
        table_html = None
        stats_html = None
        plot_b64 = None
        text_preview = None

        if isinstance(value, pd.DataFrame):
            table_html = value.head(200).to_html(classes="dataframe", border=0)
            try:
                desc = value.describe(include="all").transpose().head(50)
                stats_html = desc.to_html(border=0)
            except Exception:
                pass

        elif isinstance(value, pd.Series):
            table_html = value.head(200).to_frame(name=value.name or "value").to_html(border=0)
            try:
                stats_html = value.describe().to_frame(name="value").to_html(border=0)
            except Exception:
                pass
            plot_b64 = self._series_plot(value)

        elif isinstance(value, np.ndarray):
            text_preview = np.array2string(value, threshold=200, edgeitems=10)
            stats_html = self._ndarray_stats_html(value)
            plot_b64 = self._ndarray_plot(value)

        elif isinstance(value, dict):
            try:
                df = pd.DataFrame(list(value.items()), columns=["key", "value"])
                table_html = df.head(200).to_html(border=0)
            except Exception:
                text_preview = repr(value)

        elif isinstance(value, (list, tuple, set)):
            try:
                df = pd.DataFrame({"value": list(value)})
                table_html = df.head(200).to_html(border=0)
                if df["value"].map(lambda x: isinstance(x, (int, float, np.number))).all():
                    s = pd.Series(df["value"], name="value")
                    stats_html = s.describe().to_frame().to_html(border=0)
                    plot_b64 = self._series_plot(s)
            except Exception:
                text_preview = repr(value)

        else:
            text_preview = repr(value)

        return {
            "table_html": table_html,
            "stats_html": stats_html,
            "plot_b64": plot_b64,
            "text_preview": text_preview,
        }

    def _ndarray_stats_html(self, arr: np.ndarray) -> str:
        if arr.size == 0:
            return "<p>Leeres Array</p>"
        if not np.issubdtype(arr.dtype, np.number):
            return pd.DataFrame({"info": [f"Nicht numerisches Array mit dtype={arr.dtype}"]}).to_html(border=0)
        flat = arr.ravel()
        stats = pd.DataFrame(
            {
                "metric": ["size", "min", "max", "mean", "std", "nan_count"],
                "value": [
                    int(flat.size),
                    float(np.nanmin(flat)),
                    float(np.nanmax(flat)),
                    float(np.nanmean(flat)),
                    float(np.nanstd(flat)),
                    int(np.isnan(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else 0,
                ],
            }
        )
        return stats.to_html(index=False, border=0)

    def _series_plot(self, s: pd.Series) -> str | None:
        try:
            if not pd.api.types.is_numeric_dtype(s):
                return None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(s.dropna().values, bins=30)
            ax.set_title("Histogramm")
            ax.set_xlabel(s.name or "value")
            ax.set_ylabel("frequency")
            return self._fig_to_b64(fig)
        except Exception:
            return None

    def _ndarray_plot(self, arr: np.ndarray) -> str | None:
        try:
            if not np.issubdtype(arr.dtype, np.number):
                return None
            fig, ax = plt.subplots(figsize=(8, 4))
            if arr.ndim == 1:
                ax.hist(arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr, bins=30)
                ax.set_title("Histogramm")
            elif arr.ndim == 2:
                im = ax.imshow(arr, aspect="auto")
                fig.colorbar(im, ax=ax)
                ax.set_title("Heatmap")
            else:
                flat = arr.ravel()
                ax.hist(flat[~np.isnan(flat)] if np.issubdtype(flat.dtype, np.floating) else flat, bins=30)
                ax.set_title("Histogramm der flatten()-Werte")
            return self._fig_to_b64(fig)
        except Exception:
            return None

    def _fig_to_b64(self, fig: plt.Figure) -> str:
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _human_bytes(self, n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(n)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.2f} {unit}"
            value /= 1024
        return f"{n} B"

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        open_browser: bool = True,
        debug: bool = False,
    ) -> None:
        if open_browser:
            threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)



def explore(namespace: Mapping[str, Any], host: str = "127.0.0.1", port: int = 5001) -> None:
    """
    Starte den Mini Variable Explorer fuer den uebergebenen Namespace.

    Beispiel:
        explore(locals())
    """
    explorer = VariableExplorer(namespace)
    explorer.run(host=host, port=port)


if __name__ == "__main__":
    # Demo-Daten
    rng = np.random.default_rng(42)
    demo_df = pd.DataFrame(
        {
            "x": rng.normal(size=100),
            "y": rng.integers(0, 10, size=100),
            "label": rng.choice(["A", "B", "C"], size=100),
        }
    )
    demo_arr = rng.normal(size=(20, 30))
    demo_counts = pd.Series(rng.poisson(3.5, size=200), name="counts")
    demo_dict = {"alpha": 1.2, "beta": [1, 2, 3], "name": "quadtree"}

    explore(
        {
            "demo_df": demo_df,
            "demo_arr": demo_arr,
            "demo_counts": demo_counts,
            "demo_dict": demo_dict,
            "some_text": "Hallo aus dem Mini Variable Explorer",
            "number": 42,
        }
    )
