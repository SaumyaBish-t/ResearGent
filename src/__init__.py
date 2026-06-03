"""ResearGent — Agentic Research Engine with Corrective RAG & Self-Reflection."""

# ---------------------------------------------------------------------------
# Windows / Python 3.12 WMI hang guard  (must run before torch is imported).
#
# torch's import path calls platform.machine() -> platform.win32_ver() ->
# platform._wmi_query(), which issues a WMI (Windows Management Instrumentation)
# query for OS metadata. When the Windows WMI service is hung or slow, that
# query BLOCKS indefinitely instead of raising — so `import torch` (and hence
# `researgent serve`, which imports the agent graph -> sentence-transformers ->
# torch) hangs at startup with no output and never binds the port.
#
# platform.win32_ver() already has a non-WMI fallback that triggers on OSError.
# We force _wmi_query to fail fast so that fallback runs. torch only needs the
# CPU architecture (from the PROCESSOR_ARCHITECTURE env var), so there is no
# functional impact; on non-Windows / healthy-WMI systems this is a no-op.
# ---------------------------------------------------------------------------
import platform as _platform

if hasattr(_platform, "_wmi_query"):

    def _researgent_wmi_disabled(*_args, **_kwargs):  # pragma: no cover (win-only)
        raise OSError("WMI query disabled by ResearGent (avoids torch import hang)")

    _platform._wmi_query = _researgent_wmi_disabled

__version__ = "0.1.0"
