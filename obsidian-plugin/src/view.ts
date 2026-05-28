// The sidebar ItemView. Vanilla DOM, no React — keeps the bundle <50KB.

import { ItemView, WorkspaceLeaf, Notice, TFile, normalizePath } from "obsidian";
import { streamResearch } from "./api";
import type { StreamEvent, FinalEvent, NodeCompleteEvent, SourcePayload } from "./types";
import type ResearGentPlugin from "./main";

export const VIEW_TYPE_RESEARGENT = "researgent-sidebar";

export class ResearGentView extends ItemView {
  plugin: ResearGentPlugin;

  private qBox!: HTMLTextAreaElement;
  private goBtn!: HTMLButtonElement;
  private saveBtn!: HTMLButtonElement;
  private cancelBtn!: HTMLButtonElement;
  private traceEl!: HTMLDivElement;
  private answerEl!: HTMLDivElement;
  private metaEl!: HTMLDivElement;
  private sourcesEl!: HTMLDivElement;

  private currentRun: AbortController | null = null;
  private lastFinal: FinalEvent | null = null;

  constructor(leaf: WorkspaceLeaf, plugin: ResearGentPlugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType() { return VIEW_TYPE_RESEARGENT; }
  getDisplayText() { return "ResearGent"; }
  getIcon() { return "book-open"; }

  async onOpen() {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("researgent-root");

    // ---- Header ----
    const header = root.createDiv({ cls: "researgent-header" });
    header.createEl("h3", { text: "ResearGent" });
    const sub = header.createEl("div", { cls: "researgent-sub" });
    sub.setText("Plan / retrieve / critique / web / reflect");

    // ---- Input row ----
    const form = root.createDiv({ cls: "researgent-form" });
    this.qBox = form.createEl("textarea", {
      attr: { placeholder: "Ask anything. Cmd/Ctrl+Enter to run." },
    });
    const btnRow = form.createDiv({ cls: "researgent-btnrow" });
    this.goBtn = btnRow.createEl("button", { text: "Research" });
    this.goBtn.addClass("mod-cta");
    this.cancelBtn = btnRow.createEl("button", { text: "Cancel" });
    this.cancelBtn.disabled = true;

    this.qBox.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        void this.runFromInput();
      }
    });
    this.goBtn.addEventListener("click", () => void this.runFromInput());
    this.cancelBtn.addEventListener("click", () => this.cancelRun());

    // ---- Live trace ----
    const traceWrap = root.createDiv({ cls: "researgent-section" });
    traceWrap.createEl("h4", { text: "Trace" });
    this.traceEl = traceWrap.createDiv({ cls: "researgent-trace researgent-empty" });
    this.traceEl.setText("Each agent node fires here as it runs.");

    // ---- Answer + meta ----
    const ansWrap = root.createDiv({ cls: "researgent-section" });
    ansWrap.createEl("h4", { text: "Answer" });
    this.metaEl = ansWrap.createDiv({ cls: "researgent-meta" });
    this.metaEl.hide();
    this.answerEl = ansWrap.createDiv({ cls: "researgent-answer researgent-empty" });
    this.answerEl.setText("Ask a question to begin.");

    const ansActions = ansWrap.createDiv({ cls: "researgent-btnrow" });
    this.saveBtn = ansActions.createEl("button", { text: "Save answer as note" });
    this.saveBtn.disabled = true;
    this.saveBtn.addEventListener("click", () => void this.saveCurrentAnswer());

    // ---- Sources ----
    const srcWrap = root.createDiv({ cls: "researgent-section" });
    srcWrap.createEl("h4", { text: "Sources" });
    this.sourcesEl = srcWrap.createDiv({ cls: "researgent-sources researgent-empty" });
    this.sourcesEl.setText("Citations appear here once the agent finishes.");
  }

  async onClose() {
    this.cancelRun();
  }

  // -----------------------------------------------------------------
  // Public — command palette entry points use these.
  // -----------------------------------------------------------------

  setQuestion(q: string) {
    this.qBox.value = q;
  }

  async runQuestion(question: string, context?: string) {
    if (!question.trim()) return;
    this.qBox.value = question;
    await this.run(question, context);
  }

  // -----------------------------------------------------------------
  // Internals
  // -----------------------------------------------------------------

  private async runFromInput() {
    const q = this.qBox.value.trim();
    if (!q) return;
    await this.run(q);
  }

  private async run(question: string, context?: string) {
    this.cancelRun();
    this.resetPanels();
    this.goBtn.disabled = true;
    this.cancelBtn.disabled = false;
    this.saveBtn.disabled = true;
    this.appendTrace("run_started", `question="${question.slice(0, 80)}"`, "start");

    this.currentRun = streamResearch(this.plugin.settings.backendUrl, question, {
      k: this.plugin.settings.defaultK,
      context: context,
      onEvent: (e) => this.onEvent(e),
      onError: (msg) => {
        this.appendTrace("error", msg, "err");
        this.finish();
      },
      onDone: () => this.finish(),
    });
  }

  private cancelRun() {
    if (this.currentRun) {
      this.currentRun.abort();
      this.currentRun = null;
    }
    this.goBtn.disabled = false;
    this.cancelBtn.disabled = true;
  }

  private finish() {
    this.currentRun = null;
    this.goBtn.disabled = false;
    this.cancelBtn.disabled = true;
  }

  private resetPanels() {
    this.traceEl.empty();
    this.traceEl.removeClass("researgent-empty");
    this.answerEl.empty();
    this.answerEl.addClass("researgent-empty");
    this.answerEl.setText("Streaming…");
    this.sourcesEl.empty();
    this.sourcesEl.addClass("researgent-empty");
    this.sourcesEl.setText("Citations appear when the agent finishes.");
    this.metaEl.empty();
    this.metaEl.hide();
    this.lastFinal = null;
  }

  private onEvent(e: StreamEvent) {
    switch (e.type) {
      case "run_started":
        this.appendTrace("run_started", `run_id=${e.run_id}`, "start");
        break;
      case "node_complete":
        this.renderNode(e);
        break;
      case "final":
        this.renderFinal(e);
        this.appendTrace("done", `${e.sources?.length ?? 0} sources, ${e.answer?.length ?? 0} chars`, "ok");
        break;
      case "error":
        this.appendTrace("error", e.error, "err");
        break;
    }
  }

  private renderNode(e: NodeCompleteEvent) {
    const s = (e.summary || {}) as Record<string, unknown>;
    let detail = "";
    let cls = "ok";
    switch (e.node) {
      case "planner":
        detail = `is_complex=${s.is_complex}  sub_q=${(s.sub_questions as unknown[] | undefined)?.length ?? 0}`;
        break;
      case "retriever":
        detail = `chunks=${(s.total_chunks ?? "?") as string | number}`;
        break;
      case "critic": {
        const g = (s.grades || {}) as Record<string, number>;
        detail = `verdict=${s.confidence}  in=${s.chunks_in} kept=${s.chunks_kept}  (rel=${g.relevant ?? 0} part=${g.partial ?? 0} irr=${g.irrelevant ?? 0})`;
        if (s.confidence === "low") cls = "warn";
        break;
      }
      case "rewriter":
        detail = `attempt=${s.rewrite_attempt}  rewrote=${s.rewritten_count}`;
        cls = "warn";
        break;
      case "paper_discovery":
        detail = `results=${(s.results as number | undefined) ?? 0}`;
        cls = "warn";
        break;
      case "web_fallback": {
        const providers = (s.providers_used as string[] | undefined) || [];
        detail = `+${s.web_chunks_added ?? 0} chunks  providers=[${providers.join(",")}]`;
        cls = "warn";
        break;
      }
      case "generator":
        detail = `${s.answer_chars} chars  n_sources=${s.n_sources}`;
        break;
      case "reflector": {
        const fu = (s.follow_ups as unknown[] | undefined) || [];
        detail = `attempt=${s.attempt}  gaps_found=${s.gaps_found}  follow_ups=${fu.length}`;
        break;
      }
      case "no_answer":
        detail = `reason=${s.reason}`;
        cls = "err";
        break;
      case "llm_reasoning":
        detail = `chars=${s.answer_chars}  (NO SOURCES)`;
        cls = "warn";
        break;
    }
    this.appendTrace(e.node, detail, cls);
  }

  private appendTrace(label: string, detail: string, cls: string) {
    const step = this.traceEl.createDiv({ cls: `researgent-step researgent-${cls}` });
    step.createDiv({ cls: "researgent-step-name", text: label });
    step.createDiv({ cls: "researgent-step-detail", text: detail });
    this.traceEl.scrollTop = this.traceEl.scrollHeight;
  }

  private renderFinal(e: FinalEvent) {
    this.lastFinal = e;
    this.saveBtn.disabled = false;

    // ---- Meta chips ----
    this.metaEl.empty();
    this.metaEl.show();
    const chip = (text: string, cls = "") => {
      const c = this.metaEl.createSpan({ cls: `researgent-chip ${cls}`.trim(), text });
      return c;
    };
    if (e.confidence) chip(`conf=${e.confidence}`);
    if (e.rewrite_attempts) chip(`rewrites=${e.rewrite_attempts}`, "warn");
    if (e.web_used) chip("web=YES", "web");
    if (e.reflection_attempts) chip(`reflections=${e.reflection_attempts}`);
    if (e.error === "no_sources_used_llm_priors") chip("LLM_PRIORS_ONLY", "err");

    // ---- Answer body ----
    this.answerEl.empty();
    this.answerEl.removeClass("researgent-empty");
    // Render simple ## sections as <h4> for readability; keep [Sn] highlights.
    const html = (e.answer || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/^## (.+)$/gm, '<h4 class="researgent-h">$1</h4>')
      .replace(/\[(S\d+)\]/g, '<span class="researgent-cite">[$1]</span>')
      .replace(/\n/g, "<br>");
    this.answerEl.innerHTML = html;

    // ---- Sources ----
    this.sourcesEl.empty();
    this.sourcesEl.removeClass("researgent-empty");
    for (const s of e.sources || []) {
      this.sourcesEl.appendChild(this.renderSource(s));
    }
    if ((e.sources || []).length === 0) {
      const empty = this.sourcesEl.createDiv({ cls: "researgent-empty" });
      empty.setText("(no sources — LLM priors only)");
    }
  }

  private renderSource(s: SourcePayload): HTMLDivElement {
    const div = createDiv({ cls: "researgent-source" });
    const head = div.createDiv({ cls: "researgent-source-head" });
    head.createSpan({ cls: "researgent-source-tag", text: `[${s.tag}]` });

    const citeWrap = head.createSpan({ cls: "researgent-source-cite" });
    // If citation looks like a URL, render an anchor that opens externally.
    // If it looks like a vault note (".md" suffix), render as an Obsidian
    // internal-link click target.
    const cit = s.citation || "";
    if (cit.startsWith("http://") || cit.startsWith("https://")) {
      const a = citeWrap.createEl("a", { text: cit, attr: { href: cit } });
      a.setAttr("target", "_blank");
    } else if (/\.md(\s|$|\s+p\.\d+)/i.test(cit)) {
      const noteName = cit.replace(/\s+p\.\d+\s*$/i, "");
      const a = citeWrap.createEl("a", { text: cit });
      a.addEventListener("click", (ev) => {
        ev.preventDefault();
        const file = this.app.metadataCache.getFirstLinkpathDest(noteName.replace(/\.md$/i, ""), "");
        if (file instanceof TFile) {
          void this.app.workspace.openLinkText(file.path, "", false);
        } else {
          new Notice(`Note not found in vault: ${noteName}`);
        }
      });
    } else if (cit.startsWith("arxiv:")) {
      const id = cit.slice("arxiv:".length);
      const a = citeWrap.createEl("a", {
        text: cit,
        attr: { href: `https://arxiv.org/abs/${id}` },
      });
      a.setAttr("target", "_blank");
    } else {
      citeWrap.setText(cit);
    }

    head.createSpan({
      cls: `researgent-signal researgent-signal-${(s.signal || "").replace(/[^A-Za-z]/g, "")}`,
      text: s.signal || "?",
    });

    const preview = div.createDiv({ cls: "researgent-source-preview" });
    preview.setText(s.preview || "");
    return div;
  }

  // -----------------------------------------------------------------
  // Save → write the current answer back into the vault as a note.
  // Uses Obsidian's own vault API — no HTTP needed.
  // -----------------------------------------------------------------

  private async saveCurrentAnswer() {
    if (!this.lastFinal) return;
    const e = this.lastFinal;
    const folder = this.plugin.settings.outputFolder.replace(/\/+$/, "") || "ResearGent";
    const date = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const dir = normalizePath(`${folder}/${date}`);

    // Ensure the folder exists.
    if (!(await this.app.vault.adapter.exists(dir))) {
      await this.app.vault.createFolder(dir);
    }

    // Filesystem-safe filename.
    const slug = (this.qBox.value || "untitled")
      .replace(/[<>:"/\\|?*\x00-\x1f]+/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 60) || "untitled";

    // Resolve a unique path.
    let target = normalizePath(`${dir}/${slug}.md`);
    let counter = 2;
    while (await this.app.vault.adapter.exists(target)) {
      target = normalizePath(`${dir}/${slug} (${counter}).md`);
      counter += 1;
    }

    const note = buildNoteMarkdown(this.qBox.value, e);
    const file = await this.app.vault.create(target, note);
    new Notice(`ResearGent: saved to ${target}`);

    if (this.plugin.settings.autoOpenSavedNote) {
      const leaf = this.app.workspace.getLeaf(false);
      await leaf.openFile(file);
    }
  }
}

// ---------- helpers ----------

function buildNoteMarkdown(question: string, e: FinalEvent): string {
  const date = new Date().toISOString().slice(0, 10);
  const tags = ["researgent"];
  if (e.confidence === "low") tags.push("low-confidence");
  if (!(e.sources || []).length) tags.push("no-sources");

  const fm = [
    "---",
    `title: ${yamlStr(question)}`,
    `date: ${yamlStr(date)}`,
    "source: ResearGent",
    `run_id: ${yamlStr(e.run_id)}`,
    `confidence: ${yamlStr(e.confidence || "unknown")}`,
    `rewrites: ${e.rewrite_attempts || 0}`,
    `web_used: ${e.web_used ? "true" : "false"}`,
    `reflections: ${e.reflection_attempts || 0}`,
    `n_sources: ${(e.sources || []).length}`,
    `tags: [${tags.map(yamlStr).join(", ")}]`,
    "---",
    "",
  ].join("\n");

  let body = `# ${question}\n\n`;
  if (e.is_complex && (e.sub_questions || []).length > 1) {
    body += "**Decomposed into sub-questions:**\n";
    for (const sq of e.sub_questions) body += `- ${sq}\n`;
    body += "\n";
  }
  body += (e.answer || "_(no answer produced)_") + "\n\n";

  if ((e.sources || []).length) {
    body += "## Sources\n\n";
    for (const s of e.sources) body += renderSourceLine(s) + "\n";
    body += "\n";
  }

  body += "## Provenance\n\n```yaml\n";
  body += `run_id:        ${e.run_id}\n`;
  body += `confidence:    ${e.confidence}\n`;
  body += `rewrites:      ${e.rewrite_attempts}\n`;
  body += `web_fallback:  ${e.web_used}\n`;
  body += `reflections:   ${e.reflection_attempts}\n`;
  body += `sources_total: ${(e.sources || []).length}\n`;
  body += "```\n";

  return fm + body;
}

function renderSourceLine(s: SourcePayload): string {
  const cit = s.citation || "";
  if (cit.startsWith("http://") || cit.startsWith("https://")) {
    return `- **[${s.tag}]** [${cit}](${cit})`;
  }
  if (cit.startsWith("arxiv:")) {
    const id = cit.slice("arxiv:".length);
    return `- **[${s.tag}]** [${cit}](https://arxiv.org/abs/${id})`;
  }
  // Vault note: "Foo.md p.0" → [[Foo]] wikilink
  const m = cit.match(/^(.+?)\.md(?:\s+p\.\d+)?\s*$/i);
  if (m) {
    const stem = m[1].split(/[\\/]/).pop() || m[1];
    return `- **[${s.tag}]** [[${stem}]]`;
  }
  return `- **[${s.tag}]** \`${cit}\``;
}

function yamlStr(s: string): string {
  if (!/[:#\-\[\]{},]/.test(s) && !s.startsWith(" ")) return s;
  return `"${s.replace(/"/g, '\\"')}"`;
}
