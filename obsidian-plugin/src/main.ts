// ResearGent — Obsidian plugin entrypoint.
//
// Wires up:
//   - the sidebar view (`ResearGentView`)
//   - 3 command-palette commands
//   - a ribbon icon
//   - the settings tab
//   - load/save of plugin settings

import { Plugin, WorkspaceLeaf, Notice, MarkdownView, Editor } from "obsidian";
import { ResearGentView, VIEW_TYPE_RESEARGENT } from "./view";
import { ResearGentSettingTab } from "./settings";
import { DEFAULT_SETTINGS, type ResearGentSettings } from "./types";

export default class ResearGentPlugin extends Plugin {
  settings!: ResearGentSettings;

  async onload() {
    await this.loadSettings();

    this.registerView(
      VIEW_TYPE_RESEARGENT,
      (leaf: WorkspaceLeaf) => new ResearGentView(leaf, this),
    );

    this.addRibbonIcon("book-open", "Open ResearGent", () => {
      void this.activateView();
    });

    this.addCommand({
      id: "open-sidebar",
      name: "Open ResearGent sidebar",
      callback: () => {
        void this.activateView();
      },
    });

    this.addCommand({
      id: "research-current-note",
      name: "Research the current note",
      checkCallback: (checking) => {
        const file = this.app.workspace.getActiveFile();
        if (!file) return false;
        if (checking) return true;
        void this.researchAboutCurrentNote();
        return true;
      },
    });

    this.addCommand({
      id: "research-selection",
      name: "Research the current selection",
      editorCheckCallback: (checking, editor: Editor) => {
        const sel = editor.getSelection().trim();
        if (!sel) return false;
        if (checking) return true;
        void this.researchSelection(sel);
        return true;
      },
    });

    this.addSettingTab(new ResearGentSettingTab(this.app, this));
  }

  onunload() {
    // Detaching the view leaves the user's panel layout intact next time
    // they enable the plugin. Obsidian handles the actual detach on disable.
  }

  // ---- Persistence ----

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  // ---- View management ----

  async activateView(): Promise<ResearGentView | null> {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = null;
    const existing = workspace.getLeavesOfType(VIEW_TYPE_RESEARGENT);
    if (existing.length) {
      leaf = existing[0];
    } else {
      leaf = workspace.getRightLeaf(false);
      if (!leaf) {
        // Fallback to root if the right sidebar isn't available (rare).
        leaf = workspace.getLeaf(true);
      }
      await leaf.setViewState({ type: VIEW_TYPE_RESEARGENT, active: true });
    }
    workspace.revealLeaf(leaf);
    return leaf.view instanceof ResearGentView ? leaf.view : null;
  }

  // ---- Commands ----

  async researchAboutCurrentNote() {
    const file = this.app.workspace.getActiveFile();
    if (!file) {
      new Notice("ResearGent: no active note");
      return;
    }
    const view = await this.activateView();
    if (!view) return;

    let context = "";
    if (this.settings.includeActiveNote) {
      try {
        context = await this.app.vault.cachedRead(file);
      } catch {
        context = "";
      }
    }

    // Default prompt — user can tweak in the textarea before re-running.
    const question = `Research this note: "${file.basename}". Summarize it, find related concepts, and surface any open questions or contradictions.`;
    await view.runQuestion(question, context || undefined);
  }

  async researchSelection(selection: string) {
    const view = await this.activateView();
    if (!view) return;
    const trimmed = selection.length > 400 ? selection.slice(0, 400) + "…" : selection;
    const question = `What does this passage describe, and what's the most relevant follow-up to research?\n\n"${trimmed}"`;
    await view.runQuestion(question);
  }
}
