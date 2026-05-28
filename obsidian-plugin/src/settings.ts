// Settings tab — standard Obsidian PluginSettingTab.

import { App, PluginSettingTab, Setting, Notice } from "obsidian";
import { pingBackend } from "./api";
import type ResearGentPlugin from "./main";

export class ResearGentSettingTab extends PluginSettingTab {
  plugin: ResearGentPlugin;

  constructor(app: App, plugin: ResearGentPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl("h2", { text: "ResearGent" });
    containerEl.createEl("p", {
      text:
        "Sidebar agent that streams a planning / retrieval / critic / web / " +
        "reflection pipeline live, and writes answers back as Obsidian notes " +
        "with wikilink citations. Requires the ResearGent backend running locally.",
      cls: "setting-item-description",
    });

    new Setting(containerEl)
      .setName("Backend URL")
      .setDesc(
        "Where the ResearGent FastAPI server is listening. Default is the " +
          "local `researgent serve` default port.",
      )
      .addText((t) =>
        t
          .setPlaceholder("http://127.0.0.1:8000")
          .setValue(this.plugin.settings.backendUrl)
          .onChange(async (v) => {
            this.plugin.settings.backendUrl = v.trim();
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Test connection")
      .setDesc("Hit /api/status on the backend and show the tier routing.")
      .addButton((b) =>
        b.setButtonText("Ping").onClick(async () => {
          b.setDisabled(true).setButtonText("…");
          const { ok, detail } = await pingBackend(this.plugin.settings.backendUrl);
          b.setDisabled(false).setButtonText("Ping");
          new Notice(`ResearGent: ${ok ? "OK" : "FAIL"} — ${detail}`, ok ? 5000 : 8000);
        }),
      );

    new Setting(containerEl)
      .setName("Default top-k")
      .setDesc("Total retrieved chunks budget shared across sub-questions.")
      .addSlider((s) =>
        s
          .setLimits(2, 20, 1)
          .setValue(this.plugin.settings.defaultK)
          .setDynamicTooltip()
          .onChange(async (v) => {
            this.plugin.settings.defaultK = v;
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Output folder")
      .setDesc(
        "Subfolder inside the vault where saved answers live. A " +
          "`<folder>/<YYYY-MM-DD>/` date subfolder is added automatically.",
      )
      .addText((t) =>
        t
          .setPlaceholder("ResearGent")
          .setValue(this.plugin.settings.outputFolder)
          .onChange(async (v) => {
            this.plugin.settings.outputFolder = v.trim() || "ResearGent";
            await this.plugin.saveSettings();
          }),
      );

    new Setting(containerEl)
      .setName("Use active note as context")
      .setDesc(
        "When you run 'Research current note', send the note's body as " +
          "additional context to the agent. Disable for pure question-answering.",
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.includeActiveNote).onChange(async (v) => {
          this.plugin.settings.includeActiveNote = v;
          await this.plugin.saveSettings();
        }),
      );

    new Setting(containerEl)
      .setName("Auto-open saved notes")
      .setDesc("After 'Save answer to vault', immediately open the new note.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.autoOpenSavedNote).onChange(async (v) => {
          this.plugin.settings.autoOpenSavedNote = v;
          await this.plugin.saveSettings();
        }),
      );
  }
}
