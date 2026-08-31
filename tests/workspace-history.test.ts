import { access, copyFile, mkdtemp, mkdir, readFile, readdir, rm, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import {
  type CustomEntry,
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import workspaceHistoryExtension, {
  rebuildTurnSnapshotsFromLegacyEntries,
  isWindowsReservedSnapshotPath,
} from "../.pi/extensions/workspace-history.ts";
import {
  fauxProvider,
  fauxAssistantMessage,
  fauxToolCall,
} from "@earendil-works/pi-ai";

type TestContext = {
  rootDir: string;
  cwd: string;
  resourceLoader: DefaultResourceLoader;
  modelRuntime: ModelRuntime;
  settingsManager: SettingsManager;
  provider: ReturnType<typeof fauxProvider> & { unregister(): void };
};

type TurnSnapshotState = {
  version: 1;
  turns: Array<{
    turnId: string;
    userEntryId: string;
    assistantEntryId: string;
    beforeCommit: string;
    afterCommit: string;
    createdAt: string;
  }>;
};

const execFileAsync = promisify(execFile);

async function createContextForWorkspace(rootDir: string, cwd: string, withProjectMarker = true): Promise<TestContext> {
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false },
    retry: { enabled: false, maxRetries: 0 },
    branchSummary: { skipPrompt: true },
  });

  const provider = Object.assign(
    fauxProvider({
      provider: "timemachine-test",
      api: "faux",
      models: [
        {
          id: "faux-1",
          name: "Timemachine Test Model",
          reasoning: false,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 32000,
          maxTokens: 4096,
        },
      ],
    }),
    { unregister(): void {} },
  );

  const modelRuntime = await ModelRuntime.create({
    modelsPath: null,
    refreshOnCreate: false,
  });
  modelRuntime.registerNativeProvider(provider.provider);

  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir: getAgentDir(),
    settingsManager,
    additionalExtensionPaths: [path.join(process.cwd(), ".pi", "extensions", "workspace-history.ts")],
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    systemPrompt: "Use tools when asked. Keep responses short.",
  });
  await resourceLoader.reload();

  if (withProjectMarker) {
    await writeFile(path.join(cwd, "package.json"), JSON.stringify({ name: "timemachine-test-workspace" }, null, 2) + "\n", "utf8");
  }
  await mkdir(path.join(cwd, ".pi"), { recursive: true });
  await writeFile(
    path.join(cwd, ".pi", "settings.json"),
    JSON.stringify({ workspaceHistory: { storageDir: getWorkspaceHistoryStateDir(rootDir) } }, null, 2) + "\n",
    "utf8",
  );

  return {
    rootDir,
    cwd,
    resourceLoader,
    modelRuntime,
    settingsManager,
    provider,
  };
}

async function createContext(): Promise<TestContext> {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "pi-timemachine-test-"));
  const cwd = path.join(rootDir, "workspace");
  await mkdir(cwd, { recursive: true });
  return createContextForWorkspace(rootDir, cwd, true);
}

async function createNonProjectContext(): Promise<TestContext> {
  const rootDir = await mkdtemp(path.join(os.tmpdir(), "pi-timemachine-test-"));
  const cwd = path.join(rootDir, "workspace");
  await mkdir(cwd, { recursive: true });
  return createContextForWorkspace(rootDir, cwd, false);
}

async function disposeContext(ctx: TestContext): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      await rm(ctx.rootDir, { recursive: true, force: true });
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EBUSY") {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
  }
}

async function createSession(ctx: TestContext, sessionManager: SessionManager = SessionManager.inMemory(ctx.cwd)) {
  const model = ctx.provider.getModel();
  const result = await createAgentSession({
    cwd: ctx.cwd,
    agentDir: getAgentDir(),
    model,
    thinkingLevel: "off",
    modelRuntime: ctx.modelRuntime,
    resourceLoader: ctx.resourceLoader,
    tools: ["read", "write", "edit"],
    sessionManager,
    settingsManager: ctx.settingsManager,
  });

  const session = result.session;
  await session.bindExtensions({
    commandContextActions: {
      waitForIdle: () => session.agent.waitForIdle(),
      newSession: async () => ({ cancelled: true }),
      fork: async () => ({ cancelled: true }),
      navigateTree: async (targetId, options) => {
        const nav = await session.navigateTree(targetId, {
          summarize: options?.summarize,
          customInstructions: options?.customInstructions,
          replaceInstructions: options?.replaceInstructions,
          label: options?.label,
        });
        return { cancelled: nav.cancelled };
      },
      switchSession: async () => ({ cancelled: true }),
      reload: async () => {
        await session.reload();
      },
    },
    onError: (err) => {
      throw new Error(`Extension error (${err.event}): ${err.error}\n${err.stack ?? ""}`);
    },
  });

  return session;
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await readFile(filePath, "utf8");
    return true;
  } catch {
    return false;
  }
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readText(filePath: string): Promise<string> {
  return readFile(filePath, "utf8");
}

function normalizeEol(text: string): string {
  return text.replace(/\r\n/g, "\n");
}

function getWorkspaceHistoryStateDir(rootDir: string): string {
  return path.join(rootDir, "workspace-history-state");
}

async function waitFor(condition: () => boolean | Promise<boolean>, message: string, timeoutMs = 5000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(message);
}

async function waitForExists(filePath: string, expected: boolean, message: string): Promise<void> {
  await waitFor(async () => (await exists(filePath)) === expected, message);
}

async function waitForText(filePath: string, expected: string, message: string): Promise<void> {
  await waitFor(async () => {
    try {
      return normalizeEol(await readText(filePath)) === expected;
    } catch {
      return false;
    }
  }, message);
}

async function holdWindowsFileWithoutDeleteSharing(
  filePath: string,
  tempDir: string,
  options: {
    releaseAfterMs?: number;
    mutateAfterMs?: number;
    mutateContent?: string;
  } = {},
): Promise<() => Promise<void>> {
  const scriptPath = path.join(tempDir, "hold-file-lock.ps1");
  const releaseSignalPath = path.join(tempDir, "release-file-lock.signal");
  await writeFile(scriptPath, [
    "param([string]$filePath, [string]$releaseMode, [string]$releaseSignalPath, [string]$mutateAfterMs, [string]$mutateContentBase64)",
    "$access = if ($mutateAfterMs -ne 'none') { [System.IO.FileAccess]::ReadWrite } else { [System.IO.FileAccess]::Read }",
    "$handle = [System.IO.File]::Open($filePath, [System.IO.FileMode]::Open, $access, [System.IO.FileShare]::ReadWrite)",
    "[Console]::Out.WriteLine('READY')",
    "[Console]::Out.Flush()",
    "if ($mutateAfterMs -ne 'none') {",
    "  Start-Sleep -Milliseconds ([int]$mutateAfterMs)",
    "  $content = [Convert]::FromBase64String($mutateContentBase64)",
    "  $handle.Position = 0",
    "  $handle.SetLength(0)",
    "  $handle.Write($content, 0, $content.Length)",
    "  $handle.Flush()",
    "}",
    "if ($releaseMode -eq 'signal') { while (-not (Test-Path -LiteralPath $releaseSignalPath)) { Start-Sleep -Milliseconds 25 } } else { Start-Sleep -Milliseconds ([int]$releaseMode) }",
    "$handle.Dispose()",
  ].join("\n"), "utf8");

  const windowsRoot = process.env.SystemRoot ?? process.env.WINDIR ?? "C:\\Windows";
  const powershellExe = path.join(windowsRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const child = spawn(powershellExe, [
    "-NoLogo",
    "-NoProfile",
    "-File",
    scriptPath,
    filePath,
    options.releaseAfterMs === undefined ? "signal" : String(options.releaseAfterMs),
    releaseSignalPath,
    options.mutateAfterMs === undefined ? "none" : String(options.mutateAfterMs),
    Buffer.from(options.mutateContent ?? "none", "utf8").toString("base64"),
  ], {
    stdio: "pipe",
    windowsHide: true,
  });
  let stderr = "";
  let ready = false;
  const exited = new Promise<void>((resolve, reject) => {
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.once("error", reject);
    child.once("exit", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`file lock process exited with ${String(code)}: ${stderr}`));
      }
    });
  });
  void exited.catch(() => undefined);
  try {
    await new Promise<void>((resolve, reject) => {
      let stdout = "";
      const timeout = setTimeout(() => reject(new Error("timed out waiting for the Windows file lock helper")), 5_000);
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
        if (!ready && stdout.includes("READY")) {
          ready = true;
          clearTimeout(timeout);
          resolve();
        }
      });
      void exited.catch((error) => {
        if (!ready) {
          clearTimeout(timeout);
          reject(error);
        }
      });
    });
  } catch (error) {
    if (child.exitCode === null) {
      child.kill();
    }
    await exited.catch(() => undefined);
    throw error;
  }

  return async () => {
    if (child.exitCode === null && options.releaseAfterMs === undefined) {
      await writeFile(releaseSignalPath, "release\n", "utf8");
    }
    await exited;
  };
}

function getSessionHistoryDir(session: Awaited<ReturnType<typeof createSession>>, cwd: string): string {
  const workspaceHash = createHash("sha256").update(path.normalize(cwd)).digest("hex").slice(0, 24);
  return path.join(
    getWorkspaceHistoryStateDir(path.dirname(cwd)),
    "workspaces",
    workspaceHash,
    "sessions",
    session.sessionManager.getSessionId(),
  );
}

function getTurnSnapshotFile(session: Awaited<ReturnType<typeof createSession>>, cwd: string): string {
  return path.join(getSessionHistoryDir(session, cwd), "turn-snapshots.json");
}

function getPendingRecoveryFile(session: Awaited<ReturnType<typeof createSession>>, cwd: string): string {
  return path.join(getSessionHistoryDir(session, cwd), "pending-recovery.json");
}

function getShadowGitDir(session: Awaited<ReturnType<typeof createSession>>, cwd: string): string {
  return path.join(getSessionHistoryDir(session, cwd), "repo.git");
}

function shadowGitArgs(
  session: Awaited<ReturnType<typeof createSession>>,
  cwd: string,
  ...args: string[]
): string[] {
  return [
    "-c",
    "i18n.logOutputEncoding=utf-8",
    "-c",
    "core.autocrlf=false",
    "-c",
    "core.safecrlf=false",
    "-c",
    "core.filemode=false",
    "-c",
    "core.quotepath=false",
    "--git-dir",
    getShadowGitDir(session, cwd),
    "--work-tree",
    cwd,
    ...args,
  ];
}

async function refreshShadowIndex(session: Awaited<ReturnType<typeof createSession>>, cwd: string): Promise<void> {
  await execFileAsync("git", shadowGitArgs(session, cwd, "add", "-A", "--", "."), { cwd });
}

async function getShadowStatus(session: Awaited<ReturnType<typeof createSession>>, cwd: string): Promise<string> {
  const { stdout } = await execFileAsync("git", shadowGitArgs(
    session,
    cwd,
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
    "--",
    ".",
  ), { cwd });
  return stdout;
}

async function assertValidShadowRepo(gitDir: string, cwd: string, description: string): Promise<void> {
  const bareResult = await execFileAsync("git", ["--git-dir", gitDir, "rev-parse", "--is-bare-repository"], { cwd });
  assert.equal(bareResult.stdout.trim(), "true", `${description} should be bare`);
  const headResult = await execFileAsync("git", ["--git-dir", gitDir, "rev-parse", "--verify", "HEAD^{commit}"], { cwd });
  assert.match(headResult.stdout.trim(), /^[0-9a-f]{40,64}$/i, `${description} should have a resolvable HEAD commit`);
}

function getReusableGitDir(session: Awaited<ReturnType<typeof createSession>>, cwd: string): string {
  return path.join(path.dirname(path.dirname(getSessionHistoryDir(session, cwd))), "repo.git");
}

async function pruneUnreachableGitObjects(gitDir: string, cwd: string): Promise<void> {
  await execFileAsync("git", ["--git-dir", gitDir, "config", "gc.cruftPacks", "false"], { cwd });
  await execFileAsync("git", ["--git-dir", gitDir, "reflog", "expire", "--expire=now", "--all"], { cwd });
  await execFileAsync("git", ["--git-dir", gitDir, "repack", "-Ad"], { cwd });
  await execFileAsync("git", ["--git-dir", gitDir, "prune-packed"], { cwd });
  await execFileAsync("git", ["--git-dir", gitDir, "prune", "--expire=now"], { cwd });
}

async function readTurnSnapshots(session: Awaited<ReturnType<typeof createSession>>, cwd: string): Promise<TurnSnapshotState> {
  try {
    return JSON.parse(await readFile(getTurnSnapshotFile(session, cwd), "utf8")) as TurnSnapshotState;
  } catch {
    return { version: 1, turns: [] };
  }
}

function getMessageText(entry: any): string | undefined {
  if (entry?.type !== "message") {
    return undefined;
  }
  const content = entry.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((item) => item?.type === "text")
      .map((item) => item.text)
      .join("");
  }
  return undefined;
}

async function countSnapshots(session: Awaited<ReturnType<typeof createSession>>, cwd: string, kind?: string): Promise<number> {
  if (kind === "after") {
    return (await readTurnSnapshots(session, cwd)).turns.length;
  }

  return session.sessionManager.getEntries().filter((entry) => {
    return (
      entry.type === "custom" &&
      entry.customType === "workspace-history.snapshot" &&
      (!kind || (entry as any).data?.kind === kind)
    );
  }).length;
}

function findBaselineSnapshot(session: Awaited<ReturnType<typeof createSession>>) {
  return session.sessionManager.getEntries().find((entry) => {
    if (entry.type !== "custom" || entry.customType !== "workspace-history.snapshot") {
      return false;
    }
    const data: unknown = entry.data;
    return typeof data === "object" && data !== null && "kind" in data && data.kind === "baseline";
  });
}

function captureNotifications(session: Awaited<ReturnType<typeof createSession>>): string[] {
  return configureTestUI(session, [], true).notifications;
}

interface TestUIState {
  editorText?: string;
  notifications: string[];
  selections: Array<{ title: string; options: string[] }>;
}

function configureTestUI(
  session: Awaited<ReturnType<typeof createSession>>,
  choices: string[],
  selectFirstByDefault = false,
): TestUIState {
  const state: TestUIState = {
    notifications: [],
    selections: [],
  };
  type UIContext = Parameters<typeof session.extensionRunner.setUIContext>[0];
  const uiContext = new Proxy({
    async select(title: string, options: string[]): Promise<string | undefined> {
      state.selections.push({ title, options: [...options] });
      return choices.shift() ?? (selectFirstByDefault ? options[0] : undefined);
    },
    notify(message: string): void {
      state.notifications.push(message);
    },
    setEditorText(text: string): void {
      state.editorText = text;
    },
  }, {
    get(target, property) {
      return Reflect.get(target, property) ?? (() => undefined);
    },
  }) as UIContext;
  session.extensionRunner.setUIContext(uiContext);
  return state;
}

async function testUndoConversationOnlyKeepsWorkspace(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "kept.txt");
    const ui = configureTestUI(session, ["Conversation only (keep current files)"]);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "kept.txt", content: "keep me\n" })]),
      fauxAssistantMessage("created kept.txt"),
    ]);

    await session.prompt("create kept.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot was not created");

    await session.prompt("/undo");

    assert.equal(await exists(filePath), true, "conversation-only undo should keep workspace files");
    assert.equal(normalizeEol(await readText(filePath)), "keep me\n");
    assert.equal(ui.editorText, "create kept.txt", "undo should restore the original prompt to the editor");
    assert.deepEqual(
      ui.selections,
      [{
        title: "Undo",
        options: [
          "Conversation and workspace",
          "Conversation only (keep current files)",
        ],
      }],
    );

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testRedoReusesConversationOnlyMode(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "redo-kept.txt");
    const ui = configureTestUI(session, ["Conversation only (keep current files)"]);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "redo-kept.txt", content: "kept through redo\n" })]),
      fauxAssistantMessage("created redo-kept.txt"),
    ]);

    await session.prompt("create redo-kept.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot was not created");

    await session.prompt("/undo");
    await session.prompt("/redo");

    assert.equal(normalizeEol(await readText(filePath)), "kept through redo\n");
    assert.equal(
      session.sessionManager.getBranch().some((entry) => {
        return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created redo-kept.txt";
      }),
      true,
      "redo should restore the conversation branch",
    );
    assert.equal(ui.selections.length, 1, "redo should reuse the undo mode without prompting again");
    assert.equal(
      ui.notifications.includes("Redo complete. Conversation restored; current files kept."),
      true,
    );

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testNavigationChoiceCancellationKeepsConversationAndWorkspace(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "cancelled-navigation.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "cancelled-navigation.txt", content: "unchanged\n" })]),
      fauxAssistantMessage("created cancelled-navigation.txt"),
    ]);

    await session.prompt("create cancelled-navigation.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot was not created");
    const originalLeafId = session.sessionManager.getLeafId();
    const userEntry = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "user" && getMessageText(entry) === "create cancelled-navigation.txt";
    });
    assert.ok(originalLeafId && userEntry, "cancel fixture history entries should exist");

    const undoUI = configureTestUI(session, []);
    await session.prompt("/undo");
    assert.equal(session.sessionManager.getLeafId(), originalLeafId, "cancelled undo should keep the conversation leaf");
    assert.equal(normalizeEol(await readText(filePath)), "unchanged\n");
    assert.equal(undoUI.editorText, undefined);

    const treeUI = configureTestUI(session, []);
    const treeResult = await session.navigateTree(userEntry.id, { summarize: false });
    assert.equal(treeResult.cancelled, true, "cancelled tree choice should cancel navigation");
    assert.equal(session.sessionManager.getLeafId(), originalLeafId);
    assert.equal(normalizeEol(await readText(filePath)), "unchanged\n");
    assert.deepEqual(treeUI.selections.map(({ title }) => title), ["Tree navigation"]);

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testConversationOnlyUndoPreservesManualChangesAsBranchState(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "manual-kept.txt");
    configureTestUI(session, ["Conversation only (keep current files)"]);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "manual-kept.txt", content: "agent state\n" })]),
      fauxAssistantMessage("created manual-kept.txt"),
    ]);

    await session.prompt("create manual-kept.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot was not created");
    const originalAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created manual-kept.txt";
    });
    assert.ok(originalAssistant, "manual change fixture assistant entry should exist");

    await writeFile(filePath, "manual state\n", "utf8");
    await session.prompt("/undo");
    assert.equal(normalizeEol(await readText(filePath)), "manual state\n");

    ctx.provider.setResponses([fauxAssistantMessage("continued from kept manual state")]);
    await session.prompt("continue from the kept workspace");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "continued branch snapshot was not created");
    const keptBranchAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" &&
        entry.message.role === "assistant" &&
        getMessageText(entry) === "continued from kept manual state";
    });
    assert.ok(keptBranchAssistant, "the kept workspace branch should have a visible assistant node");

    configureTestUI(session, ["Conversation and workspace", "Conversation and workspace"]);
    const originalResult = await session.navigateTree(originalAssistant.id, { summarize: false });
    assert.equal(originalResult.cancelled, false);
    assert.equal(normalizeEol(await readText(filePath)), "agent state\n");

    const keptResult = await session.navigateTree(keptBranchAssistant.id, { summarize: false });
    assert.equal(keptResult.cancelled, false, "manual state should remain reachable through the visible continued branch");
    assert.equal(normalizeEol(await readText(filePath)), "manual state\n");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testTreeConversationOnlyAnchorsKeptWorkspace(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "tree-kept.txt");
    const ui = configureTestUI(session, [
      "Conversation only (keep current files)",
      "Conversation and workspace",
    ]);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "tree-kept.txt", content: "A\n" })]),
      fauxAssistantMessage("created tree state A"),
      fauxAssistantMessage([fauxToolCall("write", { path: "tree-kept.txt", content: "B\n" })]),
      fauxAssistantMessage("created tree state B"),
    ]);

    await session.prompt("create tree state A");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "A snapshot was not created");
    await session.prompt("create tree state B");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "B snapshot was not created");

    const aAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created tree state A";
    });
    const bAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created tree state B";
    });
    assert.ok(aAssistant && bAssistant, "tree fixture assistant messages should exist");

    const conversationOnlyResult = await session.navigateTree(aAssistant.id, { summarize: false });
    assert.equal(conversationOnlyResult.cancelled, false);
    assert.equal(normalizeEol(await readText(filePath)), "B\n", "conversation-only tree navigation should keep B");

    const returnToBResult = await session.navigateTree(bAssistant.id, { summarize: false });
    assert.equal(returnToBResult.cancelled, false, "kept workspace should be anchored to the new conversation branch");
    assert.equal(normalizeEol(await readText(filePath)), "B\n");
    assert.deepEqual(ui.selections.map(({ title }) => title), ["Tree navigation", "Tree navigation"]);

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testTreeConversationOnlySupportsBranchSummary(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "tree-summary-kept.txt");
    const ui = configureTestUI(session, ["Conversation only (keep current files)"]);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "tree-summary-kept.txt", content: "A\n" })]),
      fauxAssistantMessage("created summary state A"),
      fauxAssistantMessage([fauxToolCall("write", { path: "tree-summary-kept.txt", content: "B\n" })]),
      fauxAssistantMessage("created summary state B"),
      fauxAssistantMessage("Summary of the abandoned B branch."),
    ]);

    await session.prompt("create summary state A");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "summary A snapshot was not created");
    await session.prompt("create summary state B");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "summary B snapshot was not created");

    const aAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created summary state A";
    });
    assert.ok(aAssistant, "summary fixture A assistant message should exist");

    const result = await session.navigateTree(aAssistant.id, { summarize: true });

    assert.equal(result.cancelled, false, "conversation-only tree navigation should allow branch summaries");
    assert.ok(result.summaryEntry, "summary navigation should create a branch summary entry");
    assert.equal(normalizeEol(await readText(filePath)), "B\n", "summary navigation should keep the current workspace");
    assert.deepEqual(ui.selections.map(({ title }) => title), ["Tree navigation"]);

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testCancelledConversationOnlySummaryDoesNotLeakAnchor(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "cancelled-summary.txt");
    const ui = configureTestUI(session, [
      "Conversation only (keep current files)",
      "Conversation and workspace",
    ]);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "cancelled-summary.txt", content: "A\n" })]),
      fauxAssistantMessage("created cancelled summary A"),
      fauxAssistantMessage([fauxToolCall("write", { path: "cancelled-summary.txt", content: "B\n" })]),
      fauxAssistantMessage("created cancelled summary B"),
      fauxAssistantMessage("This summary should be aborted."),
    ]);

    await session.prompt("create cancelled summary A");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "cancelled summary A snapshot was not created");
    await session.prompt("create cancelled summary B");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "cancelled summary B snapshot was not created");

    const aAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created cancelled summary A";
    });
    assert.ok(aAssistant, "cancelled summary fixture A assistant message should exist");
    const leafBeforeSummary = session.sessionManager.getLeafId();

    const summaryNavigation = session.navigateTree(aAssistant.id, { summarize: true });
    session.abortBranchSummary();
    await assert.rejects(
      summaryNavigation,
      /Branch summarization failed: This operation was aborted/,
      "summary navigation should report the abort",
    );

    assert.equal(session.sessionManager.getLeafId(), leafBeforeSummary, "an aborted summary should keep the conversation leaf");
    assert.equal(normalizeEol(await readText(filePath)), "B\n", "an aborted summary should keep the workspace");

    const combinedResult = await session.navigateTree(aAssistant.id, { summarize: false });
    assert.equal(combinedResult.cancelled, false, "navigation after an aborted summary should not reuse its pending anchor");
    assert.equal(normalizeEol(await readText(filePath)), "A\n");
    assert.equal(await countSnapshots(session, ctx.cwd, "manual"), 0, "an aborted summary must not anchor its snapshot later");
    assert.deepEqual(ui.selections.map(({ title }) => title), ["Tree navigation", "Tree navigation"]);

    const aSnapshot = (await readTurnSnapshots(session, ctx.cwd)).turns[0];
    assert.ok(aSnapshot, "cancelled summary A turn snapshot should exist");
    const { stdout: shadowHead } = await execFileAsync(
      "git",
      shadowGitArgs(session, ctx.cwd, "rev-parse", "HEAD"),
      { cwd: ctx.cwd },
    );
    assert.equal(shadowHead.trim(), aSnapshot.afterCommit, "shadow Git should end at the successful combined target");
    assert.equal(await getShadowStatus(session, ctx.cwd), "", "shadow Git should remain clean after the aborted summary retry");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testConversationOnlyNavigationRecoversPendingWorkspaceBeforeAnchoring(): Promise<void> {
  const ctx = await createContext();
  let resumedCtx: TestContext | undefined;
  try {
    let session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "pending-conversation-only.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "pending-conversation-only.txt", content: "A\n" })]),
      fauxAssistantMessage("created pending state A"),
      fauxAssistantMessage([fauxToolCall("write", { path: "pending-conversation-only.txt", content: "B\n" })]),
      fauxAssistantMessage("created pending state B"),
    ]);

    await session.prompt("create pending state A");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "pending A snapshot was not created");
    await session.prompt("create pending state B");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "pending B snapshot was not created");

    const aAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created pending state A";
    });
    const bAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created pending state B";
    });
    const bSnapshot = (await readTurnSnapshots(session, ctx.cwd)).turns[1];
    assert.ok(aAssistant && bAssistant && bSnapshot, "pending recovery fixture should have both branches and snapshots");

    await writeFile(filePath, "partial restore\n", "utf8");
    await refreshShadowIndex(session, ctx.cwd);
    const { stdout: partialTree } = await execFileAsync(
      "git",
      shadowGitArgs(session, ctx.cwd, "write-tree"),
      { cwd: ctx.cwd },
    );
    await execFileAsync(
      "git",
      shadowGitArgs(session, ctx.cwd, "reset", "--mixed", "--no-refresh", bSnapshot.afterCommit),
      { cwd: ctx.cwd },
    );
    await writeFile(getPendingRecoveryFile(session, ctx.cwd), `${JSON.stringify({
      version: 1,
      commit: bSnapshot.afterCommit,
      workspaceTree: partialTree.trim(),
      createdAt: new Date().toISOString(),
    }, null, 2)}\n`, "utf8");

    const sessionManager = session.sessionManager;
    session.dispose();
    resumedCtx = await createContextForWorkspace(ctx.rootDir, ctx.cwd);
    session = await createSession(resumedCtx, sessionManager);
    configureTestUI(session, [
      "Conversation only (keep current files)",
      "Conversation only (keep current files)",
      "Conversation and workspace",
    ]);

    resumedCtx.provider.setResponses([fauxAssistantMessage("Summary generation must not start while recovery is pending.")]);
    const leafBeforeSummary = session.sessionManager.getLeafId();
    const summaryResult = await session.navigateTree(aAssistant.id, { summarize: true });
    assert.equal(summaryResult.cancelled, true, "pending recovery should safely block summary navigation");
    assert.equal(session.sessionManager.getLeafId(), leafBeforeSummary, "blocked summary navigation should keep the conversation leaf");
    assert.equal(normalizeEol(await readText(filePath)), "partial restore\n", "blocked summary navigation should not run recovery early");
    assert.equal(await pathExists(getPendingRecoveryFile(session, ctx.cwd)), true, "blocked summary should leave recovery pending");

    const conversationOnlyResult = await session.navigateTree(aAssistant.id, { summarize: false });
    assert.equal(conversationOnlyResult.cancelled, false);
    assert.equal(
      normalizeEol(await readText(filePath)),
      "B\n",
      "pending rollback should finish before anchoring a conversation-only branch",
    );
    assert.equal(await pathExists(getPendingRecoveryFile(session, ctx.cwd)), false, "successful recovery should clear pending state");

    const combinedResult = await session.navigateTree(bAssistant.id, { summarize: false });
    assert.equal(combinedResult.cancelled, false, "the recovered anchor should remain usable by later combined navigation");
    assert.equal(normalizeEol(await readText(filePath)), "B\n");

    session.dispose();
  } finally {
    resumedCtx?.provider.unregister();
    await disposeContext(ctx);
  }
}

async function testConversationOnlyNavigationPreservesEditsAfterPendingRecovery(): Promise<void> {
  const ctx = await createContext();
  let resumedCtx: TestContext | undefined;
  try {
    let session = await createSession(ctx);
    const trackedPath = path.join(ctx.cwd, "pending-later-edits.txt");
    const manualPath = path.join(ctx.cwd, "later-manual.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "pending-later-edits.txt", content: "tracked\n" })]),
      fauxAssistantMessage("created pending later edits state"),
    ]);
    await session.prompt("create pending later edits state");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "pending later edits snapshot was not created");

    const originalAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" &&
        entry.message.role === "assistant" &&
        getMessageText(entry) === "created pending later edits state";
    });
    const turn = (await readTurnSnapshots(session, ctx.cwd)).turns[0];
    assert.ok(originalAssistant && turn, "pending later edits fixture should have an assistant and snapshot");

    const { stdout: workspaceTree } = await execFileAsync(
      "git",
      shadowGitArgs(session, ctx.cwd, "rev-parse", `${turn.afterCommit}^{tree}`),
      { cwd: ctx.cwd },
    );
    await writeFile(getPendingRecoveryFile(session, ctx.cwd), `${JSON.stringify({
      version: 1,
      commit: turn.afterCommit,
      workspaceTree: workspaceTree.trim(),
      createdAt: new Date().toISOString(),
    }, null, 2)}\n`, "utf8");

    const sessionManager = session.sessionManager;
    session.dispose();
    resumedCtx = await createContextForWorkspace(ctx.rootDir, ctx.cwd);
    session = await createSession(resumedCtx, sessionManager);
    configureTestUI(session, [
      "Conversation only (keep current files)",
      "Conversation and workspace",
    ]);
    await writeFile(manualPath, "keep later manual edit\n", "utf8");

    await session.prompt("/undo");

    assert.equal(normalizeEol(await readText(trackedPath)), "tracked\n");
    assert.equal(normalizeEol(await readText(manualPath)), "keep later manual edit\n");
    assert.equal(await pathExists(getPendingRecoveryFile(session, ctx.cwd)), false, "kept manual edits should supersede pending recovery");

    const combinedResult = await session.navigateTree(originalAssistant.id, { summarize: false });
    assert.equal(combinedResult.cancelled, false, "the preserved workspace anchor should support later combined navigation");
    assert.equal(normalizeEol(await readText(trackedPath)), "tracked\n");
    assert.equal(await pathExists(manualPath), false, "combined navigation should restore the original branch after preserving edits");

    session.dispose();
  } finally {
    resumedCtx?.provider.unregister();
    await disposeContext(ctx);
  }
}

async function testUndoRedo(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "hello.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([
        fauxToolCall("write", {
          path: "hello.txt",
          content: "hello from turn 1\n",
        }),
      ]),
      fauxAssistantMessage("created hello.txt"),
    ]);

    await session.prompt("create hello.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "first-turn after snapshot was not created");
    assert.equal(await exists(filePath), true, "file should exist after the first turn");
    assert.equal(normalizeEol(await readText(filePath)), "hello from turn 1\n");

    await session.prompt("/undo");
    await waitForExists(filePath, false, "file should be removed after /undo");

    await session.prompt("/redo");
    await waitForExists(filePath, true, "file should be restored after /redo");
    await waitForText(filePath, "hello from turn 1\n", "hello.txt should match after /redo");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testSessionStartDoesNotCreateBaselineEagerly(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);

    assert.equal(await countSnapshots(session, ctx.cwd), 0, "session start should not create baseline eagerly");
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal(await countSnapshots(session, ctx.cwd), 0, "idle baseline warmup should not append session entries before the first turn");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "lazy.txt", content: "lazy baseline\n" })]),
      fauxAssistantMessage("created lazy file"),
    ]);

    await session.prompt("create lazy.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot was not created for lazy baseline flow");

    assert.equal(await countSnapshots(session, ctx.cwd, "baseline") >= 1, true, "baseline should be created lazily before the first turn");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testManualChangesProtectedAcrossUndo(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const fileA = path.join(ctx.cwd, "A.txt");
    const fileB = path.join(ctx.cwd, "B.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([
        fauxToolCall("write", {
          path: "A.txt",
          content: "created by turn A\n",
        }),
      ]),
      fauxAssistantMessage("created A"),
      fauxAssistantMessage([
        fauxToolCall("write", {
          path: "B.txt",
          content: "created by turn B\n",
        }),
      ]),
      fauxAssistantMessage("created B"),
    ]);

    await session.prompt("create A.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "A turn after snapshot was not created");
    assert.equal(await exists(fileA), true);

    await rm(fileA, { force: true });
    assert.equal(await exists(fileA), false, "A should not exist after manual deletion");

    await session.prompt("create B.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "B turn after snapshot was not created");
    assert.equal(await exists(fileB), true);

    await session.prompt("/undo");
    await waitForExists(fileB, false, "B should be removed after undoing the second turn");
    await waitForExists(fileA, false, "manually deleted A should not reappear");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testCheckpointAndTreeGuard(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "checkpoint.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([
        fauxToolCall("write", {
          path: "checkpoint.txt",
          content: "base\n",
        }),
      ]),
      fauxAssistantMessage("created checkpoint file"),
    ]);

    await session.prompt("create checkpoint.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "checkpoint test after snapshot was not created");
    assert.equal(normalizeEol(await readText(filePath)), "base\n");

    await writeFile(filePath, "manual edit\n", "utf8");
    const originalLeafId = session.sessionManager.getLeafId();
    assert.ok(originalLeafId, "current leaf should exist");

    const baseline = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "custom" && entry.customType === "workspace-history.snapshot" && (entry as any).data?.kind === "baseline");
    assert.ok(baseline, "baseline snapshot should exist");

    const treeResult = await session.navigateTree(baseline!.id, { summarize: false });
    assert.equal(treeResult.cancelled, true, "manual edits without a checkpoint should block /tree");
    assert.equal(session.sessionManager.getLeafId(), originalLeafId, "leaf should not change after cancelled navigation");
    assert.equal(normalizeEol(await readText(filePath)), "manual edit\n", "manual edits should be preserved after cancelled navigation");

    await session.prompt("/checkpoint saved-manual");
    const checkpointEntries = session.sessionManager
      .getEntries()
      .filter((entry) => entry.type === "custom" && entry.customType === "workspace-history.snapshot" && (entry as any).data?.kind === "manual");
    assert.equal(checkpointEntries.length > 0, true, "manual checkpoint should be created");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testRepeatedUndo(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const fileA = path.join(ctx.cwd, "A.txt");
    const fileB = path.join(ctx.cwd, "B.txt");
    const fileC = path.join(ctx.cwd, "C.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "A.txt", content: "A\n" })]),
      fauxAssistantMessage("created A"),
      fauxAssistantMessage([
        fauxToolCall("write", { path: "B.txt", content: "B\n" }),
        fauxToolCall("write", { path: "C.txt", content: "C\n" }),
      ]),
      fauxAssistantMessage("created B and C"),
    ]);

    await session.prompt("create A.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "A turn after snapshot was not created");
    await session.prompt("create B.txt and C.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "B/C turn after snapshot was not created");

    assert.equal(await exists(fileA), true);
    assert.equal(await exists(fileB), true);
    assert.equal(await exists(fileC), true);

    await session.prompt("/undo");
    await waitForExists(fileA, true, "A should remain after the first undo");
    await waitForExists(fileB, false, "B should be removed after the first undo");
    await waitForExists(fileC, false, "C should be removed after the first undo");

    await session.prompt("/undo");
    await waitForExists(fileA, false, "A should be removed after the second undo");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testTreeBranchSwitching(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const fileA = path.join(ctx.cwd, "branch.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "branch.txt", content: "A\n" })]),
      fauxAssistantMessage("created A branch state"),
      fauxAssistantMessage([fauxToolCall("write", { path: "branch.txt", content: "C\n" })]),
      fauxAssistantMessage("created C branch state"),
      fauxAssistantMessage([fauxToolCall("write", { path: "branch.txt", content: "D\n" })]),
      fauxAssistantMessage("created D branch state"),
    ]);

    await session.prompt("create branch.txt as A");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "A branch after snapshot was not created");
    await session.prompt("change branch.txt to C");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "C branch after snapshot was not created");

    const cAssistant = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created C branch state");
    assert.ok(cAssistant, "C assistant message should exist");

    await session.prompt("/undo");
    await waitForText(fileA, "A\n", "after undoing back before C, the file should be A");

    await session.prompt("change branch.txt to D");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 3, "D branch after snapshot was not created");
    assert.equal(normalizeEol(await readText(fileA)), "D\n", "D branch should write D");

    const dAssistant = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created D branch state");
    assert.ok(dAssistant, "D assistant message should exist");

    const cTreeResult = await session.navigateTree(cAssistant!.id, { summarize: false });
    assert.equal(cTreeResult.cancelled, false, "switching back to C branch should not be cancelled");
    assert.equal(normalizeEol(await readText(fileA)), "C\n", "workspace should restore C when switching back to C branch");

    const dTreeResult = await session.navigateTree(dAssistant!.id, { summarize: false });
    assert.equal(dTreeResult.cancelled, false, "switching back to D branch should not be cancelled");
    assert.equal(normalizeEol(await readText(fileA)), "D\n", "workspace should restore D when switching back to D branch");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testUndoDoesNotLeakAcrossSessions(): Promise<void> {
  const ctx1 = await createContext();
  try {
    const session1 = await createSession(ctx1);
    const fileA = path.join(ctx1.cwd, "session-a.txt");

    ctx1.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "session-a.txt", content: "from session 1\n" })]),
      fauxAssistantMessage("created session 1 file"),
    ]);

    await session1.prompt("create session-a.txt");
    await waitFor(async () => await countSnapshots(session1, ctx1.cwd, "after") >= 1, "session1 after snapshot was not created");
    assert.equal(normalizeEol(await readText(fileA)), "from session 1\n");
    session1.dispose();
    await new Promise((resolve) => setTimeout(resolve, 0));

    const ctx2 = await createContextForWorkspace(ctx1.rootDir, ctx1.cwd);
    const session2 = await createSession(ctx2);
    const beforeLeaf = session2.sessionManager.getLeafId();
    await session2.prompt("/undo");
    const afterLeaf = session2.sessionManager.getLeafId();

    assert.equal(beforeLeaf, afterLeaf, "a new session should not jump into old session history on /undo");
    assert.equal(normalizeEol(await readText(fileA)), "from session 1\n", "a new session /undo should not restore other old session states");

    session2.dispose();
    ctx2.provider.unregister();
  } finally {
    await disposeContext(ctx1);
  }
}

async function testNonProjectWorkspaceDisablesExtension(): Promise<void> {
  const ctx = await createNonProjectContext();
  try {
    const session = await createSession(ctx);

    await waitFor(async () => (await countSnapshots(session, ctx.cwd)) === 0, "non-project workspace should not create snapshots");

    const nav = await session.navigateTree(session.sessionManager.getLeafId() ?? "", { summarize: false }).catch(() => ({ cancelled: false }));
    assert.equal(nav.cancelled, false, "tree navigation should not be cancelled when workspace history is disabled");

    await session.prompt("/undo");
    await new Promise((resolve) => setTimeout(resolve, 100));
    await waitFor(async () => (await countSnapshots(session, ctx.cwd)) === 0, "non-project workspace should remain disabled for commands");

    await writeFile(path.join(ctx.cwd, "package.json"), JSON.stringify({ name: "timemachine-test-workspace" }, null, 2) + "\n", "utf8");
    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "activated.txt", content: "activated\n" })]),
      fauxAssistantMessage("activated"),
    ]);

    await session.prompt("create activated.txt");
    await waitFor(async () => (await countSnapshots(session, ctx.cwd, "after")) >= 1, "workspace history should re-enable after adding a project marker");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testUndoWorksFromTreeSelectedUserNode(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "tree-undo.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "tree-undo.txt", content: "turn one\n" })]),
      fauxAssistantMessage("created tree undo file"),
    ]);

    await session.prompt("create tree-undo.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "tree undo after snapshot was not created");
    assert.equal(normalizeEol(await readText(filePath)), "turn one\n");

    const userEntry = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "message" && entry.message.role === "user" && getMessageText(entry) === "create tree-undo.txt");
    assert.ok(userEntry, "user entry should exist for tree undo test");

    const nav = await session.navigateTree(userEntry!.id, { summarize: false });
    assert.equal(nav.cancelled, false, "navigating to the user node should succeed");

    await session.prompt("/undo");
    await waitForExists(filePath, false, "file should be removed when undo runs from a tree-selected user node");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testLegacySnapshotEntriesRebuildTurnSnapshots(): Promise<void> {
  const ctx1 = await createContext();
  try {
    const session1 = await createSession(ctx1);
    const filePath = path.join(ctx1.cwd, "legacy-fallback.txt");

    ctx1.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "legacy-fallback.txt", content: "legacy\n" })]),
      fauxAssistantMessage("created legacy fallback file"),
    ]);

    await session1.prompt("create legacy-fallback.txt");
    await waitFor(async () => await countSnapshots(session1, ctx1.cwd, "after") >= 1, "legacy fallback after snapshot was not created");
    assert.equal(normalizeEol(await readText(filePath)), "legacy\n");

    const entries = session1.sessionManager.getEntries();
    const baseline = entries.find((entry) => entry.type === "custom" && entry.customType === "workspace-history.snapshot" && (entry as any).data?.kind === "baseline");
    const userEntry = entries.find((entry) => entry.type === "message" && entry.message.role === "user" && getMessageText(entry) === "create legacy-fallback.txt");
    const assistantEntry = entries.find((entry) => entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created legacy fallback file");
    const turnSnapshots = await readTurnSnapshots(session1, ctx1.cwd);
    const latestTurn = turnSnapshots.turns.at(-1);

    assert.ok(baseline, "baseline snapshot should exist for legacy fallback test");
    assert.ok(userEntry, "user entry should exist for legacy fallback test");
    assert.ok(assistantEntry, "assistant entry should exist for legacy fallback test");
    assert.ok(latestTurn, "turn snapshot should exist for legacy fallback test");

    const legacyEntries = entries.filter((entry) => entry !== baseline) as Array<any>;
    legacyEntries.push({
      type: "custom",
      customType: "workspace-history.snapshot",
      id: `legacy-after-${Date.now()}`,
      parentId: assistantEntry!.id,
      timestamp: Date.now(),
      data: {
        v: 1,
        kind: "after",
        commit: latestTurn!.afterCommit,
        turnId: latestTurn!.turnId,
        beforeSnapshotId: baseline!.id,
        userEntryId: userEntry!.id,
        resultLeafId: assistantEntry!.id,
        createdAt: latestTurn!.createdAt,
      },
    } satisfies CustomEntry<any>);

    const rebuilt = rebuildTurnSnapshotsFromLegacyEntries({
      sessionManager: {
        getEntries: () => [baseline!, ...legacyEntries],
        getEntry: (id: string) => [baseline!, ...legacyEntries].find((entry) => entry.id === id),
      },
    } as any);

    assert.equal(rebuilt.turns.length >= 1, true, "legacy snapshot entries should rebuild at least one turn snapshot");
    assert.equal(rebuilt.turns.at(-1)?.userEntryId, userEntry!.id, "rebuilt legacy turn should preserve user entry id");
    assert.equal(rebuilt.turns.at(-1)?.assistantEntryId, assistantEntry!.id, "rebuilt legacy turn should preserve assistant entry id");

    session1.dispose();
  } finally {
    await disposeContext(ctx1);
  }
}

async function testWindowsReservedNamesAreExcludedFromSnapshotPaths(): Promise<void> {
  assert.equal(isWindowsReservedSnapshotPath("nul"), true, "nul should be treated as a reserved Windows device path");
  assert.equal(isWindowsReservedSnapshotPath("NUL.txt"), true, "nul with extension should be treated as reserved");
  assert.equal(isWindowsReservedSnapshotPath("dir/aux"), true, "reserved device names in subdirectories should be excluded");
  assert.equal(isWindowsReservedSnapshotPath("notes/null.txt"), false, "ordinary names should remain snapshot-manageable");
}

async function testBeforeCommitReusesPreviousAfterCommitWhenWorkspaceUnchanged(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "reuse-before.txt", content: "turn one\n" })]),
      fauxAssistantMessage("created reuse-before file"),
      fauxAssistantMessage("no workspace changes"),
    ]);

    await session.prompt("create reuse-before.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "first after snapshot was not created");

    await session.prompt("just reply without editing files");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "second after snapshot was not created");

    const turns = (await readTurnSnapshots(session, ctx.cwd)).turns;
    assert.equal(turns.length >= 2, true, "expected at least two turn snapshots");

    const first = turns.at(-2);
    const second = turns.at(-1);
    assert.ok(first, "first turn snapshot should exist");
    assert.ok(second, "second turn snapshot should exist");
    assert.equal(second!.beforeCommit, first!.afterCommit, "second before commit should reuse first after commit when workspace is unchanged");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testPiFilesAreSnapshotManagedExceptInternalState(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const piFile = path.join(ctx.cwd, ".pi", "notes.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: ".pi/notes.txt", content: "pi managed file\n" })]),
      fauxAssistantMessage("created .pi note"),
    ]);

    await session.prompt("create .pi/notes.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, ".pi file after snapshot was not created");
    assert.equal(normalizeEol(await readText(piFile)), "pi managed file\n");

    await session.prompt("/undo");
    await waitForExists(piFile, false, ".pi regular file should be removed after /undo");

    await session.prompt("/redo");
    await waitForText(piFile, "pi managed file\n", ".pi regular file should be restored after /redo");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testHistoryIsStoredOutsideWorkspace(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "outside.txt", content: "outside\n" })]),
      fauxAssistantMessage("created outside file"),
    ]);

    await session.prompt("create outside.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "outside storage after snapshot was not created");

    const legacyDir = path.join(ctx.cwd, ".pi", "workspace-history");
    assert.equal(await pathExists(legacyDir), false, "legacy workspace history dir should not be created in workspace");

    const externalRoot = getWorkspaceHistoryStateDir(ctx.rootDir);
    const workspacesDir = path.join(externalRoot, "workspaces");
    assert.equal(await pathExists(workspacesDir), true, "external workspace history dir should exist");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testIdleWarmupIsReusedByFirstTurn(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);

    await waitFor(async () => {
      const turns = await readTurnSnapshots(session, ctx.cwd);
      return turns.turns.length === 0 && (await countSnapshots(session, ctx.cwd, "baseline")) === 0;
    }, "idle warmup should not append session entries", 4000);

    await new Promise((resolve) => setTimeout(resolve, 1800));

    ctx.provider.setResponses([
      fauxAssistantMessage("no workspace changes after warmup"),
    ]);

    await session.prompt("reply without editing files");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot was not created after idle warmup");

    const turns = (await readTurnSnapshots(session, ctx.cwd)).turns;
    const first = turns.at(-1);
    assert.ok(first, "first turn snapshot should exist after idle warmup");
    assert.equal(first!.beforeCommit, first!.afterCommit, "warm first turn with no edits should reuse the warmed baseline commit");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testNewSessionReusesWorkspaceShadowRepo(): Promise<void> {
  const ctx1 = await createContext();
  try {
    const session1 = await createSession(ctx1);

    ctx1.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "reuse-shadow.txt", content: "base\n" })]),
      fauxAssistantMessage("created reusable shadow state"),
    ]);

    await session1.prompt("create reusable shadow state");
    await waitFor(async () => await countSnapshots(session1, ctx1.cwd, "after") >= 1, "first session after snapshot was not created");
    const session1GitDir = getShadowGitDir(session1, ctx1.cwd);
    const retainedSessionRefs = await execFileAsync(
      "git",
      ["--git-dir", session1GitDir, "for-each-ref", "--format=%(refname)", "refs/workspace-history"],
      { cwd: ctx1.cwd },
    );
    assert.match(retainedSessionRefs.stdout, /refs\/workspace-history\/snapshots\//, "session repo should retain snapshot refs");
    session1.dispose();

    const ctx2 = await createContextForWorkspace(ctx1.rootDir, ctx1.cwd);
    const session2 = await createSession(ctx2);
    const workspaceHash = createHash("sha256").update(path.normalize(ctx1.cwd)).digest("hex").slice(0, 24);
    const gitDir = path.join(
      getWorkspaceHistoryStateDir(ctx1.rootDir),
      "workspaces",
      workspaceHash,
      "sessions",
      session2.sessionManager.getSessionId(),
      "repo.git",
    );

    await waitFor(async () => await pathExists(path.join(gitDir, "objects")), "second session shadow git repo should exist", 10000);
    const head = await readFile(path.join(gitDir, "HEAD"), "utf8");
    assert.match(head, /refs\/heads|[0-9a-f]{40}/, "second session should have a cloned shadow repo with HEAD");
    const inheritedRetentionRefs = await execFileAsync(
      "git",
      ["--git-dir", gitDir, "for-each-ref", "--format=%(refname)", "refs/workspace-history"],
      { cwd: ctx1.cwd },
    );
    assert.equal(inheritedRetentionRefs.stdout.trim(), "", "new sessions should not inherit another session's retention refs");

    session2.dispose();
    ctx2.provider.unregister();
  } finally {
    await disposeContext(ctx1);
  }
}

async function testInvalidCurrentShadowRepoIsQuarantinedAndRebuilt(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const sessionRoot = getSessionHistoryDir(session, ctx.cwd);
    const gitDir = getShadowGitDir(session, ctx.cwd);
    const markerFile = "recoverable-marker.txt";
    await mkdir(gitDir, { recursive: true });
    await writeFile(path.join(gitDir, markerFile), "keep this data\n", "utf8");
    const notifications = captureNotifications(session);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "recovered-first.txt", content: "first\n" })]),
      fauxAssistantMessage("created first recovery file"),
      fauxAssistantMessage([fauxToolCall("write", { path: "recovered-second.txt", content: "second\n" })]),
      fauxAssistantMessage("created second recovery file"),
    ]);

    await session.prompt("create first recovery file");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "first snapshot was not created after shadow repo recovery");
    await session.prompt("create second recovery file");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "second snapshot was not created after shadow repo recovery");
    assert.equal(normalizeEol(await readFile(path.join(ctx.cwd, "recovered-first.txt"), "utf8")), "first\n");
    assert.equal(normalizeEol(await readFile(path.join(ctx.cwd, "recovered-second.txt"), "utf8")), "second\n");

    const quarantinedDirs = (await readdir(sessionRoot, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() && entry.name.startsWith("repo.git.invalid-"));
    assert.equal(quarantinedDirs.length, 1, "invalid current repo should be quarantined once");
    assert.equal(
      await readFile(path.join(sessionRoot, quarantinedDirs[0]!.name, markerFile), "utf8"),
      "keep this data\n",
      "quarantining an invalid repo should preserve its data",
    );

    await assertValidShadowRepo(gitDir, ctx.cwd, "rebuilt shadow repo");
    assert.equal(
      notifications.filter((message) => /invalid.*rebuilt/i.test(message)).length,
      1,
      "automatic recovery should notify the user once",
    );

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testRepeatedUndoAcrossManualInterTurnChanges(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "manual-between-undos.txt");
    const notifications = captureNotifications(session);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "manual-between-undos.txt", content: "first agent state\n" })]),
      fauxAssistantMessage("created first agent state"),
      fauxAssistantMessage([fauxToolCall("write", { path: "manual-between-undos.txt", content: "second agent state\n" })]),
      fauxAssistantMessage("created second agent state"),
    ]);

    await session.prompt("create first agent state");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "first inter-turn snapshot was not created");
    await writeFile(filePath, "manual state between turns\n", "utf8");
    await session.prompt("create second agent state");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, "second inter-turn snapshot was not created");
    const firstAssistant = session.sessionManager.getEntries().find((entry) => {
      return entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created first agent state";
    });
    const turns = (await readTurnSnapshots(session, ctx.cwd)).turns;
    assert.ok(firstAssistant && turns[0] && turns[1], "inter-turn fixture should have both assistant entries and snapshots");
    assert.notEqual(
      turns[1].beforeCommit,
      turns[0].afterCommit,
      "the manual inter-turn state should differ from the first turn's after snapshot",
    );

    await session.prompt("/undo");
    assert.equal(
      normalizeEol(await readText(filePath)),
      "manual state between turns\n",
      "first undo should restore the manual state captured before the second turn",
    );
    const firstUndoLeafId = session.sessionManager.getLeafId();
    const firstUndoAnchor = firstUndoLeafId ? session.sessionManager.getEntry(firstUndoLeafId) : undefined;
    assert.equal(firstUndoAnchor?.type, "custom", "first undo should anchor the divergent restored workspace");
    assert.equal(firstUndoAnchor?.parentId, firstAssistant.id, "the restored workspace anchor should follow the first assistant");
    if (firstUndoAnchor?.type === "custom") {
      const anchorData = firstUndoAnchor.data as Partial<{ commit: string; kind: string; label: string }> | undefined;
      assert.equal(firstUndoAnchor.customType, "workspace-history.snapshot");
      assert.equal(anchorData?.kind, "manual");
      assert.equal(anchorData?.commit, turns[1].beforeCommit);
      assert.equal(anchorData?.label, "restored workspace navigation");
    }
    const { stdout: shadowHead } = await execFileAsync(
      "git",
      shadowGitArgs(session, ctx.cwd, "rev-parse", "HEAD"),
      { cwd: ctx.cwd },
    );
    assert.equal(shadowHead.trim(), turns[1].beforeCommit, "first undo should align shadow HEAD with the restored manual state");
    assert.equal(await getShadowStatus(session, ctx.cwd), "", "first undo should leave the shadow index clean");

    notifications.length = 0;
    await session.prompt("/undo");

    assert.equal(
      notifications.some((message) => message.includes("unsnapshotted changes")),
      false,
      "the restored inter-turn state should not be mistaken for unsnapshotted changes",
    );
    assert.equal(await pathExists(filePath), false, "second undo should restore the workspace from before the first turn");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testInvalidReusableShadowRepoIsQuarantinedAndRebuilt(): Promise<void> {
  const ctx1 = await createContext();
  try {
    const session1 = await createSession(ctx1);
    ctx1.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "reusable-source.txt", content: "source\n" })]),
      fauxAssistantMessage("created reusable source"),
    ]);
    await session1.prompt("create reusable source");
    await waitFor(async () => await countSnapshots(session1, ctx1.cwd, "after") >= 1, "source session snapshot was not created");

    const sourceGitDir = getShadowGitDir(session1, ctx1.cwd);
    const reusableGitDir = getReusableGitDir(session1, ctx1.cwd);
    await waitFor(async () => {
      try {
        const result = await execFileAsync("git", ["--git-dir", reusableGitDir, "rev-parse", "--verify", "HEAD^{commit}"], { cwd: ctx1.cwd });
        return /^[0-9a-f]{40,64}$/i.test(result.stdout.trim());
      } catch {
        return false;
      }
    }, "workspace reusable repo was not created", 10000);
    session1.dispose();

    const markerFile = "reusable-marker.txt";
    await rm(reusableGitDir, { recursive: true, force: true });
    await mkdir(reusableGitDir, { recursive: true });
    await writeFile(path.join(reusableGitDir, markerFile), "preserve reusable data\n", "utf8");

    const ctx2 = await createContextForWorkspace(ctx1.rootDir, ctx1.cwd);
    try {
      const session2 = await createSession(ctx2);
      ctx2.provider.setResponses([
        fauxAssistantMessage([fauxToolCall("write", { path: "reusable-recovered.txt", content: "recovered\n" })]),
        fauxAssistantMessage("created after reusable recovery"),
      ]);
      await session2.prompt("create after reusable recovery");
      await waitFor(async () => await countSnapshots(session2, ctx2.cwd, "after") >= 1, "snapshot was not created after reusable repo recovery");
      assert.equal(normalizeEol(await readFile(path.join(ctx2.cwd, "reusable-recovered.txt"), "utf8")), "recovered\n");

      const reusableRoot = path.dirname(reusableGitDir);
      const quarantinedDirs = (await readdir(reusableRoot, { withFileTypes: true }))
        .filter((entry) => entry.isDirectory() && entry.name.startsWith("repo.git.invalid-"));
      assert.equal(quarantinedDirs.length, 1, "invalid reusable repo should be quarantined once");
      assert.equal(
        await readFile(path.join(reusableRoot, quarantinedDirs[0]!.name, markerFile), "utf8"),
        "preserve reusable data\n",
        "quarantining the reusable repo should preserve its data",
      );
      assert.equal(await pathExists(path.join(sourceGitDir, "HEAD")), true, "recovery should not modify the source session repo");

      const session2GitDir = getShadowGitDir(session2, ctx2.cwd);
      await assertValidShadowRepo(session2GitDir, ctx2.cwd, "new session shadow repo");

      session2.dispose();
    } finally {
      ctx2.provider.unregister();
    }
  } finally {
    await disposeContext(ctx1);
  }
}

async function testFailedShadowRepoRebuildDoesNotLeaveCanonicalRepo(): Promise<void> {
  const ctx1 = await createContext();
  try {
    const session1 = await createSession(ctx1);
    ctx1.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "rebuild-source.txt", content: "source\n" })]),
      fauxAssistantMessage("created rebuild source"),
    ]);
    await session1.prompt("create rebuild source");
    await waitFor(async () => await countSnapshots(session1, ctx1.cwd, "after") >= 1, "rebuild source snapshot was not created");

    const sourceGitDir = getShadowGitDir(session1, ctx1.cwd);
    const reusableGitDir = getReusableGitDir(session1, ctx1.cwd);
    await waitFor(async () => await pathExists(path.join(reusableGitDir, "HEAD")), "reusable repo was not created", 10000);
    session1.dispose();

    const headResult = await execFileAsync("git", ["--git-dir", sourceGitDir, "rev-parse", "--verify", "HEAD^{commit}"], { cwd: ctx1.cwd });
    const headCommit = headResult.stdout.trim();
    const sourceCommitObject = path.join(sourceGitDir, "objects", headCommit.slice(0, 2), headCommit.slice(2));
    assert.equal(await pathExists(sourceCommitObject), true, "source commit should be loose for the corrupt repo fixture");

    await rm(reusableGitDir, { recursive: true, force: true });
    await execFileAsync("git", ["init", "--bare", reusableGitDir], { cwd: ctx1.cwd });
    await execFileAsync("git", ["--git-dir", reusableGitDir, "symbolic-ref", "HEAD", "refs/heads/main"], { cwd: ctx1.cwd });
    await mkdir(path.join(reusableGitDir, "refs", "heads"), { recursive: true });
    await writeFile(path.join(reusableGitDir, "refs", "heads", "main"), `${headCommit}\n`, "utf8");
    const corruptCommitObject = path.join(reusableGitDir, "objects", headCommit.slice(0, 2), headCommit.slice(2));
    await mkdir(path.dirname(corruptCommitObject), { recursive: true });
    await copyFile(sourceCommitObject, corruptCommitObject);

    const reusableHead = await execFileAsync("git", ["--git-dir", reusableGitDir, "rev-parse", "--verify", "HEAD^{commit}"], { cwd: ctx1.cwd });
    assert.equal(reusableHead.stdout.trim(), headCommit, "corrupt fixture should pass lightweight reusable HEAD validation");

    const ctx2 = await createContextForWorkspace(ctx1.rootDir, ctx1.cwd);
    try {
      const session2 = await createSession(ctx2);
      const sessionRoot = getSessionHistoryDir(session2, ctx2.cwd);
      const session2GitDir = getShadowGitDir(session2, ctx2.cwd);
      await mkdir(session2GitDir, { recursive: true });
      await writeFile(path.join(session2GitDir, "failure-marker.txt"), "preserve failed recovery data\n", "utf8");
      const notifications = captureNotifications(session2);
      ctx2.provider.setResponses([
        fauxAssistantMessage([fauxToolCall("write", { path: "rebuild-retried.txt", content: "retried\n" })]),
        fauxAssistantMessage("created after rebuild retry"),
      ]);

      await assert.rejects(
        session2.prompt("create after rebuild retry"),
        /Unable to rebuild shadow repository[\s\S]*(?:tree|object)/i,
        "a corrupt reusable source should surface the Git rebuild failure",
      );
      assert.equal(await pathExists(session2GitDir), false, "failed rebuild should not leave a repo at the canonical path");
      const failedBuildDirs = (await readdir(sessionRoot, { withFileTypes: true }))
        .filter((entry) => entry.isDirectory() && entry.name.startsWith(".wh-"));
      assert.equal(failedBuildDirs.length, 1, "failed rebuild should preserve its partial repo separately");
      assert.equal(notifications.some((message) => /invalid.*rebuilt/i.test(message)), false, "failed rebuild should not claim recovery succeeded");

      await rm(reusableGitDir, { recursive: true, force: true });
      await session2.prompt("create after rebuild retry");
      await waitFor(async () => await countSnapshots(session2, ctx2.cwd, "after") >= 1, "snapshot was not created after retrying shadow repo recovery");
      assert.equal(normalizeEol(await readFile(path.join(ctx2.cwd, "rebuild-retried.txt"), "utf8")), "retried\n");
      await assertValidShadowRepo(session2GitDir, ctx2.cwd, "retried shadow repo");
      assert.equal(
        notifications.filter((message) => /invalid.*rebuilt/i.test(message)).length,
        1,
        "successful retry should report the earlier invalid repo recovery once",
      );

      session2.dispose();
    } finally {
      ctx2.provider.unregister();
    }
  } finally {
    await disposeContext(ctx1);
  }
}

async function testStaleShadowRepoLockIsRecovered(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "lock-recovery.txt", content: "recovered\n" })]),
      fauxAssistantMessage("created lock recovery file"),
    ]);

    const workspaceHash = createHash("sha256").update(path.normalize(ctx.cwd)).digest("hex").slice(0, 24);
    const sessionRoot = path.join(
      getWorkspaceHistoryStateDir(ctx.rootDir),
      "workspaces",
      workspaceHash,
      "sessions",
      session.sessionManager.getSessionId(),
    );
    const gitDir = path.join(sessionRoot, "repo.git");
    await rm(gitDir, { recursive: true, force: true });
    await mkdir(sessionRoot, { recursive: true });
    await execFileAsync("git", ["init", "--bare", gitDir], { cwd: ctx.cwd });
    const lockPath = path.join(gitDir, "index.lock");
    await writeFile(lockPath, "stale lock\n", "utf8");
    const staleTime = new Date(Date.now() - 60_000);
    await utimes(lockPath, staleTime, staleTime);

    await session.prompt("create lock recovery file");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "after snapshot should be created after stale lock recovery");
    assert.equal(await exists(lockPath), false, "stale index.lock should be removed automatically");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testUnicodePathsSurviveUndoRedo(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const relativePath = "src/后台/views/系统设置/工程设置/微信设置/油品设置/utils.ts";
    const filePath = path.join(ctx.cwd, ...relativePath.split("/"));

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: relativePath, content: "export const value = 1;\n" })]),
      fauxAssistantMessage("created unicode path file"),
    ]);

    await session.prompt("create unicode path file");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "unicode path after snapshot was not created");
    await waitForText(filePath, "export const value = 1;\n", "unicode path file should be created");

    await session.prompt("/undo");
    await waitForExists(filePath, false, "unicode path file should be removed after /undo");

    await session.prompt("/redo");
    await waitForText(filePath, "export const value = 1;\n", "unicode path file should be restored after /redo");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testUndoAndRedoBlockOnUnsnapshottedManualChanges(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "guard.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "guard.txt", content: "turn one\n" })]),
      fauxAssistantMessage("created guard file"),
    ]);

    await session.prompt("create guard.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "guard after snapshot was not created");

    await writeFile(filePath, "manual edit\n", "utf8");
    const undoLeafBefore = session.sessionManager.getLeafId();
    await session.prompt("/undo");
    const undoLeafAfter = session.sessionManager.getLeafId();

    assert.equal(undoLeafAfter, undoLeafBefore, "/undo should be blocked by unsnapshotted manual edits");
    assert.equal(normalizeEol(await readText(filePath)), "manual edit\n", "manual edits should remain after blocked /undo");

    await session.prompt("/checkpoint guard-manual");
    await session.prompt("/undo");
    await waitForExists(filePath, false, "file should be removed after undo once manual edits are checkpointed");

    await session.prompt("/redo");
    await waitForText(filePath, "manual edit\n", "redo should restore the last successfully undone location");

    await writeFile(filePath, "manual redo edit\n", "utf8");
    const redoLeafBefore = session.sessionManager.getLeafId();
    await session.prompt("/undo");
    await session.prompt("/redo");
    const redoLeafAfter = session.sessionManager.getLeafId();

    assert.equal(redoLeafAfter, redoLeafBefore, "/redo should be blocked by unsnapshotted manual edits");
    assert.equal(normalizeEol(await readText(filePath)), "manual redo edit\n", "manual edits should remain after blocked /redo");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testGitignoreStopsManagingIgnoredPaths(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const ignoredFilePath = path.join(ctx.cwd, "generated.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "generated.txt", content: "turn one\n" })]),
      fauxAssistantMessage("created generated file"),
      fauxAssistantMessage([fauxToolCall("write", { path: ".gitignore", content: "generated.txt\n" })]),
      fauxAssistantMessage("ignored generated file"),
    ]);

    await session.prompt("create generated.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "generated file after snapshot was not created");

    await session.prompt("ignore generated.txt in .gitignore");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 2, ".gitignore update after snapshot was not created");

    await writeFile(ignoredFilePath, "manual ignored edit\n", "utf8");
    await session.prompt("/undo");

    await waitForText(ignoredFilePath, "manual ignored edit\n", "ignored file should no longer be managed after .gitignore excludes it");
    await waitForExists(path.join(ctx.cwd, ".gitignore"), false, ".gitignore should be removed when undoing to the earlier snapshot");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testRestoreFailureDoesNotDeleteCurrentWorkspace(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "safe.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "safe.txt", content: "keep me\n" })]),
      fauxAssistantMessage("created safe file"),
    ]);

    await session.prompt("create safe.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "safe file after snapshot was not created");

    const sessionId = session.sessionManager.getSessionId();
    const workspaceHash = createHash("sha256").update(ctx.cwd).digest("hex").slice(0, 24);
    const workspaceRoot = path.join(getWorkspaceHistoryStateDir(ctx.rootDir), "workspaces", workspaceHash);
    const gitDir = path.join(workspaceRoot, "sessions", sessionId, "repo.git");
    await rm(gitDir, { recursive: true, force: true });
    await rm(path.join(workspaceRoot, "repo.git"), { recursive: true, force: true });

    const baseline = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "custom" && entry.customType === "workspace-history.snapshot" && (entry as any).data?.kind === "baseline");
    assert.ok(baseline, "baseline snapshot should exist for restore failure test");

    const originalLeaf = session.sessionManager.getLeafId();
    const nav = await session.navigateTree(baseline!.id, { summarize: false });

    assert.equal(nav.cancelled, true, "tree navigation should be cancelled when restore fails");
    assert.equal(session.sessionManager.getLeafId(), originalLeaf, "leaf should remain unchanged after restore failure");
    assert.equal(normalizeEol(await readText(filePath)), "keep me\n", "current workspace should remain intact after restore failure");

    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testTreeRestoreWaitsForTransientWindowsFileLock(): Promise<void> {
  if (process.platform !== "win32") {
    return;
  }

  const ctx = await createContext();
  let releaseLock: (() => Promise<void>) | undefined;
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "transient-lock.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "transient-lock.txt", content: "locked briefly\n" })]),
      fauxAssistantMessage("created transient lock file"),
    ]);
    await session.prompt("create transient-lock.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "transient lock after snapshot was not created");

    const baseline = findBaselineSnapshot(session);
    assert.ok(baseline, "baseline snapshot should exist for transient lock test");
    const notifications = captureNotifications(session);

    releaseLock = await holdWindowsFileWithoutDeleteSharing(filePath, ctx.rootDir, { releaseAfterMs: 500 });
    await refreshShadowIndex(session, ctx.cwd);
    assert.equal(await getShadowStatus(session, ctx.cwd), "", "file lock test must start from a clean workspace");
    const nav = await session.navigateTree(baseline!.id, { summarize: false });

    assert.equal(nav.cancelled, false, `tree navigation should wait for a transient Windows file lock: ${notifications.join(" | ")}`);
    assert.equal(session.sessionManager.getLeafId(), baseline!.id, "tree navigation should reach the selected history node");
    await waitForExists(filePath, false, "target snapshot should be restored after the transient lock is released");
    session.dispose();
  } finally {
    await releaseLock?.();
    await disposeContext(ctx);
  }
}

async function testTreeRestoreRetriesLockedWindowsFileReplacement(): Promise<void> {
  if (process.platform !== "win32") {
    return;
  }

  const ctx = await createContext();
  let releaseLock: (() => Promise<void>) | undefined;
  try {
    const filePath = path.join(ctx.cwd, "replace-lock.txt");
    await writeFile(filePath, "before\n", "utf8");
    const session = await createSession(ctx);

    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "replace-lock.txt", content: "after\n" })]),
      fauxAssistantMessage("updated replace lock file"),
    ]);
    await session.prompt("update replace-lock.txt");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "replacement lock after snapshot was not created");

    const baseline = findBaselineSnapshot(session);
    assert.ok(baseline, "baseline snapshot should exist for replacement lock test");

    releaseLock = await holdWindowsFileWithoutDeleteSharing(filePath, ctx.rootDir, { releaseAfterMs: 500 });
    await refreshShadowIndex(session, ctx.cwd);
    const nav = await session.navigateTree(baseline!.id, { summarize: false });

    assert.equal(nav.cancelled, false, "tree navigation should retry replacing a briefly locked Windows file");
    assert.equal(session.sessionManager.getLeafId(), baseline!.id, "replacement retry should reach the selected history node");
    await waitForText(filePath, "before\n", "replacement retry should restore the target file contents");
    session.dispose();
  } finally {
    await releaseLock?.();
    await disposeContext(ctx);
  }
}

async function testPersistentWindowsFileLockReportsCauseAndCanBeRetried(): Promise<void> {
  if (process.platform !== "win32") {
    return;
  }

  const ctx = await createContext();
  let resumedCtx: TestContext | undefined;
  let releaseLock: (() => Promise<void>) | undefined;
  try {
    const companionPath = path.join(ctx.cwd, "z-companion.txt");
    await writeFile(companionPath, "before\n", "utf8");
    let session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "persistent-lock.txt");

    ctx.provider.setResponses([
      fauxAssistantMessage([
        fauxToolCall("write", { path: "persistent-lock.txt", content: "still locked\n" }),
        fauxToolCall("write", { path: "z-companion.txt", content: "after\n" }),
      ]),
      fauxAssistantMessage("updated files for persistent lock test"),
    ]);
    await session.prompt("update files for persistent lock test");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "persistent lock after snapshot was not created");

    const baseline = findBaselineSnapshot(session);
    assert.ok(baseline, "baseline snapshot should exist for persistent lock test");
    const turnSnapshot = (await readTurnSnapshots(session, ctx.cwd)).turns[0];
    assert.ok(turnSnapshot, "turn snapshot should exist for persistent lock test");

    const notifications = captureNotifications(session);

    releaseLock = await holdWindowsFileWithoutDeleteSharing(filePath, ctx.rootDir, {
      mutateAfterMs: 700,
      mutateContent: "changed while locked\n",
    });
    await refreshShadowIndex(session, ctx.cwd);
    const originalLeaf = session.sessionManager.getLeafId();
    await session.prompt("/undo");

    assert.equal(session.sessionManager.getLeafId(), originalLeaf, "cancelled undo should keep the current history node");
    assert.equal(normalizeEol(await readText(filePath)), "changed while locked\n", "persistent lock should not be skipped during restore");
    assert.equal(normalizeEol(await readText(companionPath)), "after\n", "failed rollback should realign the companion file to the current snapshot");
    const failureNotification = notifications.find((message) => /unable to unlink|failed to remove/i.test(message));
    assert.ok(failureNotification?.includes("persistent-lock.txt"), "restore failure should identify the locked file and preserve the real Git error");
    assert.doesNotMatch(failureNotification, /--git-dir|repo\.git/i, "restore failure should not expose internal shadow repository paths");
    assert.equal(notifications.includes("Undo cancelled."), false, "undo should not replace the actionable restore error with a generic notification");

    const { stdout: shadowHead } = await execFileAsync("git", shadowGitArgs(session, ctx.cwd, "rev-parse", "HEAD"), { cwd: ctx.cwd });
    assert.equal(shadowHead.trim(), turnSnapshot.afterCommit, "failed rollback should realign the shadow HEAD to the current snapshot");
    await execFileAsync("git", shadowGitArgs(
      session,
      ctx.cwd,
      "diff",
      "--cached",
      "--quiet",
      turnSnapshot.afterCommit,
      "--",
      ".",
    ), { cwd: ctx.cwd });
    assert.equal(await pathExists(getPendingRecoveryFile(session, ctx.cwd)), true, "failed rollback should persist its recovery state");

    const sessionManager = session.sessionManager;
    session.dispose();
    resumedCtx = await createContextForWorkspace(ctx.rootDir, ctx.cwd);
    session = await createSession(resumedCtx, sessionManager);

    await releaseLock();
    releaseLock = undefined;
    await session.prompt("/undo");

    assert.equal(session.sessionManager.getLeafId(), baseline!.id, "undo should be retryable after the persistent lock is released");
    await waitForExists(filePath, false, "retry should restore the target snapshot without skipping the formerly locked file");
    await waitForText(companionPath, "before\n", "retry should restore every file in the target snapshot");
    assert.equal(await pathExists(getPendingRecoveryFile(session, ctx.cwd)), false, "successful recovery should clear the persisted recovery state");
    session.dispose();
  } finally {
    resumedCtx?.provider.unregister();
    await releaseLock?.();
    await disposeContext(ctx);
  }
}

async function testPendingRecoveryPreservesLaterManualEdits(): Promise<void> {
  if (process.platform !== "win32") {
    return;
  }

  const ctx = await createContext();
  let releaseLock: (() => Promise<void>) | undefined;
  try {
    const companionPath = path.join(ctx.cwd, "z-recovery-companion.txt");
    const lockedPath = path.join(ctx.cwd, "recovery-guard-lock.txt");
    const manualPath = path.join(ctx.cwd, "manual-after-failure.txt");
    await writeFile(companionPath, "before\n", "utf8");
    const session = await createSession(ctx);

    ctx.provider.setResponses([
      fauxAssistantMessage([
        fauxToolCall("write", { path: "recovery-guard-lock.txt", content: "current\n" }),
        fauxToolCall("write", { path: "z-recovery-companion.txt", content: "after\n" }),
      ]),
      fauxAssistantMessage("updated files for recovery guard test"),
    ]);
    await session.prompt("update files for recovery guard test");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") >= 1, "recovery guard after snapshot was not created");

    const baseline = findBaselineSnapshot(session);
    assert.ok(baseline, "baseline snapshot should exist for recovery guard test");
    const notifications = captureNotifications(session);
    releaseLock = await holdWindowsFileWithoutDeleteSharing(lockedPath, ctx.rootDir, {
      mutateAfterMs: 700,
      mutateContent: "changed while locked\n",
    });
    await refreshShadowIndex(session, ctx.cwd);
    const originalLeaf = session.sessionManager.getLeafId();
    await session.prompt("/undo");
    assert.equal(session.sessionManager.getLeafId(), originalLeaf, "persistent lock should cancel the first undo");

    await writeFile(manualPath, "preserve me\n", "utf8");
    await releaseLock();
    releaseLock = undefined;
    notifications.length = 0;
    await session.prompt("/undo");

    assert.equal(session.sessionManager.getLeafId(), originalLeaf, "later manual edits should block automatic pending recovery");
    await waitForText(manualPath, "preserve me\n", "pending recovery must not overwrite later manual edits");
    assert.ok(
      notifications.some((message) => message.includes("Run /checkpoint")),
      "blocked recovery should direct the user to checkpoint later edits",
    );

    await session.prompt("/checkpoint keep recovery edits");
    await waitForText(manualPath, "preserve me\n", "checkpoint should preserve edits made after the failed restore");
    assert.equal(await pathExists(getPendingRecoveryFile(session, ctx.cwd)), false, "checkpoint should clear obsolete recovery state");

    await session.prompt("/undo");
    assert.equal(session.sessionManager.getLeafId(), baseline!.id, "undo should work after preserving later edits with a checkpoint");
    await waitForExists(lockedPath, false, "undo should restore the target snapshot after checkpointing later edits");
    await waitForText(companionPath, "before\n", "undo should restore the companion file after checkpointing later edits");
    session.dispose();
  } finally {
    await releaseLock?.();
    await disposeContext(ctx);
  }
}

async function testBranchSnapshotsSurviveGitPrune(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    const filePath = path.join(ctx.cwd, "branch-prune.txt");
    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "branch-prune.txt", content: "A\n" })]),
      fauxAssistantMessage("created A branch"),
      fauxAssistantMessage([fauxToolCall("write", { path: "branch-prune.txt", content: "C\n" })]),
      fauxAssistantMessage("created C branch"),
      fauxAssistantMessage([fauxToolCall("write", { path: "branch-prune.txt", content: "D\n" })]),
      fauxAssistantMessage("created D branch"),
    ]);

    await session.prompt("create A branch");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") === 1, "A snapshot missing");
    await session.prompt("create C branch");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") === 2, "C snapshot missing");

    const cAssistant = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created C branch");
    assert.ok(cAssistant, "C assistant message should exist");
    const cCommit = (await readTurnSnapshots(session, ctx.cwd)).turns[1]?.afterCommit;
    assert.ok(cCommit, "C snapshot commit should exist");

    await session.prompt("/undo");
    await waitForText(filePath, "A\n", "undo should restore A before branching");
    await session.prompt("create D branch");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") === 3, "D snapshot missing");
    const dAssistant = session.sessionManager
      .getEntries()
      .find((entry) => entry.type === "message" && entry.message.role === "assistant" && getMessageText(entry) === "created D branch");
    assert.ok(dAssistant, "D assistant message should exist");

    const gitDir = getShadowGitDir(session, ctx.cwd);
    const reusableGitDir = getReusableGitDir(session, ctx.cwd);
    await rm(reusableGitDir, { recursive: true, force: true });
    await writeFile(path.join(ctx.cwd, "reusable-marker.txt"), "current branch only\n", "utf8");
    await session.prompt("/checkpoint rebuild-reusable");
    await waitFor(async () => {
      try {
        await execFileAsync("git", ["--git-dir", reusableGitDir, "rev-parse", "--verify", "HEAD"], { cwd: ctx.cwd });
        return true;
      } catch {
        return false;
      }
    }, "reusable repo should be rebuilt");
    await assert.rejects(
      execFileAsync("git", ["--git-dir", reusableGitDir, "cat-file", "-e", `${cCommit}^{commit}`], { cwd: ctx.cwd }),
      "reusable repo should not copy objects reachable only from retention refs",
    );

    await pruneUnreachableGitObjects(gitDir, ctx.cwd);
    await execFileAsync("git", ["--git-dir", gitDir, "cat-file", "-e", `${cCommit}^{commit}`], { cwd: ctx.cwd });

    const cNavigation = await session.navigateTree(cAssistant!.id, { summarize: false });
    assert.equal(cNavigation.cancelled, false, "C branch should remain restorable after Git prune");
    assert.equal(normalizeEol(await readText(filePath)), "C\n");

    const dNavigation = await session.navigateTree(dAssistant!.id, { summarize: false });
    assert.equal(dNavigation.cancelled, false, "D branch should remain restorable after Git prune");
    assert.equal(normalizeEol(await readText(filePath)), "D\n");

    await execFileAsync(
      "git",
      ["--git-dir", gitDir, "update-ref", "-d", `refs/workspace-history/snapshots/${cCommit}`],
      { cwd: ctx.cwd },
    );
    await pruneUnreachableGitObjects(gitDir, ctx.cwd);
    await assert.rejects(execFileAsync("git", ["--git-dir", gitDir, "cat-file", "-e", `${cCommit}^{commit}`], { cwd: ctx.cwd }));

    const leafBeforeMissingTarget = session.sessionManager.getLeafId();
    const missingNavigation = await session.navigateTree(cAssistant!.id, { summarize: false });
    assert.equal(missingNavigation.cancelled, true, "navigation should cancel when only the target snapshot is missing");
    assert.equal(session.sessionManager.getLeafId(), leafBeforeMissingTarget, "cancelled navigation should preserve the current history node");
    assert.equal(normalizeEol(await readText(filePath)), "D\n", "cancelled navigation should preserve the current workspace");
    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testMissingPreviousSnapshotFallsBackToFreshBefore(): Promise<void> {
  const ctx = await createContext();
  try {
    const session = await createSession(ctx);
    ctx.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "missing-snapshot.txt", content: "first\n" })]),
      fauxAssistantMessage("created first state"),
      fauxAssistantMessage("completed recovery turn"),
    ]);

    await session.prompt("create first state");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") === 1, "first snapshot missing");
    const first = (await readTurnSnapshots(session, ctx.cwd)).turns[0];
    assert.ok(first, "first turn snapshot should exist");
    assert.notEqual(first.beforeCommit, first.afterCommit, "fixture requires a distinct after commit");

    const gitDir = getShadowGitDir(session, ctx.cwd);
    const headRef = (await execFileAsync("git", ["--git-dir", gitDir, "symbolic-ref", "HEAD"], { cwd: ctx.cwd })).stdout.trim();
    await execFileAsync("git", ["--git-dir", gitDir, "update-ref", headRef, first.beforeCommit], { cwd: ctx.cwd });
    await execFileAsync(
      "git",
      ["--git-dir", gitDir, "update-ref", "-d", `refs/workspace-history/snapshots/${first.afterCommit}`],
      { cwd: ctx.cwd },
    );
    await pruneUnreachableGitObjects(gitDir, ctx.cwd);
    await assert.rejects(execFileAsync("git", ["--git-dir", gitDir, "cat-file", "-e", `${first.afterCommit}^{commit}`], { cwd: ctx.cwd }));

    await session.prompt("/checkpoint recovery-head");
    await session.prompt("continue after stale snapshot");
    await waitFor(async () => await countSnapshots(session, ctx.cwd, "after") === 2, "recovery turn snapshot missing");

    const recovered = (await readTurnSnapshots(session, ctx.cwd)).turns[1];
    assert.ok(recovered?.beforeCommit, "recovery turn should receive a fresh before commit");
    await execFileAsync("git", ["--git-dir", gitDir, "cat-file", "-e", `${recovered.beforeCommit}^{commit}`], { cwd: ctx.cwd });
    session.dispose();
  } finally {
    await disposeContext(ctx);
  }
}

async function testSessionStartRebuildsSnapshotRetentionRefs(): Promise<void> {
  const ctx1 = await createContext();
  let ctx2: TestContext | undefined;
  try {
    const session1 = await createSession(ctx1);
    ctx1.provider.setResponses([
      fauxAssistantMessage([fauxToolCall("write", { path: "migration.txt", content: "migrated\n" })]),
      fauxAssistantMessage("created migration state"),
    ]);

    await session1.prompt("create migration state");
    await waitFor(async () => await countSnapshots(session1, ctx1.cwd, "after") === 1, "migration snapshot missing");
    const turn = (await readTurnSnapshots(session1, ctx1.cwd)).turns[0];
    assert.ok(turn, "migration turn should exist");
    const gitDir = getShadowGitDir(session1, ctx1.cwd);
    const sessionManager = session1.sessionManager;
    session1.dispose();

    const refs = await execFileAsync(
      "git",
      ["--git-dir", gitDir, "for-each-ref", "--format=%(refname)", "refs/workspace-history"],
      { cwd: ctx1.cwd },
    );
    for (const ref of refs.stdout.split(/\r?\n/).filter((value) => value.length > 0)) {
      await execFileAsync("git", ["--git-dir", gitDir, "update-ref", "-d", ref], { cwd: ctx1.cwd });
    }

    ctx2 = await createContextForWorkspace(ctx1.rootDir, ctx1.cwd);
    const session2 = await createSession(ctx2, sessionManager);
    await waitFor(async () => {
      const rebuiltRefs = await execFileAsync(
        "git",
        ["--git-dir", gitDir, "for-each-ref", "--format=%(refname)", "refs/workspace-history"],
        { cwd: ctx1.cwd },
      );
      return rebuiltRefs.stdout.includes(turn.beforeCommit) && rebuiltRefs.stdout.includes(turn.afterCommit);
    }, "session start should rebuild retention refs for existing snapshots");

    session2.dispose();
    ctx2.provider.unregister();
  } finally {
    await disposeContext(ctx1);
  }
}

async function main(): Promise<void> {
  const tests: Array<{ name: string; run: () => Promise<void> }> = [
    { name: "missing previous snapshot falls back to a fresh before snapshot", run: testMissingPreviousSnapshotFallsBackToFreshBefore },
    { name: "branch snapshots survive Git prune", run: testBranchSnapshotsSurviveGitPrune },
    { name: "session start rebuilds snapshot retention refs", run: testSessionStartRebuildsSnapshotRetentionRefs },
    { name: "session start does not create baseline eagerly", run: testSessionStartDoesNotCreateBaselineEagerly },
    { name: "idle warmup is reused by first turn", run: testIdleWarmupIsReusedByFirstTurn },
    { name: "non-project workspace disables extension", run: testNonProjectWorkspaceDisablesExtension },
    { name: "conversation-only undo keeps workspace", run: testUndoConversationOnlyKeepsWorkspace },
    { name: "redo reuses conversation-only undo mode", run: testRedoReusesConversationOnlyMode },
    { name: "cancelling navigation choice keeps conversation and workspace", run: testNavigationChoiceCancellationKeepsConversationAndWorkspace },
    { name: "conversation-only undo preserves manual changes as branch state", run: testConversationOnlyUndoPreservesManualChangesAsBranchState },
    { name: "conversation-only tree navigation anchors kept workspace", run: testTreeConversationOnlyAnchorsKeptWorkspace },
    { name: "conversation-only tree navigation supports branch summaries", run: testTreeConversationOnlySupportsBranchSummary },
    { name: "cancelled conversation-only summary does not leak its anchor", run: testCancelledConversationOnlySummaryDoesNotLeakAnchor },
    { name: "conversation-only navigation recovers pending workspace before anchoring", run: testConversationOnlyNavigationRecoversPendingWorkspaceBeforeAnchoring },
    { name: "conversation-only navigation preserves edits after pending recovery", run: testConversationOnlyNavigationPreservesEditsAfterPendingRecovery },
    { name: "undo/redo restores workspace", run: testUndoRedo },
    { name: "undo preserves manual changes before next turn", run: testManualChangesProtectedAcrossUndo },
    { name: "repeated undo walks back turn by turn", run: testRepeatedUndo },
    { name: "repeated undo crosses manual inter-turn changes", run: testRepeatedUndoAcrossManualInterTurnChanges },
    { name: "checkpoint and dirty tree guard", run: testCheckpointAndTreeGuard },
    { name: "tree switching restores branch-specific workspace", run: testTreeBranchSwitching },
    { name: "undo does not leak across sessions", run: testUndoDoesNotLeakAcrossSessions },
    { name: "undo works from tree-selected user node", run: testUndoWorksFromTreeSelectedUserNode },
    { name: "legacy snapshot entries rebuild turn snapshots", run: testLegacySnapshotEntriesRebuildTurnSnapshots },
    { name: "windows reserved names are excluded from snapshot paths", run: testWindowsReservedNamesAreExcludedFromSnapshotPaths },
    { name: "before commit reuses previous after commit when workspace unchanged", run: testBeforeCommitReusesPreviousAfterCommitWhenWorkspaceUnchanged },
    { name: ".pi files are managed except internal state", run: testPiFilesAreSnapshotManagedExceptInternalState },
    { name: "history is stored outside workspace", run: testHistoryIsStoredOutsideWorkspace },
    { name: "new session reuses workspace shadow repo", run: testNewSessionReusesWorkspaceShadowRepo },
    { name: "invalid current shadow repo is quarantined and rebuilt", run: testInvalidCurrentShadowRepoIsQuarantinedAndRebuilt },
    { name: "invalid reusable shadow repo is quarantined and rebuilt", run: testInvalidReusableShadowRepoIsQuarantinedAndRebuilt },
    { name: "failed shadow repo rebuild does not leave canonical repo", run: testFailedShadowRepoRebuildDoesNotLeaveCanonicalRepo },
    { name: "stale shadow repo lock is recovered", run: testStaleShadowRepoLockIsRecovered },
    { name: "unicode paths survive undo and redo", run: testUnicodePathsSurviveUndoRedo },
    { name: "undo and redo block on unsnapshotted manual changes", run: testUndoAndRedoBlockOnUnsnapshottedManualChanges },
    { name: ".gitignore stops managing ignored paths", run: testGitignoreStopsManagingIgnoredPaths },
    { name: "restore failure does not delete current workspace", run: testRestoreFailureDoesNotDeleteCurrentWorkspace },
    { name: "tree restore waits for a transient Windows file lock", run: testTreeRestoreWaitsForTransientWindowsFileLock },
    { name: "tree restore retries locked Windows file replacement", run: testTreeRestoreRetriesLockedWindowsFileReplacement },
    { name: "persistent Windows file lock reports cause and can be retried", run: testPersistentWindowsFileLockReportsCauseAndCanBeRetried },
    { name: "pending recovery preserves later manual edits", run: testPendingRecoveryPreservesLaterManualEdits },
  ];

  const testFilter = process.env.WORKSPACE_HISTORY_TEST_FILTER?.trim().toLowerCase();
  const selectedTests = testFilter
    ? tests.filter((test) => test.name.toLowerCase().includes(testFilter))
    : tests;
  assert.ok(selectedTests.length > 0, `No tests matched WORKSPACE_HISTORY_TEST_FILTER=${String(testFilter)}`);

  for (const test of selectedTests) {
    process.stdout.write(`RUN ${test.name}\n`);
    await test.run();
    process.stdout.write(`PASS ${test.name}\n`);
  }
}

await main();
