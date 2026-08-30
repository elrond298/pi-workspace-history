# pi-workspace-history

[Chinese version / 中文版](./README.zh-CN.md)

Real workspace undo/redo for Pi.

Bring OpenCode style `/undo` to Pi, with the kind of workspace rollback safety that makes Claude Code feel trustworthy.

![workspace-history demo](./demo.gif)

## Why It Matters

- Undo the real workspace, not just chat history
- Roll back agent turns with confidence
- Restore branch-specific workspace state with `/tree`
- Rewind conversation context without discarding current files
- Protect manual edits with `/checkpoint`

## What It Is

`workspace-history` is a workspace history plugin for `@mariozechner/pi-coding-agent`.

It is not just an extra `/undo` command. The goal is to keep chat history navigation and real workspace state coordinated, while letting the user choose whether a navigation should restore files or keep the current workspace.

Its core goal is:

```text
When the user navigates to any node in the chat history tree,
they can restore both conversation and workspace state,
or rewind only the conversation while keeping current files.
```

In other words:

- `/tree` is the actual time machine
- `/undo` is a shortcut that moves one step backward through `/tree`
- `/redo` moves back to the location that was just undone

## Why It Exists

When using an agent for coding, these problems happen often:

- The agent breaks working code
- The agent deletes files by mistake
- The agent creates many useless files
- You want to go back to an earlier branch and try a different path
- You manually edit, create, or delete files between agent turns
- You do not want bad context to keep affecting later reasoning

This plugin does not try to solve simple text-editor undo. It coordinates whole-workspace snapshots with chat history navigation, including conversation-only rewinds that preserve current files.

Its value is:

- `/undo` can revert a whole agent turn instead of partially rolling back files
- `/tree` becomes real workspace history navigation, not just chat navigation
- You can move safely between historical branches
- Manual changes made between agent turns are preserved correctly
- Plugin state stays isolated from the user project Git history

## Requirements It Is Designed Around

This plugin is built around the following concrete requirements:

1. Record a `before` snapshot before each agent turn starts.
2. Record an `after` snapshot after each agent turn completes.
3. Let the user choose whether `/tree` or `/undo` restores both conversation and workspace, or conversation only.
4. When workspace restore is selected, `/undo` must restore the real state from before that turn started, not just the previous post-agent state.
5. If the user manually deletes files, edits code, or creates files before the next prompt, those changes must be captured in the next `before` snapshot.
6. If the workspace contains unsnapshotted manual changes, workspace restore must not silently overwrite them. Conversation-only navigation preserves and anchors those changes automatically.
7. Internal plugin state must stay isolated from the user project's main Git history.
8. Multiple sessions must be isolated so snapshots and redo state do not leak across sessions.

## Main Features

- `/undo`
  - Choose between restoring conversation and workspace together or rewinding conversation only
  - The existing combined restore is the first/default choice
  - Put the original user prompt back into the editor for retrying

- `/redo`
  - Restore the location that was just undone
  - Reuse the mode chosen by `/undo`, without asking again

- `/checkpoint [label]`
  - Save the current workspace as a manual checkpoint
  - Protect manual changes before the next prompt is sent

- Workspace restore through `/tree`
  - Choose whether to restore the matching workspace state after selecting a history node
  - Applies to `/tree` and Pi's double-Escape tree shortcut
  - Supports moving between historical branches

- Dirty guard
  - Blocks risky workspace restore when the workspace contains unsnapshotted manual changes
  - Conversation-only navigation keeps and snapshots those changes instead of overwriting them

- Session isolation
  - Each session uses its own shadow git and redo state
  - Prevents a new session from undoing into an older session's history

## How It Works

The plugin stores snapshots in an internal shadow git repository instead of relying on the user's project `.git` history.

For conversation-only navigation, the plugin snapshots the current files before moving the conversation and anchors that snapshot to the new history branch. Later `/tree` and `/redo` operations therefore remain consistent instead of treating the intentionally kept files as unknown manual changes. Cancelling the choice leaves both conversation and workspace unchanged. In non-interactive modes, navigation keeps the previous combined conversation-and-workspace behavior.

Default snapshot scope:

- Git tracked files
- Untracked files that are not ignored
- Paths matched by the workspace `.gitignore` are filtered out even if they were previously snapshotted

Default exclusions:

- `.git/`
- `.pi/workspace-history/`
- `node_modules/`
- `dist/`
- `build/`
- `.cache/`
- `.next/`
- `.turbo/`
- `coverage/`
- `.env`
- `.env.*`

During restore, the plugin restores only the managed file set instead of doing a broad destructive cleanup of the entire workspace.

On Windows, restore operations retry briefly locked managed files. If a lock persists, navigation is cancelled without skipping the file and the notification identifies the Git file operation that failed. Pending recovery survives a session or extension reload; edits made after the failed restore are never overwritten automatically and can be preserved with `/checkpoint`.

The plugin validates each session's shadow repository before using it. If the current session repository or the workspace reusable repository is invalid, it is preserved beside the replacement as `repo.git.invalid-<timestamp>-<uuid>` and a usable repository is rebuilt automatically. Snapshotting then continues normally, but older snapshots stored only in the invalid repository may be unavailable. Invalid repositories belonging to other sessions are skipped without modifying them.

## Configuration

Configure via Pi settings:

- Global: `~/.pi/agent/settings.json`
- Project: `.pi/settings.json`

Example:

```json
{
  "workspaceHistory": {
    "storageDir": "D:\\pi-history",
    "maxSessionsPerWorkspace": 3,
    "maxWorkspaces": 10
  }
}
```

Settings:

- `workspaceHistory.storageDir`
  - External storage root for shadow history
  - Default: `~/.pi/agent/state/workspace-history`
- `workspaceHistory.maxSessionsPerWorkspace`
  - Keep only the most recently used sessions per workspace
  - Default: `3`
- `workspaceHistory.maxWorkspaces`
  - Keep only the most recently used workspaces globally
  - Default: `10`
- `workspaceHistory.enabled`
  - `auto` (default) disables the plugin outside project-like directories
  - `true` forces it on
  - `false` disables it completely
- `workspaceHistory.allowHomeDirectory`
  - Allow enabling in the user home directory
  - Default: `false`
- `workspaceHistory.requireProjectMarker`
  - Require a project marker such as `.git` or `package.json`
  - Default: `true`
- `workspaceHistory.maxScanFiles` / `workspaceHistory.maxScanDirs` / `workspaceHistory.maxScanMs`
  - Safety budget for workspace scanning
- `workspaceHistory.gitTimeoutMs`
  - Timeout for internal git operations

## Installation And Usage

Install from a package source:

```bash
pi install npm:pi-workspace-history
```

After publishing this package to npm, users can install it directly with the command above.

Or install from a local checkout:

```bash
pi install /path/to/workspace-history
```

## Local Development

This repository is also configured for direct local extension loading while developing:

```text
.pi/extensions/workspace-history.ts
.pi/settings.json
```

Start `pi` in this directory, or run `/reload` to test local changes.

You can also place `workspace-history.ts` in:

- `~/.pi/agent/extensions/`
- `.pi/extensions/`

## Testing

Run automated tests:

```bash
npm test
```

Run type checking:

```bash
npm run typecheck
```

## Recent Changes

- History is stored outside the workspace by default
- Added `workspaceHistory.storageDir`
- Added retention limits for sessions and workspaces
- Reduced runtime overhead with cached settings/paths and throttled cleanup

## Storage Layout

The plugin stores history outside the workspace by default:

```text
~/.pi/agent/state/workspace-history/
  workspaces/
    <workspaceHash>/
      meta.json
      sessions/
        <sessionId>/
          repo.git/
          redo.json
          meta.json
  logs/
    timemachine.log
```

Notes:

- History is isolated from the user's project `.git` history
- Invalid shadow repositories are preserved as `repo.git.invalid-<timestamp>-<uuid>` when automatic recovery is needed
- Old workspace-local `.pi/workspace-history/` state is not migrated automatically
- Cleanup is LRU-style based on recent use
- In `auto` mode, the plugin disables itself in broad directories like the user home folder to avoid expensive scans and startup stalls
