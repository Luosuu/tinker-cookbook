// One bash tool backed by the isolated task sandbox.
//
// The harness process runs on the host because it needs the model credential; the
// repository lives inside a network-isolated sandbox. This plugin is the only bridge
// between them, so every repository command the model runs is a `run_command` call
// against the sandbox. That keeps the credential out of the container and keeps the
// working directory persistent across calls.
//
// It also counts tool calls, because `dsh --profile headless` exposes no tool-call cap and
// the reference baseline is defined at 80, and it nudges a long read-only streak, which is
// the measured dominant failure mode.

import net from 'node:net';
import { defineTool } from '@deepseek-ai/dsh-tools';

export const name = 'hpc-sandbox-bash';
export const inject = ['tools'];

const MAX_TOOL_CALLS = Number(process.env.HPC_MAX_TOOL_CALLS || '80');

// The binding limit is the proxy's model-request budget (40), not the tool-call cap (80):
// across 700 episodes no run ever reached 80 tool calls while 453 hit 40 requests, and the
// two counts track each other at a ratio of 0.99 because each request returns one tool
// call. Quoting the tool cap to the model understates consumption by roughly half and makes
// it think it has room it does not have, so the reported budget is the request budget and
// `calls` is a faithful proxy for requests spent.
const MAX_REQUESTS = Number(process.env.HPC_MAX_REQUESTS || '40');

// Two independent nudge policies, both riding on the tool result so they cost no model
// request and cannot consume budget.
//
// Read-streak: fires after many consecutive non-edit, non-build commands. Measured
// inconclusive-to-harmful, so it defaults off.
const NUDGE_AFTER = Number(process.env.HPC_NUDGE_AFTER || '12');
const NUDGE_EVERY = Number(process.env.HPC_NUDGE_EVERY || '4');

// Verify-after-edit: fires once the episode has edited the repository but has still not
// obtained any compiler or test feedback. This targets the measured bottleneck -- among
// episodes that edit, those that never build pass 30.1% while those that do pass 63.9%.
// It is opt-in because it changes behaviour, and it fires only once so the model is not
// nagged.
const VERIFY_NUDGE = process.env.HPC_VERIFY_NUDGE === '1';

// Coverage reminder: fires once, late in the budget, if the episode has edited only one
// file. Grounded in the strongest surviving measurement: patches covering under half the
// files the reference fix touches pass 3%, while those covering half or more pass 62%, and
// the task-controlled sign test gives 12 against 0 (p = 0.0005). 61% of failures miss at
// least one required file, and 68% of those files the model had already referenced earlier
// and dropped. Failed patches are also larger than solved ones, so this is a completeness
// problem rather than an under-editing one.
const COVERAGE_NUDGE = process.env.HPC_COVERAGE_NUDGE === '1';
const COVERAGE_AT = Number(process.env.HPC_COVERAGE_AT || '30');

// Changed-set echo: append the repository files this episode has edited to every tool
// result, once it has edited at least one. This is factual rather than instructional -- it
// states what the model has changed, not what it should do -- because four reminders have
// already failed and a stale working set is a plausible reason. Episodes that inspect
// their own diff reach 90% gold-file coverage against 71% for those that never do, but
// only 8% ever run that check.
const ECHO_CHANGED = process.env.HPC_ECHO_CHANGED === '1';

// First-edit pressure: 19% of episodes end with an empty patch, and they are not failing to
// locate the fix -- 96% of them reference the file the reference patch changes, at call 5.7 of
// 40 on average, and then read for the remaining 34 requests without writing anything. A single
// reminder fired on all 17 such episodes of one run and none of them edited.
//
// So this fires repeatedly from early in the run, and it states the budget arithmetic rather
// than restating the task: how many requests are left and that nothing has been changed yet.
const FIRST_EDIT_NUDGE = process.env.HPC_FIRST_EDIT_NUDGE === '1';
const FIRST_EDIT_AT = Number(process.env.HPC_FIRST_EDIT_AT || '8');

// Re-arming verify reminder: the once-only verify nudge could not help an episode that
// built early and then edited repeatedly without building again, which is a real
// population -- 17 of 72 failing episodes with three or more edits never built at all, and
// one corpus example made 8 edits in 40 calls with zero builds. This variant re-arms after
// each build and fires again once the episode has made N edits since that build.
const VERIFY_REARM = process.env.HPC_VERIFY_REARM === '1';
const EDITS_BEFORE_NUDGE = Number(process.env.HPC_EDITS_BEFORE_NUDGE || '3');

const WRITE_PATTERN = /cat\s*>|tee\s|sed\s+-i|patch\s+-|git\s+apply|>>?\s*\/workspace|<<\s*'?[A-Z]+'?/;
const BUILD_PATTERN = /\b(cmake|ninja|ctest|make)\b/;

export function apply(ctx) {
  let calls = 0;
  let readStreak = 0;
  let edited = false;
  let verified = false;
  let verifyNudged = false;
  let coverageNudged = false;
  let editsSinceBuild = 0;
  let rearmNudges = 0;
  const editedPaths = new Set();

  ctx.tools.register(defineTool({
    name: 'bash',
    description: "Execute a bash command in the sandbox environment.",
    parameters: {
      command: { type: 'string', required: true },
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute({ command }, execution) {
      const socketPath = process.env.HPC_BRIDGE_SOCKET;
      if (!socketPath) throw new Error('HPC_BRIDGE_SOCKET is not set');

      calls += 1;
      if (calls > MAX_TOOL_CALLS) {
        // Terminal for this episode: the sandbox is still graded for whatever candidate
        // the filesystem holds, matching the reference cap semantics.
        return `tool-call budget exhausted (${calls - 1}/${MAX_TOOL_CALLS})`;
      }

      const isWrite = WRITE_PATTERN.test(command);
      const isBuild = BUILD_PATTERN.test(command);
      if (isWrite) {
        edited = true;
        for (const path of command.matchAll(/[\w./-]*\/(?:core|containers|algorithms|simd|tpls)\/[\w./-]+/g)) {
          editedPaths.add(path[0]);
        }
      }
      if (isBuild) {
        verified = true;
        // Re-arm the reminder: a build resets the count of unverified edits.
        editsSinceBuild = 0;
      }
      if (isWrite) editsSinceBuild += 1;
      const mutating = isWrite || isBuild;
      readStreak = mutating ? 0 : readStreak + 1;

      const nudging =
        NUDGE_AFTER > 0 &&
        readStreak >= NUDGE_AFTER &&
        (readStreak - NUDGE_AFTER) % NUDGE_EVERY === 0;
      let nudge = nudging
        ? `\n\n[harness] ${readStreak} consecutive commands without editing or building. ` +
          `Budget spent: ${calls}/${MAX_REQUESTS} model requests. Make your best edit now and build it; ` +
          `an edit you can compile and correct beats more reading.`
        : '';

      // Fires every FIRST_EDIT_AT requests while the repository is still untouched.
      if (
        FIRST_EDIT_NUDGE &&
        !edited &&
        calls >= FIRST_EDIT_AT &&
        (calls - FIRST_EDIT_AT) % FIRST_EDIT_AT === 0
      ) {
        nudge +=
          `\n\n[harness] No file in the repository has been modified yet. ` +
          `${calls}/${MAX_REQUESTS} model requests used, ${MAX_REQUESTS - calls} remaining. ` +
          `Reading further cannot change the outcome: an empty patch scores zero regardless ` +
          `of how well the problem is understood. Write your best edit now, build it, and ` +
          `refine from the compiler output.`;
      }

      // Factual echo of the changed set, on every result once an edit has happened.
      if (ECHO_CHANGED && editedPaths.size > 0) {
        const list = [...editedPaths].sort().join("\n  ");
        nudge +=
          `\n\n[harness] Repository files edited so far (${editedPaths.size}):\n  ${list}`;
      }

      // Fires at most once, late, and only when the episode has edited a single file.
      if (
        COVERAGE_NUDGE &&
        !coverageNudged &&
        calls >= COVERAGE_AT &&
        editedPaths.size <= 1
      ) {
        coverageNudged = true;
        nudge +=
          `\n\n[harness] You have edited ${editedPaths.size} file(s) and used ${calls}/${MAX_REQUESTS} ` +
          `model requests. Most fixes in this repository require a companion change in a ` +
          `related header, module interface or overload, and patches that touch only one ` +
          `file rarely pass. Before continuing, re-read the files you found earlier and ` +
          `check whether your change also needs to be applied or declared elsewhere.`;
      }

      // Re-arming variant: fires whenever N edits have accumulated since the last build.
      if (
        VERIFY_REARM &&
        editsSinceBuild >= EDITS_BEFORE_NUDGE &&
        rearmNudges < 4
      ) {
        rearmNudges += 1;
        editsSinceBuild = 0;
        nudge +=
          `\n\n[harness] ${editsSinceBuild + EDITS_BEFORE_NUDGE} consecutive edits with no ` +
          `build. Budget spent: ${calls}/${MAX_REQUESTS} model requests. Compile now and ` +
          `read the compiler output before making further changes; unverified edits cannot ` +
          `be corrected.`;
      }

      // Fires at most once, and only after an edit has gone unverified by a real build.
      if (VERIFY_NUDGE && edited && !verified && !verifyNudged) {
        verifyNudged = true;
        nudge +=
          `\n\n[harness] You have edited the repository but not compiled it. ` +
          `Budget spent: ${calls}/${MAX_REQUESTS} model requests. Build the project now and read the ` +
          `compiler output before continuing; unverified edits are the most common way ` +
          `this task is failed.`;
      }

      return await new Promise((resolve, reject) => {
        const socket = net.createConnection(socketPath);
        let buffer = '';
        let settled = false;
        const finish = (error, value) => {
          if (settled) return;
          settled = true;
          socket.destroy();
          execution.signal?.removeEventListener('abort', aborted);
          if (error) reject(error);
          else resolve(value + nudge);
        };
        const aborted = () => finish(new Error('Sandbox call aborted'));
        execution.signal?.addEventListener('abort', aborted, { once: true });
        socket.on('connect', () => socket.write(JSON.stringify({ command }) + '\n'));
        socket.on('data', (chunk) => {
          buffer += chunk.toString('utf8');
          const newline = buffer.indexOf('\n');
          if (newline < 0) return;
          try {
            const response = JSON.parse(buffer.slice(0, newline));
            if (response.error) finish(new Error(response.error));
            else finish(null, `exit_code=${response.exit_code}\nstdout:\n${response.stdout}\nstderr:\n${response.stderr}`);
          } catch (error) {
            finish(error);
          }
        });
        socket.on('error', (error) => finish(error));
        socket.on('end', () => finish(new Error('Sandbox bridge closed without response')));
      });
    },
  }));
}
