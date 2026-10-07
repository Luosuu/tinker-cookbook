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

const WRITE_PATTERN = /cat\s*>|tee\s|sed\s+-i|patch\s+-|git\s+apply|>>?\s*\/workspace|<<\s*'?[A-Z]+'?/;
const BUILD_PATTERN = /\b(cmake|ninja|ctest|make)\b/;

export function apply(ctx) {
  let calls = 0;
  let readStreak = 0;
  let edited = false;
  let verified = false;
  let verifyNudged = false;
  let coverageNudged = false;
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
      if (isBuild) verified = true;
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
