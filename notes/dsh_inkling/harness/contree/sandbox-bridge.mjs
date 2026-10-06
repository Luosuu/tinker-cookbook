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

// A read-only streak is the measured dominant failure mode: episodes that never write
// anything almost never pass. The nudge rides on the tool result, so it costs no model
// request and cannot consume budget. Set HPC_NUDGE_AFTER to 0 to disable it.
const NUDGE_AFTER = Number(process.env.HPC_NUDGE_AFTER || '12');
const NUDGE_EVERY = Number(process.env.HPC_NUDGE_EVERY || '4');

const WRITE_PATTERN = /cat\s*>|tee\s|sed\s+-i|patch\s+-|git\s+apply|>>?\s*\/workspace|<<\s*'?[A-Z]+'?/;
const BUILD_PATTERN = /\b(cmake|ninja|ctest|make)\b/;

export function apply(ctx) {
  let calls = 0;
  let readStreak = 0;

  ctx.tools.register(defineTool({
    name: 'bash',
    description:
      'Run a shell command in the isolated task container. The working directory and ' +
      'filesystem persist across calls. Use this to inspect, edit, compile, and debug ' +
      'the repository.',
    parameters: {
      command: {
        type: 'string',
        required: true,
        description: 'Shell command to execute inside the task container.',
      },
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

      const mutating = WRITE_PATTERN.test(command) || BUILD_PATTERN.test(command);
      readStreak = mutating ? 0 : readStreak + 1;
      const nudging =
        NUDGE_AFTER > 0 &&
        readStreak >= NUDGE_AFTER &&
        (readStreak - NUDGE_AFTER) % NUDGE_EVERY === 0;
      const nudge = nudging
        ? `\n\n[harness] ${readStreak} consecutive commands without editing or building. ` +
          `Budget spent: ${calls}/${MAX_TOOL_CALLS}. Make your best edit now and build it; ` +
          `an edit you can compile and correct beats more reading.`
        : '';

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
