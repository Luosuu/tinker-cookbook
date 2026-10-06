// One bash tool backed by the isolated task sandbox.
//
// The harness process runs on the host because it needs the model credential; the
// repository lives inside a network-isolated Apptainer container. This plugin is the only
// bridge between them, so every repository command the model runs is a `run_command` call
// against the sandbox. That keeps the credential out of the container and keeps the
// working directory persistent across calls.
//
// It also counts tool calls, because `dsh --profile headless` exposes no tool-call cap and
// the reference baseline is defined at 80.

import net from 'node:net';
import { defineTool } from '@deepseek-ai/dsh-tools';

export const name = 'hpc-sandbox-bash';
export const inject = ['tools'];

const MAX_TOOL_CALLS = Number(process.env.HPC_MAX_TOOL_CALLS || '80');

export function apply(ctx) {
  let calls = 0;
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
