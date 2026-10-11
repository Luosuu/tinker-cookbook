"""Actual Claude Code CLI + MCP + buffered protocol replay; no model API call."""
import asyncio,json,os,sys,tempfile,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from protocol_gateway import Gateway,Ledger,sse

async def main():
    sent=[];calls=[];records=[]
    def response(content,stop='end_turn'):
        return dict(id=f'msg_{len(sent)}',type='message',role='assistant',model='test',content=content,stop_reason=stop,stop_sequence=None,usage={'input_tokens':20,'output_tokens':5})
    def provider(p):
        sent.append(p)
        if len(sent) in [1,3]:return response([{'type':'tool_use','id':f'call_{len(sent)}','name':'mcp__hpc__bash','input':{'command':'pwd'}}],'tool_use')
        if len(sent)==2:return response([{'type':'thinking','thinking':'Need another check.','signature':''}])
        return response([{'type':'text','text':'Finished the replay.'}])
    g=Gateway(provider,Ledger(max_requests=4),repair=True,journal=records.append)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            p=json.loads(self.rfile.read(int(self.headers['content-length'])))
            if 'count_tokens' in self.path:data=json.dumps({'input_tokens':20}).encode();kind='application/json'
            else:data=sse(g.complete(p));kind='text/event-stream'
            self.send_response(200);self.send_header('content-type',kind);self.send_header('content-length',str(len(data)));self.end_headers();self.wfile.write(data)
    proxy=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=proxy.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp);sock=str(d/'rpc.sock')
        async def handle(reader,writer):
            r=json.loads(await reader.readline());calls.append(r);writer.write(b'{"exit_code":0,"stdout":"/workspace/repo","stderr":""}\n');await writer.drain();writer.close();await writer.wait_closed()
        server=await asyncio.start_unix_server(handle,path=sock)
        config=d/'mcp.json';config.write_text(json.dumps({'mcpServers':{'hpc':{'command':sys.executable,'args':[str(Path(__file__).with_name('mcp_hpc.py'))],'env':{'HPC_BRIDGE_SOCKET':sock}}}}))
        env={**os.environ,'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{proxy.server_port}','ANTHROPIC_API_KEY':'mock-key','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1','HOME':str(d/'home')};(d/'home').mkdir()
        cli=Path(__file__).parent/'claude-runtime/node_modules/.bin/claude'
        proc=await asyncio.create_subprocess_exec(str(cli),'-p','Inspect using bash then report.','--model','thinkingmachines/Inkling-Small:peft:262144','--tools','','--mcp-config',str(config),'--strict-mcp-config','--allowedTools=mcp__hpc__bash','--permission-mode','dontAsk','--setting-sources','','--disable-slash-commands','--no-session-persistence','--output-format','stream-json','--verbose',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env=env)
        try:out,err=await asyncio.wait_for(proc.communicate(),90)
        finally:
            if proc.returncode is None:proc.kill();await proc.wait()
            server.close();await server.wait_closed();await asyncio.to_thread(proxy.shutdown)
        Path(sys.argv[1]).write_text(json.dumps({'requests':len(sent),'tool_calls':len(calls),'ledger':vars(g.ledger),'returncode':proc.returncode,'stdout':out.decode(),'stderr':err.decode(),'wire':records},indent=2))
        assert proc.returncode==0,(proc.returncode,err.decode())
        assert len(sent)==4 and len(calls)==2,(len(sent),len(calls),out.decode())
        assert g.ledger.repairs==1
        print('CLI_REPLAY_PASS requests=4 tools=2 repairs=1')
asyncio.run(main())
