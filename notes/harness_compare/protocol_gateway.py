"""Opt-in Tinker Anthropic boundary, with episode-wide admission and bounded repair.

Never execute text as code. Only provider-produced, schema-valid tool_use blocks
can reach the client. Repairs are new model requests, charged to the same ledger.
"""
from __future__ import annotations
import copy
import json
import threading
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import httpx

UPSTREAM = 'https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api'
REPAIR = ('The previous response did not contain an executable tool call or a final answer. '
          'Continue the original task. If an action is needed, use a structured tool_use '
          'block matching the declared tool schema, not tool-call syntax in text. '
          'Otherwise provide your final answer. Do not repeat an already executed action.')

class BudgetExhausted(Exception):
    pass

@dataclass
class Ledger:
    max_requests: int = 40
    max_tools: int = 80
    max_output: int = 65536
    max_input: int = 5_000_000
    max_seconds: float = 3600
    max_repairs: int = 2
    requests: int = 0
    tools: int = 0
    output: int = 0
    input: int = 0
    repairs: int = 0
    stop: str | None = None
    usage_complete: bool = True
    started: float = 0

    def __post_init__(self):
        if min(self.max_requests,self.max_tools,self.max_output,self.max_input,self.max_seconds) <= 0 or not 0 <= self.max_repairs <= 2:
            raise ValueError('Invalid episode budgets')
        self.started = time.monotonic()

    def reason(self):
        if self.stop: return self.stop
        for test, reason in [(self.requests >= self.max_requests,'request_budget'),
                             (self.tools >= self.max_tools,'tool_budget'),
                             (self.output >= self.max_output,'output_budget'),
                             (self.input >= self.max_input,'input_budget'),
                             (time.monotonic()-self.started >= self.max_seconds,'wall_admission_budget')]:
            if test: return reason
        return None

    def admit(self, requested):
        reason=self.reason()
        if reason:
            self.stop=reason; raise BudgetExhausted(reason)
        if not isinstance(requested,int) or isinstance(requested,bool) or requested < 1: raise ValueError('Invalid max_tokens')
        self.requests += 1
        return min(requested,self.max_output-self.output)

    def admit_tool(self):
        # The last admitted model request may still execute its tool calls.
        if self.tools >= self.max_tools or time.monotonic()-self.started >= self.max_seconds:
            self.stop='tool_budget' if self.tools >= self.max_tools else 'wall_admission_budget'
            return False
        self.tools += 1
        return True

    def account(self, response):
        usage=response.get('usage',{})
        if any(not isinstance(usage.get(k),int) or isinstance(usage.get(k),bool) or usage[k]<0 for k in ('input_tokens','output_tokens')):
            self.usage_complete=False; self.stop='missing_usage'; return
        self.input += sum((usage.get(k) or 0) for k in ('input_tokens','cache_read_input_tokens','cache_creation_input_tokens'))
        self.output += usage['output_tokens']
        if self.output>self.max_output: self.stop='provider_output_overrun'


def response_kind(response, tools):
    blocks=response.get('content',[])
    if response.get('stop_reason') in {'max_tokens','refusal','pause_turn'}: return 'terminal_'+response['stop_reason']
    calls=[b for b in blocks if b.get('type')=='tool_use']
    if calls:
        names={t['name'] for t in tools}; ids=set()
        for b in calls:
            args=b.get('input'); ident=b.get('id')
            if (b.get('name') not in names or b.get('name')!='mcp__hpc__bash' or
                not isinstance(ident,str) or not ident or ident in ids or
                not isinstance(args,dict) or set(args)!={'command'} or not isinstance(args['command'],str)):
                return 'invalid_tool'
            ids.add(ident)
        return 'tool_use'
    text='\n'.join(b.get('text','') for b in blocks if b.get('type')=='text')
    if 'mcp__' in text and ('"args"' in text or '"command"' in text): return 'text_tool'
    if text.strip(): return 'final_text'
    if blocks and all(b.get('type')=='thinking' for b in blocks): return 'thinking_only'
    return 'empty'

class Gateway:
    def __init__(self, send, ledger, *, repair=False, journal=lambda event:None):
        self.send=send; self.ledger=ledger; self.repair=repair; self.journal=journal
        self.lock=threading.RLock()

    def complete(self, request):
        with self.lock:
            payload=copy.deepcopy(request); payload['stream']=False
            payload['output_config']={'effort':'high'}
            payload['temperature']=1.0
            while True:
                payload['max_tokens']=self.ledger.admit(min(payload.get('max_tokens',16384),16384))
                self.journal({'type':'request','index':self.ledger.requests,'payload':payload})
                # No model-error retry. A thrown transport error consumes admission.
                response=self.send(payload)
                self.ledger.account(response)
                kind=response_kind(response,payload.get('tools',[]))
                self.journal({'type':'response','index':self.ledger.requests,'kind':kind,'response':response,'ledger':asdict(self.ledger)})
                if kind=='tool_use':
                    result=copy.deepcopy(response);result['stop_reason']='tool_use';return result
                if kind.startswith('terminal_') or kind=='final_text': return response
                if (not self.repair or self.ledger.repairs>=self.ledger.max_repairs or self.ledger.reason()):
                    # Invalid structured calls must never reach the executor.
                    if kind=='invalid_tool':
                        response=copy.deepcopy(response)
                        response['content']=[{'type':'text','text':'Protocol failure: invalid tool call; no action executed.'}]
                        response['stop_reason']='end_turn'
                    self.ledger.stop=self.ledger.reason() or 'protocol_unresolved'
                    return response
                self.ledger.repairs+=1
                # Drop invalid tool blocks; no fabricated tool_result or tool execution.
                content=[b for b in response.get('content',[]) if b.get('type') in {'text','thinking'}]
                if not content: content=[{'type':'text','text':'[Invalid tool call omitted; no action executed.]'}]
                payload['messages'] += [{'role':'assistant','content':content},{'role':'user','content':REPAIR}]


def sse(response):
    """Re-emit buffered Anthropic messages without exposing a partial tool call."""
    base={**response,'content':[],'stop_reason':None,'stop_sequence':None}
    events=[('message_start',{'type':'message_start','message':base})]
    for index,block in enumerate(response.get('content',[])):
        start=copy.deepcopy(block)
        if block['type']=='tool_use':
            start['input']={};delta={'type':'input_json_delta','partial_json':json.dumps(block['input'])}
        elif block['type']=='thinking':
            start['thinking']='';start.pop('signature',None)
            delta={'type':'thinking_delta','thinking':block.get('thinking','')}
        else:
            start['text']='';delta={'type':'text_delta','text':block.get('text','')}
        events += [('content_block_start',{'type':'content_block_start','index':index,'content_block':start}),
                   ('content_block_delta',{'type':'content_block_delta','index':index,'delta':delta})]
        if block.get('signature'):
            events.append(('content_block_delta',{'type':'content_block_delta','index':index,'delta':{'type':'signature_delta','signature':block['signature']}}))
        events.append(('content_block_stop',{'type':'content_block_stop','index':index}))
    events += [('message_delta',{'type':'message_delta','delta':{'stop_reason':response.get('stop_reason'),'stop_sequence':response.get('stop_sequence')},'usage':response.get('usage',{})}),('message_stop',{'type':'message_stop'})]
    return ''.join(f'event: {name}\ndata: {json.dumps(event)}\n\n' for name,event in events).encode()

class GatewayServer:
    def __init__(self, key, directory:Path, ledger:Ledger, *, repair=False, upstream=UPSTREAM):
        if upstream!=UPSTREAM: raise ValueError('Only the explicitly supported Tinker Anthropic endpoint is allowed')
        self.log=(directory/'wire.jsonl').open('x'); self.ledger=ledger
        def journal(event):
            self.log.write(json.dumps(event)+'\n');self.log.flush()
        self.client=httpx.Client(timeout=None,headers={'x-api-key':key,'anthropic-version':'2023-06-01'})
        def send(payload):
            response=self.client.post(upstream+'/v1/messages',json=payload)
            response.raise_for_status();return response.json()
        self.gateway=Gateway(send,ledger,repair=repair,journal=journal)
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                try:
                    request=json.loads(self.rfile.read(int(self.headers.get('content-length','0'))))
                    path=urlsplit(self.path).path
                    if path=='/v1/messages/count_tokens':
                        r=owner.client.post(upstream+path,json=request);r.raise_for_status()
                        data=r.content;kind='application/json'
                    elif path=='/v1/messages':
                        response=owner.gateway.complete(request)
                        data=sse(response) if request.get('stream') else json.dumps(response).encode()
                        kind='text/event-stream' if request.get('stream') else 'application/json'
                    else: raise ValueError('Unsupported endpoint')
                    self.send_response(200)
                except Exception as exc:
                    # Non-retryable response: CLI retries must not multiply model errors.
                    data=json.dumps({'type':'error','error':{'type':'invalid_request_error','message':type(exc).__name__}}).encode();kind='application/json'
                    self.send_response(400)
                    journal({'type':'gateway_error','error_type':type(exc).__name__})
                self.send_header('content-type',kind);self.send_header('content-length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.client.close();self.log.close()
