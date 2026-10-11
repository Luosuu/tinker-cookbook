"""Replay recorded real provider messages against the final adapter, no API calls."""
import json,sys
from pathlib import Path
from protocol_gateway import Gateway, Ledger, BudgetExhausted
rows=[]
for path in sorted(Path(sys.argv[1]).glob('claude*/*/wire.jsonl')):
 events=[json.loads(s) for s in path.read_text().splitlines()]
 requests=[e['payload'] for e in events if e['type']=='request'];responses=[e['response'] for e in events if e['type']=='response'];index=0
 assert len(requests)==len(responses)
 def send(payload):
  global index
  assert payload==requests[index],f'Request differs at {path}:{index}'
  response=responses[index];index+=1;return response
 repair=path.parent.parent.name=='claude_protocol'
 g=Gateway(send,Ledger(),repair=repair)
 while index<len(requests):
  response=g.complete(requests[index])
  for block in response.get('content',[]):
   if block.get('type')=='tool_use':assert g.ledger.admit_tool()
 original=json.loads((path.parent/'gateway-ledger.json').read_text())
 for k in ['requests','repairs','input','output','tools','usage_complete']:assert getattr(g.ledger,k)==original[k],(path,k,getattr(g.ledger,k),original[k])
 rows.append({'path':str(path),'requests':index,'repairs':g.ledger.repairs,'tools':g.ledger.tools,'matched':True})
Path(sys.argv[2]).write_text(json.dumps(rows,indent=2));assert len(rows)==4,rows
print('RECORDED_REPLAY_PASS',len(rows),'episodes',sum(r['requests'] for r in rows),'requests')
