import json
import pytest
from protocol_gateway import Gateway, Ledger, BudgetExhausted, response_kind, sse
TOOLS=[{'name':'mcp__hpc__bash'}]
def response(content, stop='end_turn', usage=None):
    return dict(id='msg_1',type='message',role='assistant',model='test',content=content,stop_reason=stop,stop_sequence=None,usage=usage if usage is not None else {'input_tokens':20,'output_tokens':5})
def thought():return response([{'type':'thinking','thinking':'Investigating','signature':'sig'}])
def call():return response([{'type':'tool_use','id':'call_1','name':'mcp__hpc__bash','input':{'command':'pwd'}}])
def request():return dict(messages=[{'role':'user','content':'fix'}],tools=TOOLS,max_tokens=100,stream=True)
def test_thinking_repair_charges_shared_ledger_and_preserves_history():
    sent=[];responses=iter([thought(),call()]);g=Gateway(lambda p:(sent.append(json.loads(json.dumps(p))),next(responses))[1],Ledger(max_output=9),repair=True)
    result=g.complete(request())
    assert result['stop_reason']=='tool_use'
    assert [p['max_tokens'] for p in sent]==[9,4]
    assert len(sent[1]['messages'])==3
    assert (g.ledger.requests,g.ledger.repairs,g.ledger.input)==(2,1,40)
    assert g.ledger.stop=='provider_output_overrun' # mock violated the second cap

def test_repairs_bounded_and_do_not_execute_text():
    text=response([{'type':'text','text':'{"name":"mcp__hpc__bash","args":{"command":"pwd"}}'}])
    g=Gateway(lambda p:text,Ledger(max_repairs=2),repair=True)
    assert g.complete(request())==text
    assert g.ledger.requests==3 and g.ledger.tools==0 and g.ledger.stop=='protocol_unresolved'

def test_request_budget_is_admitted_before_dispatch():
    sent=[];g=Gateway(lambda p:(sent.append(p),thought())[1],Ledger(max_requests=1),repair=True)
    g.complete(request())
    with pytest.raises(BudgetExhausted):g.complete(request())
    assert len(sent)==1

def test_structured_tool_stop_normalized_but_invalid_tool_never_delivered():
    assert Gateway(lambda p:call(),Ledger()).complete(request())['stop_reason']=='tool_use'
    invalid=call();invalid['content'][0]['input']={'cmd':'pwd'}
    g=Gateway(lambda p:invalid,Ledger(max_repairs=0))
    assert all(b['type']!='tool_use' for b in g.complete(request())['content'])
    assert g.ledger.stop=='protocol_unresolved'

@pytest.mark.parametrize('reason',['max_tokens','refusal','pause_turn'])
def test_terminal_reasons_not_retried(reason):
    r=thought();r['stop_reason']=reason;g=Gateway(lambda p:r,Ledger());g.complete(request())
    assert g.ledger.requests==1

def test_transport_errors_not_retried():
    def fail(p):raise OSError('network')
    g=Gateway(fail,Ledger())
    with pytest.raises(OSError):g.complete(request())
    assert g.ledger.requests==1

def test_all_tools_share_admission():
    ledger=Ledger(max_tools=1);assert ledger.admit_tool();assert not ledger.admit_tool();assert ledger.tools==1

def test_missing_usage_fails_closed():
    r=thought();r['usage']={};g=Gateway(lambda p:r,Ledger());g.complete(request())
    assert not g.ledger.usage_complete and g.ledger.stop=='missing_usage' and g.ledger.requests==1

def test_sse_tool_arguments_and_signature_retained():
    r=call();r['content'].insert(0,thought()['content'][0]);data=sse(r).decode()
    events=[json.loads(line[6:]) for line in data.splitlines() if line.startswith('data: ')]
    deltas=[e['delta'] for e in events if e['type']=='content_block_delta']
    assert {'type':'signature_delta','signature':'sig'} in deltas
    assert any(d.get('partial_json')=='{"command": "pwd"}' for d in deltas)

def test_null_unsupported_cache_counters_do_not_break_usage():
    r=thought();r['usage'].update(cache_read_input_tokens=None,cache_creation_input_tokens=None)
    g=Gateway(lambda p:r,Ledger(max_repairs=0));g.complete(request())
    assert g.ledger.input==20 and g.ledger.usage_complete

def test_repair_envelope_cannot_be_expanded_without_changing_code():
    with pytest.raises(ValueError): Ledger(max_repairs=3)


def test_repair_is_opt_in_and_default_preserves_validation_and_accounting():
    g = Gateway(lambda p: thought(), Ledger())
    g.complete(request())
    assert (g.ledger.requests, g.ledger.repairs, g.ledger.stop) == (1, 0, 'protocol_unresolved')
    g = Gateway(lambda p: call(), Ledger())
    assert g.complete(request())['stop_reason'] == 'tool_use'
    assert g.ledger.requests == 1
