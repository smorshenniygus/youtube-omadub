const {test}=require('node:test');
const assert=require('node:assert/strict');
const {Session,requestSpec}=require('../dubbing.js');
test('heartbeat route requires a valid explicit client lease',()=>{
  for(const body of [{},{client_id:'../bad'},{client_id:null}])assert.throws(()=>requestSpec({op:'heartbeat',job:'job',body}));
  const spec=requestSpec({op:'heartbeat',job:'job',body:{client_id:'client'}});
  assert.equal(spec.path,'/v1/jobs/job/heartbeat');assert.equal(spec.method,'POST');
});
test('session renews its own lease and does not send renewals after cancellation',async()=>{
  const calls=[];const s=new Session({rpc:async(op,args)=>{calls.push([op,args]);return {id:'job'};}});
  await s.create({});await s.heartbeat();await s.cancel();await s.heartbeat();
  assert.deepEqual(calls.map(c=>c[0]),['create','heartbeat','cancel']);
  assert.equal(calls[1][1].body.client_id,s.clientId);assert.equal(calls[2][1].body.client_id,s.clientId);
});
