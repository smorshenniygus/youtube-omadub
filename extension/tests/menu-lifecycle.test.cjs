const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function fixture(chrome){
 const source=fs.readFileSync(__dirname+'/../content.js','utf8');
 const h=(tag,attrs,...children)=>({tag,...attrs,children:children.flat(),
  addEventListener(){},setAttribute(){},classList:{toggle(){}}});
 const c={h,chrome,S:{scale:1,target:'ru',style:'plate',pause:true,dual:false},
  LANGS:[['ru','Русский']],icon:()=>'',stop(){},save(){},shown:[],
  translate:async()=>'',clamp:(n,a,b)=>Math.min(b,Math.max(a,n)),trackInfo:()=>['English','']};
 vm.createContext(c);
 vm.runInContext(source.slice(source.indexOf('  function buildMenu()'),source.indexOf('  function buildUI()'))+
  '\nglobalThis.build=buildMenu;',c);
 return c;
}
function text(node){return typeof node==='string'?node:(node?.children||[]).map(text).join(' ');}

test('real menu displays the installed extension version',()=>{
 const c=fixture({runtime:{getManifest:()=>({version:'1.3.7'})}});
 const menu=c.build();menu.sync();assert.match(text(menu),/VoxDub 1\.3\.7/);
});
test('menu rebuild survives runtime removal after extension reload',()=>{
 const c=fixture({runtime:{getManifest:()=>({version:'1.3.7'})}});c.build();
 c.chrome={};const menu=c.build();menu.sync();assert.match(text(menu),/обнови страницу/);
});
test('menu tolerates missing Chrome API or invalidated getManifest',()=>{
 for(const chrome of [undefined,{}, {runtime:{}},
  {runtime:{getManifest(){throw Error('Extension context invalidated.');}}}]){
  const c=fixture(chrome);const menu=c.build();menu.sync();
  assert.match(text(menu),/VoxDub/);assert.match(text(menu),/обнови страницу/);
 }
});
