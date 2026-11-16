"""Temporal report comparisons and controls, without requiring a browser."""
import shutil
import subprocess

import pytest

from fx_forecasting.evaluation.clustering_html import RUNTIME


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript checks require Node")
def test_temporal_comparisons_masks_and_window_controls(tmp_path):
    # Use the actual report functions with a small, deliberately changing panel.
    functions = RUNTIME[RUNTIME.index("function evolutionEndpoint()"):
                        RUNTIME.index("function update()")]
    script = r"""
const assert=require('node:assert/strict');
const dates=Array.from({length:180},(_,t)=>new Date(Date.UTC(2000,t,1)).toISOString().slice(0,7));
const path=dates.map((_,t)=>t<60?[1,0,0,-1]:t<120?[1,1,0,0]:[1,t<150?1:0,0,t<174?0:-1]);
const D={countryNames:{AAA:'A',BBB:'B',CCC:'C',DDD:'D'},evolution:{dates,
 currencies:['AAA','BBB','CCC','DDD'],names:{r2k_w6:'R2K'},labels:{'r2k_w6:k2':path}}};
const elements={},figures={},window={innerHeight:850};
function byId(id){return elements[id]??=( {value:'',clientWidth:900,on(){}} );}
for(const [id,value]of Object.entries({'evolution-method':'r2k_w6','evolution-k':'2',
 'evolution-end':'179','evolution-currency':'AAA','evolution-rank':'closest',
 'evolution-coverage':'48','evolution-matrix-view':'frequency'}))byId(id).value=value;
const letter=n=>String.fromCharCode(65+n),currencyCards=()=>'',htmlEscape=x=>x;
const matrixCellColor=()=>'',colors=['blue','orange'],correlationColors=[],frequencyColors=[];
function draw(id,data,layout,config){figures[id]={data,layout,config};return Promise.resolve();}
""" + functions + r"""
evolution();
const cs=D.evolution.currencies;
function cell(a,b){const t=figures['evolution-matrix'].data[0];return t.z[t.y.indexOf(a)][t.x.indexOf(b)];}
assert.deepEqual(figures['evolution-matrix'].data[0].x,['BBB','CCC','AAA','DDD']);
assert.equal(cell('AAA','BBB'),.5);
assert.equal(cell('AAA','AAA'),null);
assert.equal(cell('CCC','DDD'),1); // 54/54, missing months are not disagreements.
assert.equal(figures['evolution-timeline'].config.edits.shapePosition,true);
assert.deepEqual(figures['evolution-timeline'].data[0].y,cs); // fixed timeline identities
let bars=figures['evolution-companion-chart'].data;
assert.deepEqual(bars[0].y,['BBB','CCC','DDD']);
assert.deepEqual(bars[0].x,[.5,0,0]);
assert.deepEqual(bars[1].x,[1,0,0]);
byId('evolution-matrix-view').value='change';evolution();
assert.equal(cell('AAA','BBB'),-.5);
assert.equal(cell('BBB','CCC'),.5);
assert.equal(cell('CCC','DDD'),0);
byId('evolution-coverage').value='60';evolution();
assert.equal(cell('CCC','DDD'),null);
byId('evolution-coverage').value='48';byId('evolution-rank').value='losses';evolution();
assert.equal(figures['evolution-companion-chart'].data[0].y[0],'BBB');
byId('evolution-matrix-view').value='persistent';evolution();
assert.equal(cell('AAA','BBB'),null);assert.equal(cell('CCC','DDD'),1);
setEvolutionEndpoint(59);
assert.equal(byId('evolution-month').value,'2004-12');
assert.equal(byId('evolution-prev-month').disabled,true);
byId('evolution-rank').value='closest';byId('evolution-matrix-view').value='change';evolution();
assert(figures['evolution-companion-chart'].data[1].x.every(x=>x===null));
assert(figures['evolution-matrix'].data[0].z.flat().every(x=>x===null));
assert(byId('evolution-comparison').textContent.includes('no complete preceding'));
handleEvolutionClick({points:[{x:'2010-06-01',y:'CCC'}]});
assert.equal(byId('evolution-currency').value,'CCC');assert.equal(byId('evolution-month').value,'2010-06');
handleEvolutionRelayout({'shapes[0].x0':'2000-01-01','shapes[0].x1':'2005-01-01'});
assert.equal(byId('evolution-month').value,'2004-12');
handleEvolutionRelayout({'shapes[0].x1':'2004-12-31 23:59:59'});
assert.equal(byId('evolution-month').value,'2004-12'); // pixel rounding must not drift
handleEvolutionRelayout({'shapes[0].x0':'2010-01-01'});
assert.equal(byId('evolution-month').value,'2014-12');
handleEvolutionRelayout({'shapes[0].y0':.3});assert.equal(byId('evolution-end').value,'179');
assert.equal(figures['evolution-timeline'].layout.shapes[0].y0,0);
handleEvolutionRelayout({'shapes[0].x1':'1900-01-01'});
assert.equal(byId('evolution-end').value,'59');
handleEvolutionRelayout({'shapes[0].x1':'2050-01-01'});
assert.equal(byId('evolution-end').value,'179');
assert.equal(byId('evolution-next-month').disabled,true);
"""
    path = tmp_path / "temporal_report.cjs"
    path.write_text(script)
    result = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
