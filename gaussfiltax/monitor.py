"""
Live experiment monitor: a small web server showing the `main.py trials` jobs that are currently running.

Reads what the runner already writes (per-trial `out.log`, `results.pkl` and `runtime.txt` on completion),
the running `main.py` processes and the tmux panes of all tmux servers, and serves a self-refreshing,
phone-friendly page. Finished experiments get a button that runs `main.py errors` and links to the
`Error summary.txt` that it writes.

Usage:
    python gaussfiltax/monitor.py [--port 8001] [--active-minutes 30] [--grace-minutes 15]
    python gaussfiltax/monitor.py --watch [--port 8001]    # start the monitor whenever a job is running
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)    # the modules of gaussfiltax use bare imports

REPORT_DIR = os.path.join(os.path.dirname(HERE), 'reports')
WORK_DIR = os.path.join(REPORT_DIR, '.work')
CLOSED_FILE = os.path.join(WORK_DIR, 'monitor_closed.json')
JAXENV_PY = os.path.expanduser('~/miniforge3/envs/jaxenv/bin/python')
TMUX_DIR = f'/private/tmp/tmux-{os.getuid()}'
SUMMARY = 'Error summary.txt'
SCAN_TTL = 5

PYTHON = JAXENV_PY if os.path.exists(JAXENV_PY) else sys.executable
REG = {}            # key -> what is needed to evaluate that experiment (filled by the scan)
REPORT_JOBS = {}    # key -> {'stage': 'queued'|'errors'|'error', 'msg': str}
JOB_SEEN = {}       # job -> {'first': t, 'last_alive': t}; lets a finished job stay on the page for a while
_scan_cache = {}    # job -> (time, rows)
_report_lock = threading.Lock()   # one evaluation at a time; they are CPU-heavy
_ed = None


def _exp_root():
    import misc
    return os.path.realpath(os.path.join(misc.get_root(), 'experiments'))


EXP_ROOT = None     # set at start-up (needs the heavy imports of misc)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ''


def _cwd(pid):
    for line in _run(['lsof', '-a', '-p', pid, '-d', 'cwd', '-Fn']).splitlines():
        if line.startswith('n'):
            return line[1:]
    return os.path.dirname(HERE)


def split_files(tokens, cwd):
    """File names from the split command line: ps drops the quotes, so join tokens while the path exists."""
    files, i = [], 0
    while i < len(tokens) and not tokens[i].startswith('--'):
        for j in range(len(tokens), i, -1):
            name = ' '.join(tokens[i:j])
            if os.path.isfile(os.path.join(cwd, name)):
                files.append(name)
                i = j
                break
        else:
            files.append(tokens[i])
            i += 1
    return files


def get_processes():
    """Running `main.py trials|errors` python processes; `trials` jobs also carry (seed, start, end, files, cwd)."""
    rows = []
    for line in _run(['ps', '-axo', 'pid=,etime=,command=']).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, cmd = parts
        if 'python' in cmd.split()[0].lower() and re.search(r'main\.py (trials|errors)\b', cmd):
            args = cmd.split('main.py', 1)[1].split()
            row = {'pid': pid, 'etime': etime, 'cmd': ' '.join(args), 'job': None}
            if args[0] == 'trials' and len(args) >= 5 and args[2].isdigit() and args[3].isdigit():
                cwd = _cwd(pid)
                row['job'] = (args[1], int(args[2]), int(args[3]), tuple(split_files(args[4:], cwd)), cwd)
            rows.append(row)
    return rows


def get_tmux(lines=12):
    tmux = shutil.which('tmux') or next((p for p in ('/usr/local/bin/tmux', '/opt/homebrew/bin/tmux') if os.path.exists(p)), 'tmux')
    out = []
    if not os.path.isdir(TMUX_DIR):
        return out
    for sock in sorted(os.listdir(TMUX_DIR)):
        panes = _run([tmux, '-L', sock, 'list-panes', '-a', '-F', '#{session_name}:#{window_index}.#{pane_index}'])
        for target in panes.split():
            text = _run([tmux, '-L', sock, 'capture-pane', '-p', '-t', target, '-S', f'-{lines * 3}'])
            tail = [ln for ln in text.splitlines() if ln.strip()][-lines:]
            out.append({'server': sock, 'pane': target, 'tail': '\n'.join(tail)})
    return out


def describe(desc):
    """(main line, simulator tag) readable summary of an experiment descriptor."""
    inf, sim = desc.inf, desc.sim
    names = {'lgsf': 'L-GSF', 'ugsf': 'U-GSF', 'lagsf': 'L-AGSF', 'uagsf': 'U-AGSF', 'bpf': 'BPF', 'apf': 'APF'}
    label = names.get(inf.name, inf.name)
    if inf.name in ('lgsf', 'ugsf'):
        label += f' · {inf.n_comps} components · {inf.n_iters} iters'
    elif inf.name in ('lagsf', 'uagsf'):
        label += f' · M {inf.M} · N {inf.N} · L {inf.L} · {inf.n_iters} iters'
    elif inf.name in ('bpf', 'apf'):
        label += f' · {inf.num_prt} particles · ess {inf.ess}'
    tag = f'{sim.name} · state {sim.state_dim} · obs {sim.emission_dim} · T={sim.num_timesteps}'
    if sim.name == 'covinfl':
        tag += f' · a {sim.a} · b {sim.b}'
    return label, tag


def exp_list(files, cwd):
    """Descriptors of a job in the order `main.py trials` runs them."""
    global _ed
    if _ed is None:
        import experiment_descriptor as _ed
    descs = []
    for f in files:
        with open(os.path.join(cwd, f)) as fh:
            descs += _ed.parse(fh.read())
    return descs


def read_runtime(path):
    """Seconds from runtime.txt ('Runtime: X seconds'), or None."""
    try:
        with open(path) as f:
            return float(re.search(r'Runtime:\s*([0-9.eE+-]+)', f.read()).group(1))
    except (OSError, AttributeError, ValueError):
        return None


# ---------------------------------------------------------------- evaluation

def _report_key(variant, start, end):
    return f"{hashlib.sha1(variant.encode()).hexdigest()[:12]}_{start}_{end}"


def _summary_path(variant):
    return os.path.join(EXP_ROOT, variant, SUMMARY)


def _run_logged(cmd, log):
    with open(log, 'w') as f:
        env = dict(os.environ, MPLBACKEND='Agg')
        return subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def _ensure_work_dir():
    os.makedirs(WORK_DIR, exist_ok=True)
    gi = os.path.join(REPORT_DIR, '.gitignore')
    if not os.path.exists(gi):
        with open(gi, 'w') as f:
            f.write('*\n')


def _report_worker(key):
    r = REG[key]
    _ensure_work_dir()
    desc = os.path.join(WORK_DIR, key + '.txt')
    with open(desc, 'w') as f:
        f.write(r['desc'])
    with _report_lock:
        try:
            REPORT_JOBS[key] = {'stage': 'errors', 'msg': 'evaluating errors'}
            cmd = [PYTHON, 'main.py', 'errors', str(r['start']), str(r['end']), desc, '--no-plots']
            if _run_logged(cmd, os.path.join(WORK_DIR, key + '.errors.log')):
                raise RuntimeError('errors evaluation failed, see reports/.work/%s.errors.log' % key)
            REPORT_JOBS.pop(key, None)
        except Exception as e:
            REPORT_JOBS[key] = {'stage': 'error', 'msg': str(e)}


def start_report(key):
    """Starts `main.py errors` for one experiment; returns False if unknown or already running."""
    if key not in REG or REPORT_JOBS.get(key, {}).get('stage') in ('errors', 'queued'):
        return False
    REPORT_JOBS[key] = {'stage': 'queued', 'msg': 'waiting for another evaluation to finish' if _report_lock.locked() else 'starting'}
    threading.Thread(target=_report_worker, args=(key,), daemon=True).start()
    return True


def read_summary(key):
    r = REG.get(key)
    path = _summary_path(r['variant']) if r else ''
    if not r or not os.path.isfile(path):
        return None
    with open(path, errors='replace') as f:
        return f.read()


# ---------------------------------------------------------------- scanning

def scan_job(job, alive, active_s):
    """Rows (one per experiment, in file order) with progress through the job's trial range."""
    cached = _scan_cache.get(job)
    if cached and time.time() - cached[0] < SCAN_TTL:
        return cached[1]
    now = time.time()
    _, start, end, files, cwd = job
    try:
        descs = exp_list(files, cwd)
    except Exception:
        return []
    rows = []
    for desc in descs:
        variant = desc.get_dir()
        exp_dir = os.path.join(EXP_ROOT, variant)
        done, running, times, log, last = 0, [], [], None, 0
        for trial in range(start, end + 1):
            tdir = os.path.join(exp_dir, str(trial))
            if not os.path.isdir(tdir):
                continue
            lp = os.path.join(tdir, 'out.log')
            if os.path.exists(os.path.join(tdir, 'results.pkl')):
                done += 1
                t = read_runtime(os.path.join(tdir, 'runtime.txt'))
                if t is not None:
                    times.append(t)
            elif alive and now - max(os.path.getmtime(tdir), os.path.getmtime(lp) if os.path.exists(lp) else 0) < active_s:
                running.append(str(trial))
            if os.path.exists(lp) and os.path.getmtime(lp) > last:
                last, log = os.path.getmtime(lp), os.path.relpath(lp, EXP_ROOT)
        total = end - start + 1
        key = _report_key(variant, start, end)
        REG[key] = {'variant': variant, 'start': start, 'end': end, 'desc': desc.pprint()}
        mean = sum(times) / len(times) if times else None
        label, tag = describe(desc)
        state = 'running' if running else 'done' if done >= total else 'queued' if alive else 'stopped'
        rows.append({'key': key, 'label': label, 'tag': tag, 'desc': desc.pprint(), 'state': state, 'running': running,
                     'done': done, 'total': total, 'log': log, 'mean_time': mean,
                     'eta': mean * (total - done) if mean else None, 'age': now - last if last else None})
    _scan_cache[job] = (time.time(), rows)
    return rows


def status(active_s, grace_s):
    now = time.time()
    procs = get_processes()
    alive = {}
    for p in procs:
        if p['job']:
            alive.setdefault(p['job'], []).append(p)
    for job in alive:
        JOB_SEEN.setdefault(job, {'first': now})['last_alive'] = now
    for job in [j for j, v in JOB_SEEN.items() if j not in alive and now - v['last_alive'] > grace_s]:
        JOB_SEEN.pop(job)
        _scan_cache.pop(job, None)
    jobs = []
    for job, seen in sorted(JOB_SEEN.items(), key=lambda kv: kv[1]['first']):
        seed, start, end, files, _ = job
        rows = [dict(e, summary=os.path.exists(_summary_path(REG[e['key']]['variant'])), job=REPORT_JOBS.get(e['key']))
                for e in scan_job(job, job in alive, active_s)]
        jobs.append({'title': ' + '.join(os.path.basename(f) for f in files), 'seed': seed, 'start': start, 'end': end,
                     'alive': job in alive, 'pids': [p['pid'] for p in alive.get(job, [])],
                     'etime': alive[job][0]['etime'] if job in alive else None,
                     'ended_ago': None if job in alive else now - seen['last_alive'],
                     'experiments': rows, 'n_done': sum(e['state'] == 'done' for e in rows)})
    return {'time': time.strftime('%H:%M:%S'), 'processes': procs, 'jobs': jobs, 'tmux': get_tmux()}


def tail_log(rel, n=60):
    path = os.path.realpath(os.path.join(EXP_ROOT, rel))
    if not path.startswith(EXP_ROOT + os.sep) or os.path.basename(path) != 'out.log' or not os.path.isfile(path):
        return 'not found'
    with open(path, errors='replace') as f:
        return ''.join(f.readlines()[-n:])


# ---------------------------------------------------------------- page

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment monitor</title>
<style>
:root{--bg:#14161c;--fg:#e4e6eb;--mut:#8b90a0;--card:#1c1f27;--line:#343843;--ok:#5fd38d;--run:#5aa9ff;--bad:#ff6b6b}
@media(prefers-color-scheme:light){:root{--bg:#fafafa;--fg:#1c1e24;--mut:#666;--card:#fff;--line:#d9dbe1}}
*{box-sizing:border-box}body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,system-ui,sans-serif}
h1{font-size:18px;margin:0 0 2px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:20px 0 8px}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;flex-wrap:wrap}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:8px}
.card.click{cursor:pointer}.name{font-weight:600}.tag{font-size:12px;color:var(--mut)}
.job{margin:16px 0 8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.job b{word-break:break-all}
.pill{display:inline-block;border-radius:10px;padding:0 8px;font-size:12px;margin-right:4px;border:1px solid var(--line)}
.run{color:var(--run);border-color:var(--run)}.ok{color:var(--ok)}.bad{color:var(--bad);border-color:var(--bad)}
pre{margin:8px 0 0;font:12px/1.35 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-all;max-height:280px;overflow:auto}
.bar{height:6px;border-radius:3px;background:var(--line);margin:8px 0}.bar>div{height:100%;border-radius:3px;background:var(--run);transition:width .5s}
.card.queued,.card.stopped{opacity:.55}.card.done .bar>div{background:var(--ok)}
.act{margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{font:inherit;font-size:13px;padding:6px 12px;border-radius:6px;border:1px solid var(--run);background:transparent;color:var(--run);cursor:pointer;text-decoration:none}
.btn.go{background:var(--run);color:#fff}.btn.bad{color:var(--bad);border-color:var(--bad)}.btn.ghost{border-color:var(--line);color:var(--mut)}.btn:disabled{opacity:.6;cursor:default}
a{color:var(--run)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--run);margin-right:6px;animation:p 1.2s infinite}
@keyframes p{50%{opacity:.25}}
</style></head><body>
<div class="top"><div><h1><span class="dot"></span>Experiment monitor</h1><div class="mut" id="upd">loading…</div></div>
<div class="act" style="margin:0"><button class="btn bad" onclick="stopMon()">Stop monitor</button></div></div>
<h2>Running processes</h2><div id="procs"></div>
<h2>Experiments of running jobs</h2><div id="exps"></div>
<h2>tmux panes</h2><div id="tmux"></div>
<script>
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=s=>s==null?'–':s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+'m':(s/3600).toFixed(1)+'h';
let open=null, logs={}, last={};
function set(el,html){if(last[el.id]!==html){last[el.id]=html;el.innerHTML=html}}   // keep tooltips/selection alive between refreshes
function actions(e){
  const j=e.job, k=esc(e.key);
  if(j&&j.stage=='error') return `<div class="act"><button class="btn bad" onclick="gen(event,'${k}')">Failed – retry</button><span class="mut">${esc(j.msg)}</span></div>`;
  if(j) return `<div class="act"><button class="btn" disabled>${j.stage=='errors'?'Evaluating errors…':'Queued…'}</button><span class="mut">${esc(j.msg)}</span></div>`;
  if(e.summary) return `<div class="act"><a class="btn go" href="summary?key=${k}" target="_blank" onclick="event.stopPropagation()">Open error summary</a><button class="btn ghost" onclick="gen(event,'${k}')">Re-evaluate</button></div>`;
  return `<div class="act"><button class="btn" onclick="gen(event,'${k}')">Evaluate errors</button></div>`;
}
async function gen(ev,key){ev.stopPropagation();ev.target.disabled=true;await fetch('api/report?key='+encodeURIComponent(key),{method:'POST'});await tick()}
async function showLog(p){open=open===p?null:p;await tick()}
async function stopMon(){
  if(!confirm('Stop the monitor server? It restarts automatically when a new job starts. Running experiments are not affected.')) return;
  try{await fetch('api/shutdown',{method:'POST'})}catch(e){}
  clearInterval(timer);document.body.innerHTML='<p style="padding:16px">Monitor stopped.</p>';
}
function card(e){
  const pc=Math.round(100*e.done/e.total);
  const badge=e.state=='running'?`<span class="pill run">running: trial ${esc(e.running.join(', '))}</span>`:e.state=='queued'?'<span class="pill">queued</span>':e.state=='stopped'?'<span class="pill bad">not run</span>':'<span class="pill ok">finished</span>';
  return `<div class="card ${e.log?'click':''} ${e.state}" title="${esc(e.desc)}" ${e.log?`onclick="showLog('${esc(e.log)}')"`:''}>
    <div class="name">${esc(e.label)}</div><div class="tag">${esc(e.tag)}</div>
    <div class="bar"><div style="width:${pc}%"></div></div>
    <div>${badge}<span class="pill ${e.done?'ok':''}">${e.done}/${e.total} trials</span>
    <span class="mut">${e.mean_time?`mean ${fmt(e.mean_time)}/trial`:''}${e.state=='running'?` · ETA ${fmt(e.eta)} · last log ${fmt(e.age)} ago`:''}</span></div>
    ${e.state=='done'?actions(e):''}
    ${open===e.log?`<pre>${esc(logs[e.log]||'')}</pre>`:''}</div>`;
}
async function tick(){
  try{
    const d=await (await fetch('api/status')).json();
    upd.textContent='updated '+d.time+' · refreshes every 3s';
    set(procs,d.processes.length?d.processes.map(p=>`<div class="card"><span class="pill run">pid ${p.pid}</span><span class="mut">up ${esc(p.etime)}</span><pre>${esc(p.cmd)}</pre></div>`).join(''):'<div class="mut">none</div>');
    if(open) logs[open]=await (await fetch('api/log?path='+encodeURIComponent(open))).text();
    set(exps,d.jobs.map(j=>`<div class="job"><b>${esc(j.title)}</b><span class="mut">seed ${esc(j.seed)} · trials ${j.start}–${j.end}</span>
      <span class="pill ${j.alive?'run':''}">${j.n_done}/${j.experiments.length} experiments</span>
      <span class="mut">${j.alive?'pid '+esc(j.pids.join(', '))+' · up '+esc(j.etime):'job ended '+fmt(j.ended_ago)+' ago'}</span></div>`+j.experiments.map(card).join('')).join('')||'<div class="mut">no job running</div>');
    set(tmux,d.tmux.map(t=>`<div class="card"><span class="pill">${esc(t.server)}</span><span class="mut">${esc(t.pane)}</span><pre>${esc(t.tail)}</pre></div>`).join('')||'<div class="mut">no tmux servers</div>');
  }catch(e){upd.textContent='disconnected – retrying…'}
}
tick();const timer=setInterval(tick,3000);
</script></body></html>"""


def shutdown_monitor():
    """Close the server; the watcher will not restart it for the jobs that are running now."""
    _ensure_work_dir()
    with open(CLOSED_FILE, 'w') as f:
        json.dump({'pids': [p['pid'] for p in get_processes() if p['job']]}, f)
    threading.Timer(0.5, lambda: os._exit(0)).start()


def make_handler(active_s, grace_s):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype, code=200):
            data = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/':
                self._send(PAGE, 'text/html; charset=utf-8')
            elif url.path == '/api/status':
                self._send(json.dumps(status(active_s, grace_s)), 'application/json')
            elif url.path == '/summary':
                text = read_summary(parse_qs(url.query).get('key', [''])[0])
                if text is None:
                    self.send_error(404)
                else:
                    self._send(text, 'text/plain; charset=utf-8')
            elif url.path == '/api/log':
                self._send(tail_log(parse_qs(url.query).get('path', [''])[0]), 'text/plain; charset=utf-8')
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == '/api/report':
                ok = start_report(parse_qs(url.query).get('key', [''])[0])
                self._send(json.dumps({'started': ok}), 'application/json')
            elif url.path == '/api/shutdown':
                self._send(json.dumps({'stopping': True}), 'application/json')
                shutdown_monitor()
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return Handler


# ---------------------------------------------------------------- watcher

def _listening(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


def watch(port, args, every=10):
    """Keeps a monitor up while jobs run. A job that was running when the monitor was closed does not restart it."""
    print(f'Watching for running jobs; monitor on port {port}', flush=True)
    while True:
        pids = {p['pid'] for p in get_processes() if p['job']}
        if not pids:
            if os.path.exists(CLOSED_FILE):
                os.remove(CLOSED_FILE)
        else:
            try:
                with open(CLOSED_FILE) as f:
                    closed = set(json.load(f)['pids'])
            except (OSError, ValueError, KeyError):
                closed = set()
            if pids - closed and not _listening(port):
                print(time.strftime('%H:%M:%S'), 'job running, starting monitor', flush=True)
                subprocess.Popen([PYTHON,os.path.abspath(__file__), '--port', str(port),
                                  '--active-minutes', str(args.active_minutes), '--grace-minutes', str(args.grace_minutes)],
                                 cwd=HERE, start_new_session=True, stdin=subprocess.DEVNULL)
        time.sleep(every)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='live experiment monitor')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--active-minutes', type=float, default=30,
                        help='a trial with no log output for longer than this is no longer shown as running')
    parser.add_argument('--grace-minutes', type=float, default=15,
                        help='how long a finished job stays on the page')
    parser.add_argument('--watch', action='store_true', help='start the monitor whenever a job is running')
    args = parser.parse_args()
    if args.watch:
        watch(args.port, args)
    else:
        EXP_ROOT = _exp_root()
        print(f'Monitoring {EXP_ROOT} on port {args.port}', flush=True)
        ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(args.active_minutes * 60, args.grace_minutes * 60)).serve_forever()
