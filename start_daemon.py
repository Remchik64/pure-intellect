import os, sys, time, subprocess
subprocess.run("pkill -9 -f 'pure_intellect serve'", shell=True)
time.sleep(2)
pid = os.fork()
if pid > 0:
    sys.exit(0)
os.setsid()
pid2 = os.fork()
if pid2 > 0:
    sys.exit(0)
os.chdir('/a0/usr/workdir/pure-intellect')
with open('server_daemon.log', 'w') as f:
    os.dup2(f.fileno(), sys.stdout.fileno())
    os.dup2(f.fileno(), sys.stderr.fileno())
os.execv('/a0/usr/workdir/pure-intellect/venv/bin/python', ['python', '-m', 'pure_intellect', 'serve', '--port', '3005', '--host', '0.0.0.0'])
