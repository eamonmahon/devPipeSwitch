import os

class RunRemoteRepo:
    def __init__(self, server, branch):
        print("rrr initialised")
        self.server = server
        self.branch = branch

    def __enter__(self):
        print("starting enter")
        os.system("ssh %s 'git clone --quiet --branch %s https://github.com/eamonmahon/PipeSwitch.git'" % (self.server['id'], self.branch))
        return self

    def __exit__(self, *args, **kwargs):
        print("starting exit")
        os.system("ssh %s 'rm -rf ~/PipeSwitch'" % self.server['id'])

    def run(self, cmd):
        print("starting run")
        print("server: " + self.server['id'])
        print("command: " + cmd)
        # equivalent to "ssh whitebox 'bash ~/PipeSwitch/scripts/environment/server_run_warmup.sh'""
        os.system("ssh %s '%s'" % (self.server['id'], cmd))
        print("after run")

class RunDocker:
    def __init__(self, image, branch):
        self.image = image
        self.name = 'pipeswitch-%s' % branch
        self.branch = branch
        print("Initialising RunDocker\nimage: " + self.image + "\nname: " + self.name + "\nbranch: " + self.branch)

    def __enter__(self):
        os.system('docker run --name %s --rm -it -d --gpus all -w /workspace %s bash' % (self.name, self.image))
        self.run('git clone --quiet --branch %s https://github.com/eamonmahon/PipeSwitch.git' % self.branch)
        return self

    def __exit__(self, *args, **kwargs):
        os.system('docker stop %s' % self.name)

    def run(self, cmd):
        print("Running docker with command: " + cmd)
        os.system('docker exec -w /workspace %s %s' % (self.name, cmd))

def import_server_list(path):
    server_list = []
    with open(path) as f:
        for line in f.readlines():
            parts = line.split(',')
            ser_ip_str = parts[0].strip()
            ser_name = parts[1].strip()
            server_list.append({'ip': ser_ip_str, 'id': ser_name})
    return server_list