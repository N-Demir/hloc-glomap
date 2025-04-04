"""
Sets up an SSH server in a Modal container.

This requires you to `pip install sshtunnel` locally.

After running this with `modal run launch_server.py`, connect to SSH with `ssh -p 9090 root@localhost`,
or from VSCode/Pycharm.

This uses simple password authentication, but you can store your own key in a modal Secret instead.
"""
import modal
import threading
import socket
import subprocess
import time

app = modal.App(
    "example-get-started",
    image=modal.Image.from_dockerfile("Dockerfile")
    .apt_install("openssh-server")
    .run_commands(
        "mkdir -p /run/sshd",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "echo 'root: ' | chpasswd"
    )
    .workdir("/root")
    .run_commands(
        "git clone https://github.com/N-Demir/hloc-glomap.git"
    )
    .workdir("/root/hloc-glomap")
    .run_commands(
        "pixi install",
        "pixi run post-install",
        gpu="T4"
    )
)

LOCAL_PORT = 9090

def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
        q.put((host, port))

# Could add a volume but not sure what I would be using it for
# volumes={"/root/workspace": modal.Volume.from_name("modal-server", create_if_missing=True)}
@app.function(timeout=3600 * 24, volumes={"/root/.cursor-server": modal.Volume.from_name("cursor-server", create_if_missing=True)}, gpu="T4")
def launch_ssh(q):
    subprocess.run(["git", "pull"])

    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()

        subprocess.run(["/usr/sbin/sshd", "-D"]) # TODO: I don't know why I need to start this here

@app.local_entrypoint()
def main():
    import sshtunnel

    with modal.Queue.ephemeral() as q:
        launch_ssh.spawn(q)
        host, port = q.get()
        print(f"SSH server running at {host}:{port}")

        server = sshtunnel.SSHTunnelForwarder(
            (host, port),
            ssh_username="root",
            ssh_password=" ",
            remote_bind_address=('127.0.0.1', 22),
            local_bind_address=('127.0.0.1', LOCAL_PORT),
            allow_agent=False
        )

        try:
            server.start()
            print(f"SSH tunnel forwarded to localhost:{server.local_bind_port}")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down SSH tunnel...")
        finally:
            server.stop()
